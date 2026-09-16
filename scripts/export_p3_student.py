#!/usr/bin/env python3
"""Export the Phase-3 LSTM student from an rsl_rl checkpoint into a deploy bundle.

The checkpoint is a full `Phase3StudentTeacher` (teacher MLP + student LSTM + two
auxiliary heads). Deployment only needs the student path, and that path is fully
determined by the state_dict:

    obs(49) --x scale--> LSTM(49 -> 256) --> MLP [512, 256, 128] elu --> action(12)

so this runs on plain torch — no Isaac Sim, no Isaac Lab, no GPU, no env.

Outputs (default: <checkpoint dir>/exported/):
    policy.pt          TorchScript, (obs, h_in, c_in) -> (actions, h_out, c_out)
    policy.onnx        same graph, opset 13
    policy_numpy.npz   pure-NumPy bundle for the onboard NX (no torch/onnxruntime)
    aux_heads.npz      splint_head / vel_head, applied to h_out
    policy_io.json     observation layout + bundle metadata
    reference_io.json  20-step reference episode from zero hidden state

The obs scale (train.normalize, joint_vel x 0.23) lives INSIDE every artifact:
baked into the graph for .pt/.onnx, folded into lstm_weight_ih for the .npz. All
three therefore consume the same RAW 49-dim observation.

    python3 scripts/export_p3_student.py \
        --checkpoint sdk_deploy/model/P3-final/model_3999.pt

Needs torch (and onnx for the .onnx). Run it with the isaac env interpreter:
    /home/shw/miniconda3/envs/isaac/bin/python
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Observation layout — go1_lod mdp/obs_normalizer.py is the ground truth.
# Keep the two in sync; a mismatch here is silent and lands on the hardware.
# ---------------------------------------------------------------------------
OBS_LAYOUT = [
    ("base_ang_vel", 3),        # [ 0: 3] IMU gyro
    ("projected_gravity", 3),   # [ 3: 6] R^T [0,0,-1]
    ("velocity_commands", 3),   # [ 6: 9] (vx, vy, wz)
    ("joint_pos_rel", 12),      # [ 9:21] q - default
    ("joint_vel_rel", 12),      # [21:33] dq          (x 0.23 by the obs scale)
    ("last_actions", 12),       # [33:45] previous raw policy output
    ("peg_leg_one_hot", 4),     # [45:49] injured leg (FL, FR, RL, RR); healthy = 0
]
OBS_DIM = sum(d for _, d in OBS_LAYOUT)
ACTION_DIM = 12
HIDDEN = 256
MLP_DIMS = [512, 256, 128]

# nn.LSTM packs gates as [input, forget, cell, output]; the NumPy backend in
# sdk_deploy/policy.py assumes that order.
TORCH_GATE_ORDER = "ifgo"


class Phase3StudentDeploy(nn.Module):
    """The student's inference path, standalone and scriptable.

    rsl_rl's Memory.forward in inference mode is `rnn(input.unsqueeze(0), hidden)`
    (networks/memory.py), i.e. one timestep of a plain seq-first nn.LSTM — so this
    is an exact reconstruction, not an approximation.
    """

    def __init__(self, sd):
        super().__init__()
        self.register_buffer("obs_scale", sd["student_obs_normalizer.scale"].clone())
        self.rnn = nn.LSTM(OBS_DIM, HIDDEN, 1)
        self.rnn.load_state_dict(_sub(sd, "memory_s.rnn."))

        layers = []
        prev = HIDDEN
        for h in MLP_DIMS:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, ACTION_DIM))
        self.mlp = nn.Sequential(*layers)
        self.mlp.load_state_dict(_sub(sd, "student."))

    def forward(self, obs, h_in, c_in):
        out, (h_out, c_out) = self.rnn((obs * self.obs_scale).unsqueeze(0), (h_in, c_in))
        return self.mlp(out.squeeze(0)), h_out, c_out


def _sub(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def _load_state_dict(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)

    required = ["student_obs_normalizer.scale", "memory_s.rnn.weight_ih_l0",
                "student.0.weight", "splint_head.weight", "vel_head.weight"]
    missing = [k for k in required if k not in sd]
    if missing:
        sys.exit(f"checkpoint is missing {missing} — is this a Phase-3 student?")

    got = tuple(sd["memory_s.rnn.weight_ih_l0"].shape)
    if got != (4 * HIDDEN, OBS_DIM):
        sys.exit(f"LSTM input weight {got} != {(4 * HIDDEN, OBS_DIM)}; the observation "
                 f"layout changed — update OBS_LAYOUT (and go1_lod obs_normalizer.py).")

    scale = sd["student_obs_normalizer.scale"]
    if scale.numel() != OBS_DIM:
        sys.exit(f"obs scale has {scale.numel()} entries, expected {OBS_DIM}.")
    if bool(torch.allclose(scale, torch.ones_like(scale))):
        print("  warning: obs scale is all ones — was train.normalize disabled?")
    return sd, int(ckpt.get("iter", -1))


# ---------------------------------------------------------------------------
# NumPy bundle — written from the checkpoint directly, not re-parsed from ONNX.
# ---------------------------------------------------------------------------

def _numpy_bundle(model):
    """Weight bundle for the on-robot NumPy backend.

    The obs scale is folded into lstm_weight_ih: W @ (x * s) == (W * s) @ x, exactly.
    That keeps the backend a plain matmul loop with no extra step to forget, and
    makes the .npz consume the same raw obs as the .pt / .onnx.
    """
    scale = model.obs_scale.detach().numpy().astype(np.float32)
    sd = model.rnn.state_dict()

    bundle = {
        "obs_dim": np.int64(OBS_DIM),
        "action_dim": np.int64(ACTION_DIM),
        "arch": np.array(b"lstm_mlp"),
        "gate_order": np.array(TORCH_GATE_ORDER.encode()),
        "lstm_weight_ih": (sd["weight_ih_l0"].numpy() * scale[None, :]).astype(np.float32),
        "lstm_weight_hh": sd["weight_hh_l0"].numpy().astype(np.float32),
        "lstm_bias_ih": sd["bias_ih_l0"].numpy().astype(np.float32),
        "lstm_bias_hh": sd["bias_hh_l0"].numpy().astype(np.float32),
        "lstm_hidden_size": np.int64(HIDDEN),
        "lstm_input_size": np.int64(OBS_DIM),
        # audit trail: the scale is already inside lstm_weight_ih, do NOT re-apply
        "obs_scale": scale,
        "obs_scale_folded": np.int64(1),
    }
    linears = [m for m in model.mlp if isinstance(m, nn.Linear)]
    for i, lin in enumerate(linears):
        bundle[f"{2 * i}_weight"] = lin.weight.detach().numpy().astype(np.float32)
        bundle[f"{2 * i}_bias"] = lin.bias.detach().numpy().astype(np.float32)
    bundle["num_mlp_layers"] = np.int64(len(linears))
    return bundle


def _numpy_forward(bundle, obs, h, c):
    """Reference implementation of the on-robot backend, used to verify the bundle."""
    w_ih, w_hh = bundle["lstm_weight_ih"], bundle["lstm_weight_hh"]
    gates = w_ih @ obs + bundle["lstm_bias_ih"] + w_hh @ h + bundle["lstm_bias_hh"]
    H = HIDDEN
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    i, f, g, o = sig(gates[:H]), sig(gates[H:2 * H]), np.tanh(gates[2 * H:3 * H]), sig(gates[3 * H:])
    c = (f * c + i * g).astype(np.float32)
    h = (o * np.tanh(c)).astype(np.float32)

    x = h
    n = int(bundle["num_mlp_layers"])
    for k in range(n):
        x = x @ bundle[f"{2 * k}_weight"].T + bundle[f"{2 * k}_bias"]
        if k < n - 1:
            x = np.where(x > 0.0, x, np.exp(x) - 1.0)  # ELU
    return x, h, c


def _aux_heads(sd):
    """splint_head / vel_head + their de-normalisation constants.

    Both read the LSTM output h_t, which the exported graph returns as h_out — so
    the robot gets L_hat and v_hat from two small matmuls, no second graph:
        L_hat = (splint_w @ h + splint_b) * splint_std + splint_mean   [m]
        v_hat = (vel_w    @ h + vel_b)    * vel_std    + vel_mean      [m/s]
    """
    f = lambda k: sd[k].detach().numpy().astype(np.float32)
    return {
        "splint_weight": f("splint_head.weight"), "splint_bias": f("splint_head.bias"),
        "vel_weight": f("vel_head.weight"), "vel_bias": f("vel_head.bias"),
        "splint_mean": f("norm_splint_mean"), "splint_std": f("norm_splint_std"),
        "vel_mean": f("norm_vel_mean"), "vel_std": f("norm_vel_std"),
    }


def _reference_obs(n, seed=0):
    """Plausible-magnitude observations for the self-test.

    Deliberately NOT unit-variance: joint_vel is drawn wide (std 3 rad/s) so that a
    build which drops the x0.23 scale fails the reference check instead of passing it.
    """
    rng = np.random.RandomState(seed)
    obs = np.zeros((n, OBS_DIM), dtype=np.float32)
    obs[0] = 0.0  # first step from rest
    for t in range(1, n):
        v = np.concatenate([
            rng.normal(0, 0.5, 3),                          # base_ang_vel
            [rng.normal(0, 0.1), rng.normal(0, 0.1), -1.0], # projected_gravity
            [rng.uniform(0, 1), 0.0, rng.normal(0, 0.2)],   # velocity_commands
            rng.normal(0, 0.2, 12),                         # joint_pos_rel
            rng.normal(0, 3.0, 12),                         # joint_vel_rel  <- wide
            rng.normal(0, 0.5, 12),                         # last_actions
            np.zeros(4),                                    # peg_leg_one_hot
        ])
        obs[t] = v
    # exercise the one-hot on the tail steps (FR injured)
    obs[n // 2:, 46] = 1.0
    return obs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--checkpoint",
                    default=os.path.join(here, "sdk_deploy/model/P3-final/model_3999.pt"))
    ap.add_argument("--out", default=None,
                    help="output directory (default: <checkpoint dir>/exported)")
    ap.add_argument("--name", default=None, help="label recorded in policy_io.json")
    ap.add_argument("--ref-steps", type=int, default=20)
    ap.add_argument("--tol", type=float, default=1e-5,
                    help="max |torch - numpy| allowed before the export is rejected")
    args = ap.parse_args()

    ckpt_path = os.path.abspath(args.checkpoint)
    out_dir = args.out or os.path.join(os.path.dirname(ckpt_path), "exported")
    os.makedirs(out_dir, exist_ok=True)

    print(f"checkpoint : {ckpt_path}")
    sd, it = _load_state_dict(ckpt_path)
    model = Phase3StudentDeploy(sd).eval()
    print(f"  iter={it}, obs={OBS_DIM}, hidden={HIDDEN}, mlp={MLP_DIMS}, action={ACTION_DIM}")
    scale = model.obs_scale.numpy()
    spans = ", ".join(f"{n}={scale[o]:g}" for n, o in
                      zip([n for n, _ in OBS_LAYOUT], np.cumsum([0] + [d for _, d in OBS_LAYOUT])))
    print(f"  obs scale  : {spans}")

    # ---- reference episode (torch is the source of truth) -------------------
    ref_obs = _reference_obs(args.ref_steps)
    h = torch.zeros(1, 1, HIDDEN)
    c = torch.zeros(1, 1, HIDDEN)
    ref_actions, ref_latent = [], []
    with torch.no_grad():
        for o in ref_obs:
            a, h, c = model(torch.from_numpy(o).unsqueeze(0), h, c)
            ref_actions.append(a.squeeze(0).numpy())
            ref_latent.append(h.reshape(-1).numpy())
    ref_actions = np.asarray(ref_actions, dtype=np.float64)

    # ---- artifacts ---------------------------------------------------------
    ts = torch.jit.script(model)
    torch.jit.save(ts, os.path.join(out_dir, "policy.pt"))

    onnx_path = os.path.join(out_dir, "policy.onnx")
    torch.onnx.export(
        model,
        (torch.zeros(1, OBS_DIM), torch.zeros(1, 1, HIDDEN), torch.zeros(1, 1, HIDDEN)),
        onnx_path,
        input_names=["obs", "h_in", "c_in"],
        output_names=["actions", "h_out", "c_out"],
        opset_version=13,
    )

    bundle = _numpy_bundle(model)
    np.savez(os.path.join(out_dir, "policy_numpy.npz"), **bundle)
    np.savez(os.path.join(out_dir, "aux_heads.npz"), **_aux_heads(sd))

    # ---- verify every backend against the torch reference ------------------
    hn = np.zeros(HIDDEN, dtype=np.float32)
    cn = np.zeros(HIDDEN, dtype=np.float32)
    worst_npz = 0.0
    for t, o in enumerate(ref_obs):
        a, hn, cn = _numpy_forward(bundle, o.astype(np.float32), hn, cn)
        worst_npz = max(worst_npz, float(np.abs(a - ref_actions[t]).max()))

    worst_ts = 0.0
    h = torch.zeros(1, 1, HIDDEN)
    c = torch.zeros(1, 1, HIDDEN)
    with torch.no_grad():
        for t, o in enumerate(ref_obs):
            a, h, c = ts(torch.from_numpy(o).unsqueeze(0), h, c)
            worst_ts = max(worst_ts, float(np.abs(a.squeeze(0).numpy() - ref_actions[t]).max()))

    worst_onnx = None
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ho = np.zeros((1, 1, HIDDEN), dtype=np.float32)
        co = np.zeros((1, 1, HIDDEN), dtype=np.float32)
        worst_onnx = 0.0
        for t, o in enumerate(ref_obs):
            a, ho, co = sess.run(None, {"obs": o.reshape(1, -1), "h_in": ho, "c_in": co})
            worst_onnx = max(worst_onnx, float(np.abs(a.reshape(-1) - ref_actions[t]).max()))
    except ImportError:
        print("  note: onnxruntime not installed — .onnx left unverified")

    print(f"  verify     : npz {worst_npz:.2e}, torchscript {worst_ts:.2e}"
          + (f", onnx {worst_onnx:.2e}" if worst_onnx is not None else ""))
    checks = [worst_npz, worst_ts] + ([worst_onnx] if worst_onnx is not None else [])
    if max(checks) > args.tol:
        sys.exit(f"EXPORT REJECTED: backends disagree by more than {args.tol:g}.")

    # ---- metadata ----------------------------------------------------------
    io = {
        "name": args.name or os.path.basename(os.path.dirname(ckpt_path)),
        "checkpoint": ckpt_path,
        "iteration": it,
        "phase": "phase3_student",
        "is_recurrent": True,
        "rnn": {"type": "lstm", "num_layers": 1, "hidden_size": HIDDEN},
        "observation_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "observation_layout": [{"name": n, "dim": d} for n, d in OBS_LAYOUT],
        "obs_scale": [float(x) for x in scale],
        "action": {"scale": 0.25, "use_default_offset": True},
        "control": {"dt": 0.02, "kp": 20.0, "kd": 0.5},
        "aux_heads": {
            "source": "h_out (LSTM output)",
            "splint_length": {"dim": 1, "units": "m"},
            "base_lin_vel": {"dim": 3, "units": "m/s"},
        },
        "npz_keys": sorted(bundle.keys()),
        "notes": [
            "obs is the policy group only (49); privileged terms are teacher-side.",
            "peg_leg_one_hot (45:49) is [FL, FR, RL, RR]; healthy = all zeros. The "
            "student SEES the injured leg — it is not proprioception-only.",
            "The train.normalize obs scale is inside every artifact: baked into the "
            ".pt/.onnx graph, folded into lstm_weight_ih in the .npz. Feed RAW obs.",
            "Policy output is raw action (teacher units); mse_norm is a training loss "
            "scale only. target_q = default_q + 0.25 * action.",
            "Reset the LSTM hidden/cell at every episode start (stand-up handover).",
        ],
        "exported_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(out_dir, "policy_io.json"), "w") as f:
        json.dump(io, f, indent=2)

    ref = {
        "note": "torch CPU reference. pairs = ONE episode from zero hidden state, "
                "run in order. obs is the raw 49-dim deploy vector.",
        "obs_dim": OBS_DIM,
        "recurrent": True,
        "pairs": [{"obs": o.tolist(), "action": a.tolist()}
                  for o, a in zip(ref_obs, ref_actions)],
    }
    with open(os.path.join(out_dir, "reference_io.json"), "w") as f:
        json.dump(ref, f)

    print(f"\nWrote {out_dir}/")
    for n in sorted(os.listdir(out_dir)):
        print(f"  {n:20s} {os.path.getsize(os.path.join(out_dir, n)) / 1024:8.1f} KiB")


if __name__ == "__main__":
    main()
