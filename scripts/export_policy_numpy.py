#!/usr/bin/env python3
"""Convert an exported policy.onnx into a plain-NumPy .npz for on-robot inference.

The Go1 onboard NX has no torch and no onnxruntime (Python 3.6.9 / numpy 1.13.3,
no internet), so deploy_policy.py runs its NumPy backend there. This script turns
the ONNX graph into the weight bundle that backend expects.

Supports both graph shapes this repo exports:
  * feed-forward  : obs -> Gemm/Elu x4 -> actions          (Phase-1 healthy)
  * recurrent     : obs -> LSTM(1 layer) -> Gemm/Elu x4    (Phase-3 students)

Run on the DEV machine (needs `pip install onnx`), then rsync the .npz to the robot.

    python3 scripts/export_policy_numpy.py antalgic/exported/policy.onnx

Writes policy_numpy.npz next to the input unless -o is given.
"""
import argparse
import os
import sys

import numpy as np

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    sys.exit("onnx is required on the dev machine: pip install onnx")

# ONNX LSTM packs gates as [input, output, forget, cell]; PyTorch's nn.LSTM packs them as
# [input, forget, cell, output]. The bundles committed in this repo use the PyTorch layout and
# naming (lstm_weight_ih / weight_hh / bias_ih / bias_hh), which is what deploy_policy.py reads,
# so this exporter reorders the ONNX gates into PyTorch order on the way out.
ONNX_GATE_ORDER = "iofc"
TORCH_GATE_ORDER = "ifgo"


def _initializers(graph):
    return {t.name: numpy_helper.to_array(t) for t in graph.initializer}


def _mlp_layers(graph, init):
    """Collect the Gemm weight/bias pairs in graph order.

    Returns [(W, b), ...] with W shaped (out, in) — i.e. already transposed for
    `x @ W.T + b`, which is how the ONNX Gemm nodes are emitted (transB=1).
    """
    layers = []
    for node in graph.node:
        if node.op_type != "Gemm":
            continue
        trans_b = next((a.i for a in node.attribute if a.name == "transB"), 0)
        if trans_b != 1:
            sys.exit(f"Gemm {node.name!r} has transB={trans_b}; only transB=1 exports are supported.")
        w_name, b_name = node.input[1], node.input[2]
        if w_name not in init or b_name not in init:
            sys.exit(f"Gemm {node.name!r} has non-constant weights; cannot export to NumPy.")
        layers.append((init[w_name].astype(np.float32), init[b_name].astype(np.float32)))
    if not layers:
        sys.exit("No Gemm layers found in the ONNX graph.")
    return layers


def _lstm(graph, init):
    """Extract the single LSTM node, if present. Returns None for feed-forward graphs."""
    nodes = [n for n in graph.node if n.op_type == "LSTM"]
    if not nodes:
        return None
    if len(nodes) > 1:
        sys.exit(f"Found {len(nodes)} LSTM nodes; only single-layer LSTM exports are supported.")
    node = nodes[0]

    direction = next((a.s.decode() for a in node.attribute if a.name == "direction"), "forward")
    if direction != "forward":
        sys.exit(f"LSTM direction={direction!r}; only 'forward' is supported.")
    for attr in node.attribute:
        if attr.name == "activations":
            sys.exit("LSTM uses custom activations; the NumPy backend assumes sigmoid/tanh/tanh.")
    if len(node.input) > 7 and node.input[7]:
        sys.exit("LSTM has peephole weights (input P); the NumPy backend does not implement them.")

    hidden = next((a.i for a in node.attribute if a.name == "hidden_size"), None)
    if hidden is None:
        sys.exit("LSTM node is missing the hidden_size attribute.")

    # ONNX shapes: W (num_dir, 4*hidden, input), R (num_dir, 4*hidden, hidden),
    #              B (num_dir, 8*hidden) = concat(Wb, Rb)
    w = init[node.input[1]].astype(np.float32)
    r = init[node.input[2]].astype(np.float32)
    b = init[node.input[3]].astype(np.float32)
    if w.shape[0] != 1:
        sys.exit(f"LSTM has num_directions={w.shape[0]}; only 1 is supported.")

    w, r, b = w[0], r[0], b[0]

    # ONNX order i,o,f,c -> PyTorch order i,f,g,o. (g is ONNX's "c" cell-candidate gate.)
    def to_torch_order(mat):
        i, o, f, c = (mat[k * hidden:(k + 1) * hidden] for k in range(4))
        return np.concatenate([i, f, c, o], axis=0)

    return {
        "lstm_weight_ih": to_torch_order(w),               # (4*hidden, input)
        "lstm_weight_hh": to_torch_order(r),               # (4*hidden, hidden)
        "lstm_bias_ih": to_torch_order(b[: 4 * hidden]),   # (4*hidden,)
        "lstm_bias_hh": to_torch_order(b[4 * hidden:]),    # (4*hidden,)
        "lstm_hidden_size": np.int64(hidden),
        "lstm_input_size": np.int64(w.shape[1]),
    }


def _pd_gains_from_env_yaml(path):
    """Pull the actuator PD gains out of a training env.yaml.

    The students train with an explicit DCMotor (stiffness 20.0 / damping 0.5), so those are the
    gains the policy actually learned against — not a sim-to-real guess. Phase-1 used a learned
    ActuatorNetMLP with `stiffness: null`, which yields no usable PD equivalent (returns None).

    Parsed by regex rather than yaml.safe_load because these Hydra dumps carry `!!python/tuple`
    tags that safe_load rejects.
    """
    import re

    with open(path, "r") as f:
        text = f.read()

    match = re.search(r"^    actuators:\s*$", text, re.M)
    if not match:
        return None
    block = text[match.end():]
    # Stop at the next top-level-of-scene key so we only read the actuator entries.
    end = re.search(r"^    \w", block, re.M)
    if end:
        block = block[: end.start()]

    stiffness = re.search(r"^\s*stiffness:\s*([0-9.eE+-]+)\s*$", block, re.M)
    damping = re.search(r"^\s*damping:\s*([0-9.eE+-]+)\s*$", block, re.M)
    if not stiffness or not damping:
        return None
    return float(stiffness.group(1)), float(damping.group(1))


def _io_dims(graph):
    def dims(value_info):
        return [d.dim_value for d in value_info.type.tensor_type.shape.dim]

    obs_dim = dims(graph.input[0])[-1]
    action_dim = dims(graph.output[0])[-1]
    return int(obs_dim), int(action_dim)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx_path", help="path to exported policy.onnx")
    ap.add_argument("-o", "--output", default=None,
                    help="output .npz path (default: policy_numpy.npz beside the input)")
    ap.add_argument("--env-yaml", default=None,
                    help="training env.yaml; embeds the actuator PD gains into the bundle so "
                         "deploy_policy.py runs the gains the policy was trained against")
    args = ap.parse_args()

    model = onnx.load(args.onnx_path)
    graph = model.graph
    init = _initializers(graph)

    obs_dim, action_dim = _io_dims(graph)
    layers = _mlp_layers(graph, init)
    lstm = _lstm(graph, init)

    # Keep the legacy "<idx>_weight" key names so older .npz bundles and this one
    # load through the same code path in deploy_policy.py.
    bundle = {"obs_dim": np.int64(obs_dim), "action_dim": np.int64(action_dim)}
    for i, (w, b) in enumerate(layers):
        bundle[f"{2 * i}_weight"] = w
        bundle[f"{2 * i}_bias"] = b
    bundle["num_mlp_layers"] = np.int64(len(layers))

    if lstm is not None:
        bundle.update(lstm)
        bundle["arch"] = np.array(b"lstm_mlp")
        bundle["gate_order"] = np.array(TORCH_GATE_ORDER.encode())
        if lstm["lstm_input_size"] != obs_dim:
            sys.exit(f"LSTM input size {int(lstm['lstm_input_size'])} != graph obs dim {obs_dim}.")
    else:
        bundle["arch"] = np.array(b"mlp")

    gains = None
    if args.env_yaml:
        gains = _pd_gains_from_env_yaml(args.env_yaml)
        if gains is None:
            print(f"  note: no explicit PD gains in {args.env_yaml} "
                  f"(learned-actuator run?); bundle carries no gains.")
        else:
            bundle["pd_stiffness"] = np.float32(gains[0])
            bundle["pd_damping"] = np.float32(gains[1])

    out_path = args.output or os.path.join(os.path.dirname(os.path.abspath(args.onnx_path)),
                                           "policy_numpy.npz")
    np.savez(out_path, **bundle)

    arch = "LSTM+MLP" if lstm else "MLP"
    shapes = " -> ".join(str(w.shape[0]) for w, _ in layers)
    hidden = f", lstm_hidden={int(lstm['lstm_hidden_size'])}" if lstm else ""
    print(f"Wrote {out_path}")
    print(f"  arch={arch}, obs_dim={obs_dim}, action_dim={action_dim}{hidden}")
    print(f"  mlp: {obs_dim if not lstm else int(lstm['lstm_hidden_size'])} -> {shapes}")
    if gains:
        print(f"  pd gains from env.yaml: Kp={gains[0]}, Kd={gains[1]}")


if __name__ == "__main__":
    main()
