# sim_test — Deployment-parity test in Isaac Sim (before hardware)

Validates the **exact artifact + pipeline that will run on the real Go1** inside the
Isaac Lab training environment, so sim-to-real bugs are caught before they reach the
robot. Focus is the three deployable proprioception-only LSTM students
(`antalgic`, `fault_tolerant`, `symmetry`).

## What it checks

For every control step it runs three checks:

1. **Net-export fidelity** — feed the *same* env observation to the RSL-RL reference
   policy and to the exported artifact (`policy_numpy.npz` / `policy.pt`); report the
   action difference. Isolates the ONNX/npz/JIT export.
2. **Obs-assembly fidelity** — rebuild the observation the way `deploy_policy.py` does
   and diff it block-by-block against the env's own observation
   (`base_lin_vel`, `base_ang_vel`, `projected_gravity`, `velocity_commands`,
   `joint_pos`, `joint_vel`, `last_action`).
3. **Closed-loop behavior** — actually drive the robot in sim with the deployment
   pipeline and log base height, velocity tracking and falls.

Plus a **joint-order audit**: the sim articulation joint order vs the order
`deploy_policy.py` hardcodes.

`--drive_mode`:
- `asis`  — the current `deploy_policy.py` path, bug included (hardware-faithful).
- `fixed` — the corrected name-based joint mapping (what deploy_policy *should* do).
- `reference` — the raw RSL-RL policy (upper bound).

## Files

- `deploy_core.py` — ROS-free port of `scripts/deploy_policy.py` (obs assembly, npz/jit
  inference, action post-processing). Standalone self-test:
  `python3 sim_test/deploy_core.py numpy` (checks all 3 students vs their reference action).
- `sim_deploy_parity.py` — the Isaac Sim launcher (runs in the `isaac` conda env).
- `run_sim_test.sh` — convenience wrapper over the three students.

## Running

Isaac Lab lives in the `isaac` conda env (same env used for training). The env vars
that reproduce the students' training/eval config are set inside `run_sim_test.sh`
(taken from `go1_peg/scripts/rsl_rl/eval_metrics_lo.sh`):
`GO1_PROPRIO_ONLY=1` (policy obs = 48), `GO1_INJURY_ONEHOT=1` (teacher privileged = 7,
needed for the checkpoint to load), `GO1_FLAT_TERRAIN=1`, and the PD actuator
`GO1_PD_ACTUATOR=1 GO1_PD_KP=20 GO1_PD_KD=0.5`.

```bash
# all three students, corrected mapping, headless
sim_test/run_sim_test.sh

# one student, current deploy path, with video
DRIVE_MODE=asis VIDEO=1 sim_test/run_sim_test.sh antalgic

# torch (JIT) backend instead of numpy
BACKEND=torch sim_test/run_sim_test.sh antalgic
```

Reports are written to `<model>/exported/sim_parity_report_<mode>.json`
(summary + per-step series).

## Key finding (2026-07-24): joint-order bug in deploy_policy.py

The audit fails: the sim articulation order is grouped **by joint type**, while
`deploy_policy.py` assumes grouping **by leg**.

```
sim articulation : [FL_hip, FR_hip, RL_hip, RR_hip,
                    FL_thigh, FR_thigh, RL_thigh, RR_thigh,
                    FL_calf, FR_calf, RL_calf, RR_calf]      # by joint type
deploy assumed   : [FL_hip, FL_thigh, FL_calf, FR_hip, ...] # by leg  ← WRONG
```

The trained network was trained on the type-grouped order, so on hardware every joint
observation and action is permuted. Measured impact (antalgic, cmd vx = 0.5):

| metric                | `fixed` (corrected) | `asis` (current) |
|-----------------------|--------------------:|-----------------:|
| obs joint_pos err     | **0.0000**          | 1.08             |
| obs joint_vel err     | **0.0000**          | 15.58            |
| base height min       | 0.230               | 0.152 (near fall)|
| actual vx (cmd 0.5)   | **0.84** (walks)    | 0.14 (barely)    |

Everything else is correct: export fidelity ~1e-6, projected-gravity match ~6e-8,
gyro/command assembly exact. **Only the joint mapping is wrong.**

### Corrected constants for `scripts/deploy_policy.py`

Unitree SDK order `[FR, FL, RR, RL] × (hip, thigh, calf)` → sim articulation order
(type-grouped):

```python
# joint_pos_isaac = joint_pos_unitree[U2I]
self.U2I = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
# target_unitree = target_isaac[I2U]   (no longer involutive!)
self.I2U = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]

# defaults / limits in type-grouped order
self.default_joint_pos = np.array([ 0.1, -0.1,  0.1, -0.1,   # hips  FL FR RL RR
                                     0.8,  0.8,  1.0,  1.0,   # thighs
                                    -1.5, -1.5, -1.5, -1.5])  # calfs
self.joint_pos_min = np.array([-1,-1,-1,-1, -1,-1,-1,-1, -2.7,-2.7,-2.7,-2.7])
self.joint_pos_max = np.array([ 1, 1, 1, 1,  3, 3, 3, 3, -0.8,-0.8,-0.8,-0.8])
```

Also the injured-leg calf index changes: in type-grouped order leg `L`'s
(hip, thigh, calf) live at indices `(0+L, 4+L, 8+L)`, not `(3L, 3L+1, 3L+2)`.

> Verify the Unitree SDK motor order against your `unitree_legged_msgs` bridge before
> trusting `U2I`/`I2U`; the audit only proves the *Isaac side* ordering is wrong. The
> `fixed` drive mode implements this corrected mapping and walks in sim.

## Bring-up gains (stand → policy)

The harness reproduces the real bring-up: a stand phase at `--stand_kp/--stand_kd`
holding the default pose, then the policy phase at `--policy_kp/--policy_kd`. The
env's `DCMotor` actuator gains are overridden at runtime (torque clamped to the
23.7 Nm motor curve), so the sim shows how the policy behaves under the *hardware*
gains — which differ from the training actuator (Kp=20/Kd=0.5).

Second finding (2026-07-24): at the hardware bring-up gains **Kp=60/Kd=1** the
students **stay up but ride too high and track velocity poorly** — the 3× stiffness
turns the same position targets into ~3× torque:

| gains (policy)      | base height (target 0.3) | actual vx (cmd 0.5) | fell |
|---------------------|--------------------------|---------------------|------|
| Kp20/Kd0.5 (train)  | ~0.37                    | ~0.84               | no   |
| Kp60/Kd1  (bring-up)| 0.39–0.45 (too tall)     | ~0.27 (jerky)       | no   |

So Kp60/Kd1 is *survivable* but degraded; running the policy phase closer to the
training gains (Kp≈20/Kd≈0.5) restores clean tracking. Keeping Kp60/Kd1 for the
**stand** phase is fine. Sweep with e.g.
`POLICY_KP=40 sim_test/run_sim_test.sh antalgic`.

## Third finding: base_lin_vel must not be zero

The student's 48-dim obs includes `base_lin_vel` (body velocity, dims 0:3), which the
real Go1 cannot measure directly. `deploy_policy.py` originally fed **zeros**, so the
policy "thinks" it is standing still and over-accelerates. Measured (antalgic, cmd
vx=0.5, Kp20), `--linvel_source`:

| base_lin_vel source        | actual vx (cmd 0.5) | height | tilt | note                    |
|----------------------------|--------------------:|-------:|-----:|-------------------------|
| `true` (perfect estimator) | 0.46                | 0.379  | 4°   | reference-quality       |
| `command` (proxy, no sensor)| **0.46**           | 0.378  | 6°   | **the deployed fix**    |
| `zero` (original)          | 0.93 (overshoot)    | 0.358  | 10°  | broken velocity control |

Reference (true obs, ideal): vx 0.49, height 0.393, 1°.

**Fix applied** in `deploy_policy.py` / `deploy_core.py`: feed the velocity command
`[vx_cmd, vy_cmd, 0]` as the `base_lin_vel` proxy — no extra sensing, near-ideal
tracking. If a real body-velocity estimate (leg odometry / state estimator) is
available, feed that instead for the last few percent.

## Deployment config that produces normal walking (verified)

- Joint order: corrected (type-grouped) mapping.
- Stand gains: Kp60/Kd1. Policy gains: **Kp20/Kd0.5**.
- `base_lin_vel`: velocity-command proxy.

The `asis` drive mode now mirrors the patched `deploy_policy.py`, so
`sim_test/run_sim_test.sh` (defaults: `POLICY_KP=20 LINVEL_SOURCE=command`) runs the
real deployed pipeline and the students walk and track the command in sim.
