"""phase1 정책 실물 Go1 배포 메인 루프.

모드 (반드시 이 순서로 검증할 것 — README 의 안전 절차 참고):
  dry-run : 모터 명령 없이 관측/추정치만 출력 (상태 회신을 위한 zero-torque
            요청 패킷만 송신). 로봇을 매단 채 손으로 움직여 관절각
            부호·순서와 추정기 출력을 검증합니다.
  hang    : 로봇을 매단 상태에서 기립 자세 추종 + 정책(명령 0) 실행.
            다리가 발산 없이 트로트 비슷하게 움직이는지 확인합니다.
  stand   : 지면에서 기립 자세만 유지 (정책 미실행).
  walk    : 기립 → 정책 제어. --vx 로 전진 명령 (0.1~1.0 m/s).

비상 정지: Enter 키 → damping 모드로 전환 후 종료.
자동 정지: roll/pitch 가 TILT_LIMIT_RAD 초과 시 즉시 damping.

사용 예:
  python3 deploy.py --mode dry-run --mock
  python3 deploy.py --mode hang --policy exported/policy.onnx
  python3 deploy.py --mode walk --policy exported/policy.onnx --vx 0.4
"""

import os

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import select
import sys
import time

import numpy as np

import config as C
from observation import build_obs, clip_command
from state_estimator import LinearKFStateEstimator


# tty 가 아니면(파이프/백그라운드 실행) Enter e-stop 을 비활성화합니다 —
# 닫힌 stdin 은 항상 readable 이라 루프가 즉시 끊기는 오작동을 막기 위함.
_STDIN_IS_TTY = sys.stdin.isatty()


def _stdin_pressed() -> bool:
    if not _STDIN_IS_TTY:
        return False
    r, _, _ = select.select([sys.stdin], [], [], 0)
    if r:
        sys.stdin.readline()
        return True
    return False


def _flush_stdin():
    """시작 전에 tty 입력 버퍼를 비웁니다.

    명령 입력 후 Enter 를 한 번 더 눌렀거나 붙여넣기에 개행이 딸려 오면
    그 개행이 버퍼에 남아, 첫 _stdin_pressed() 가 곧바로 비상정지로
    오인합니다 (실측: hang 시작 즉시 '사용자 중단'). e-stop 은 시작 이후의
    Enter 만 받아야 하므로 여기서 묵은 입력을 버립니다.
    """
    if not _STDIN_IS_TTY:
        return
    try:
        import termios
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        while select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()


def _smoothstep(t: float) -> float:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class Deployer:
    def __init__(self, robot, est, args):
        self.robot = robot
        self.est = est
        self.args = args
        self.last_print = 0.0
        # 부상 파라미터 probe: 모델 옆에 injury_probe.npz 가 있으면 로드.
        # (시뮬에서 학습한 선형 해독기 W(K,256), b(K), names(K) —
        #  scripts/train_injury_probe.py 로 생성. 없으면 표시 생략.)
        self.probe = None
        policy_path = getattr(args, "policy", None)
        if policy_path:
            p = os.path.join(os.path.dirname(os.path.abspath(policy_path)),
                             "injury_probe.npz")
            if os.path.exists(p):
                data = np.load(p)
                self.probe = {
                    "W": data["W"].astype(np.float32),
                    "b": data["b"].astype(np.float32),
                    "names": [str(n) for n in data["names"]],
                }
                print("[PROBE] injury_probe.npz 로드 — 실시간 부상 추정 표시: "
                      + ", ".join(self.probe["names"]))

    def _probe_msg(self, policy):
        h = getattr(policy, "hidden", None)
        if self.probe is None or h is None:
            return None
        y = self.probe["W"] @ h + self.probe["b"]
        return " ".join("%s=%+.2f" % (n, v)
                        for n, v in zip(self.probe["names"], y))

    # ---- 공통 루프 유틸 --------------------------------------------------

    def _step_estimator(self, state, dt):
        # 첫 상태 패킷이 도착하기 전에는 quat 가 [0,0,0,0]입니다. 그대로
        # KF 에 넣으면 회전행렬이 NaN 이 되고 공분산까지 오염돼 이후 모든
        # 출력이 NaN 으로 남으므로, 유효한 자세가 올 때까지 스킵합니다.
        if np.linalg.norm(state.quat_wxyz) < 0.5:
            return {
                "v_body": np.zeros(3), "v_world": np.zeros(3),
                "p_world": np.zeros(3), "contact_count": 0,
            }
        contact = (state.foot_force - C.FOOT_FORCE_BIAS) > C.CONTACT_FORCE_THRESHOLD
        return self.est.update(
            state.quat_wxyz, state.gyro, state.accel,
            state.q, state.dq, contact, dt,
        )

    def _tilt_ok(self, state) -> bool:
        roll, pitch = state.rpy[0], state.rpy[1]
        if abs(roll) > C.TILT_LIMIT_RAD or abs(pitch) > C.TILT_LIMIT_RAD:
            print(f"[GUARD] tilt roll={roll:+.2f} pitch={pitch:+.2f} "
                  f"> {C.TILT_LIMIT_RAD} rad — damping")
            return False
        return True

    def _telemetry(self, now, state, est_out, action=None, cmd=None, extra=None):
        if now - self.last_print < 1.0:
            return
        self.last_print = now
        v = est_out["v_body"]
        msg = (f"v=({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f}) "
               f"rpy=({state.rpy[0]:+.2f},{state.rpy[1]:+.2f}) "
               f"contact={est_out['contact_count']}")
        if cmd is not None:
            msg += f" cmd=({cmd[0]:.2f},{cmd[2]:+.2f})"
        if action is not None:
            msg += f" |a|max={np.abs(action).max():.2f}"
        if extra:
            msg += "\n    [EST] " + extra
        print(msg, flush=True)

    def _estimator_warmup(self, seconds=1.0):
        for _ in range(int(seconds / C.CONTROL_DT)):
            # MCU 는 패킷을 보낸 클라이언트에게만 상태를 회신하므로, 아직
            # 아무 명령도 안 보내는 워밍업에서는 zero-torque 로 상태를
            # 요청해야 합니다. 없으면 stand_up() 이 시작 자세를 전부 0 으로
            # 읽어 잘못된 자세에서 보간을 시작합니다.
            self.robot.send_poll()
            state = self.robot.read_state()
            self._step_estimator(state, C.CONTROL_DT)
            time.sleep(C.CONTROL_DT if not self.args.mock else 0.0)

    # ---- 시퀀스 ----------------------------------------------------------

    def stand_up(self, duration=3.0):
        """현재 자세 → DEFAULT_JOINT_POS 로 부드럽게 보간 (STAND_KP)."""
        print("[STAND] 기립 시퀀스 시작 (Enter = 중단)")
        q0 = self.robot.read_state().q.copy()
        n = int(duration / C.CONTROL_DT)
        for k in range(n):
            t0 = time.monotonic()
            if _stdin_pressed():
                raise KeyboardInterrupt
            s = _smoothstep((k + 1) / n)
            q_des = (1.0 - s) * q0 + s * C.DEFAULT_JOINT_POS
            state = self.robot.read_state()
            self._step_estimator(state, C.CONTROL_DT)
            if not self._tilt_ok(state):
                raise RuntimeError("tilt guard during stand-up")
            self.robot.send_positions(q_des, self.args.stand_kp, self.args.stand_kd)
            self._sleep_rest(t0)
        err = float(np.abs(self.robot.read_state().q - C.DEFAULT_JOINT_POS).max())
        warn = "  <-- 기립 불충분! 정책을 켜지 마세요" if err > 0.3 else ""
        print(f"[STAND] 완료 — 기본 자세 유지 중 (최대 관절 오차 {err:.2f} rad){warn}")

    def hold_default(self, duration):
        for _ in range(int(duration / C.CONTROL_DT)):
            t0 = time.monotonic()
            if _stdin_pressed():
                raise KeyboardInterrupt
            state = self.robot.read_state()
            out = self._step_estimator(state, C.CONTROL_DT)
            if not self._tilt_ok(state):
                raise RuntimeError("tilt guard during hold")
            self.robot.send_positions(
                C.DEFAULT_JOINT_POS, self.args.stand_kp, self.args.stand_kd
            )
            self._telemetry(time.monotonic(), state, out)
            self._sleep_rest(t0)

    def lie_down(self, duration=C.LIE_DOWN_TIME):
        """정상 종료용: 현재 자세 → 엎드림 자세로 천천히 보간 후 damping 으로 인계.

        damping 직행(Kp=0)은 로봇이 뚝 떨어지듯 주저앉습니다. 여기서는 위치
        제어를 유지한 채 접힌 자세로 내려간 뒤에 damping 을 걸어 사뿐히
        앉습니다. Enter 를 누르면 즉시 중단하고 damping 으로 넘어갑니다.
        """
        print("[LIE] 천천히 주저앉는 중 (Enter = 건너뛰고 즉시 damping)")
        q0 = self.robot.read_state().q.copy()
        n = int(duration / C.CONTROL_DT)
        for k in range(n):
            t0 = time.monotonic()
            if _stdin_pressed():
                break
            s = _smoothstep((k + 1) / n)
            q_des = (1.0 - s) * q0 + s * C.LIE_DOWN_POS
            state = self.robot.read_state()
            self._step_estimator(state, C.CONTROL_DT)
            self.robot.send_positions(q_des, C.LIE_DOWN_KP, C.LIE_DOWN_KD)
            self._sleep_rest(t0)

    def run_policy(self, policy, cmd_target, duration, cmd_ramp=2.0):
        print(f"[POLICY] 시작 cmd={cmd_target} (Enter = 정지)")
        # recurrent(LSTM) 정책은 학습에서 에피소드가 hidden=0 으로 시작하므로,
        # 정책 인계 시점에 상태를 리셋합니다 (feed-forward / selftest 람다는 no-op).
        if hasattr(policy, "reset"):
            policy.reset()
        log = None
        if getattr(self.args, "log_npz", None):
            log = {k: [] for k in ("t", "q", "dq", "q_des", "action",
                                   "ff", "quat", "gyro", "v_kf", "cmd", "h")}
        last_action = np.zeros(C.NUM_ACTIONS)
        n = int(duration / C.CONTROL_DT)
        t_prev = time.monotonic()
        for k in range(n):
            t0 = time.monotonic()
            dt = np.clip(t0 - t_prev, 0.5 * C.CONTROL_DT, 2 * C.CONTROL_DT)
            t_prev = t0
            if _stdin_pressed():
                print("[POLICY] 사용자 정지")
                break

            state = self.robot.read_state()
            out = self._step_estimator(state, dt)
            if not self._tilt_ok(state):
                raise RuntimeError("tilt guard during policy")

            ramp = _smoothstep(k * C.CONTROL_DT / cmd_ramp)
            cmd = cmd_target * ramp
            # 학습 명령 분포에 하한이 있는 모델(phase3 student: vx 0.3~1.0)은
            # 램프가 하한 밑을 통과하면 분포 밖 명령이 LSTM 이력에 쌓입니다.
            # --vx-floor 로 하한을 주면 전진 명령은 시작부터 그 값 이상입니다
            # (학습 에피소드도 정지 상태 + cmd>=하한에서 시작하므로 일치).
            if self.args.vx_floor > 0.0 and cmd_target[0] > 0.0:
                cmd[0] = max(cmd[0], self.args.vx_floor)
            # base_lin_vel 입력: 이 저장소의 모든 시뮬 검증(sim_test/deploy_core.py,
            # ROS 스택)은 속도 명령을 그대로 넣는 proxy 방식입니다. 온보드 KF 는
            # 실보행에서 과소추정이 확인돼(cmd 0.3 에서 0.02~0.15) 기본값은 proxy,
            # --lin-vel kf 로 추정치를 쓸 수 있습니다 (텔레메트리는 항상 KF).
            if self.args.lin_vel == "cmd":
                lin_vel_obs = np.array([cmd[0], cmd[1], 0.0], dtype=np.float32)
            else:
                lin_vel_obs = out["v_body"]
            obs = build_obs(state, lin_vel_obs, cmd, last_action)
            action = policy(obs)
            last_action = action
            q_des = C.DEFAULT_JOINT_POS + C.ACTION_SCALE * action
            # 기립 게인(STAND_KP=60)에서 정책 게인으로 부드럽게 블렌딩.
            # 인계 직후 바로 Kp 를 20 으로 떨어뜨리면 아직 하중이 실린 뒷무릎이
            # 처지면서 정책이 학습 밖 자세에서 시작합니다 (실측: 2026-08-02 로그).
            blend = _smoothstep(k * C.CONTROL_DT / C.GAIN_BLEND_TIME)
            kp_now = self.args.stand_kp + (self.args.kp - self.args.stand_kp) * blend
            kd_now = self.args.stand_kd + (self.args.kd - self.args.stand_kd) * blend
            self.robot.send_positions(q_des, kp_now, kd_now)

            if log is not None:
                log["t"].append(t0)
                log["q"].append(state.q.copy())
                log["dq"].append(state.dq.copy())
                log["q_des"].append(q_des.copy())
                log["action"].append(np.asarray(action, dtype=np.float32))
                log["ff"].append(state.foot_force.copy())
                log["quat"].append(state.quat_wxyz.copy())
                log["gyro"].append(state.gyro.copy())
                log["v_kf"].append(out["v_body"].copy())
                log["cmd"].append(cmd.copy())
                h = getattr(policy, "hidden", None)
                if h is not None:
                    log["h"].append(h.copy())  # LSTM 상태 — probe 학습/부상 추정용

            self._telemetry(time.monotonic(), state, out, action, cmd,
                            extra=self._probe_msg(policy))
            elapsed = time.monotonic() - t0
            if elapsed > C.CONTROL_DT * (1.0 + C.LOOP_OVERRUN_LIMIT):
                print(f"[WARN] 루프 지연 {elapsed * 1000:.1f} ms")
            self._sleep_rest(t0)

        if log is not None:
            np.savez(self.args.log_npz, **{k: np.asarray(v) for k, v in log.items()})
            print(f"[LOG] {len(log['t'])} steps -> {self.args.log_npz}")

    def dry_run(self, duration):
        print("[DRY] 모터 명령 없음 (zero-torque 상태요청만 송신) — "
              "로봇을 손으로 움직여 값 확인")
        for _ in range(int(duration / C.CONTROL_DT)):
            t0 = time.monotonic()
            if _stdin_pressed():
                break
            self.robot.send_poll()
            state = self.robot.read_state()
            out = self._step_estimator(state, C.CONTROL_DT)
            now = time.monotonic()
            if now - self.last_print >= 1.0:
                self.last_print = now
                np.set_printoptions(precision=2, suppress=True)
                print(f"q={state.q}")
                print(f"  v_body={out['v_body']} rpy={state.rpy} "
                      f"ff={state.foot_force}", flush=True)
            self._sleep_rest(t0)

    def _sleep_rest(self, t0):
        if self.args.mock:
            return
        rest = C.CONTROL_DT - (time.monotonic() - t0)
        if rest > 0:
            time.sleep(rest)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True,
                    choices=["dry-run", "hang", "stand", "walk"])
    ap.add_argument("--policy", default=None, help="policy.onnx / policy.pt")
    ap.add_argument("--vx", type=float, default=0.3)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--kp", type=float, default=C.KP)
    ap.add_argument("--kd", type=float, default=C.KD)
    ap.add_argument("--power-protect", type=int,
                    default=C.POWER_PROTECT_LEVEL)
    ap.add_argument("--lin-vel", choices=["cmd", "kf"], default="cmd",
                    help="base_lin_vel 관측 소스. cmd(기본) = 속도 명령 proxy "
                         "(모든 시뮬 검증과 동일), kf = 온보드 칼만 필터 추정")
    ap.add_argument("--vx-floor", type=float, default=0.0,
                    help="전진 명령 램프의 하한. 학습 분포에 하한이 있는 모델용 "
                         "(phase3 student 는 0.3 권장)")
    ap.add_argument("--stand-kp", type=float, default=C.STAND_KP,
                    help="기립/유지 게인. 60=수평 기립(실측), 30=부드럽지만 뒷다리 처짐")
    ap.add_argument("--stand-kd", type=float, default=C.STAND_KD)
    ap.add_argument("--stand-time", type=float, default=5.0,
                    help="기립 보간 시간(s). 길수록 부드러움")
    ap.add_argument("--log-npz", default=None,
                    help="정책 구간의 스텝별 상태/명령을 .npz 로 저장 (오프라인 진단용)")
    ap.add_argument("--mock", action="store_true",
                    help="SDK 없이 mock 로봇으로 코드 경로 검증")
    args = ap.parse_args()

    if args.mock:
        from robot_io import MockGo1Interface
        robot = MockGo1Interface()
    else:
        from robot_io import Go1Interface
        robot = Go1Interface(power_protect=args.power_protect)

    policy = None
    if args.mode in ("hang", "walk"):
        if not args.policy:
            ap.error(f"--mode {args.mode} 에는 --policy 가 필요합니다")
        from policy import Policy
        policy = Policy(args.policy)

    dep = Deployer(robot, LinearKFStateEstimator(), args)
    _flush_stdin()
    try:
        dep._estimator_warmup()
        if args.mode == "dry-run":
            dep.dry_run(args.duration)
            return
        dep.stand_up(duration=args.stand_time)
        dep.hold_default(1.0)
        if args.mode == "stand":
            dep.hold_default(args.duration)
        elif args.mode == "hang":
            dep.run_policy(policy, np.zeros(3), args.duration)
        elif args.mode == "walk":
            cmd = clip_command(args.vx, args.vy, args.wz)
            dep.run_policy(policy, cmd, args.duration)
        # 정상 종료(시간 만료 / 정책 중 Enter)만 여기 도달: 사뿐히 앉기.
        # 예외 경로(기울임 가드, Ctrl-C)는 아래 finally 의 즉시 damping 으로 감.
        dep.lie_down()
    except KeyboardInterrupt:
        print("[STOP] 사용자 중단")
    finally:
        if args.mode != "dry-run":
            print("[STOP] damping 모드로 종료")
            for _ in range(int(1.0 / C.CONTROL_DT)):
                robot.send_damping()
                if not args.mock:
                    time.sleep(C.CONTROL_DT)


if __name__ == "__main__":
    main()
