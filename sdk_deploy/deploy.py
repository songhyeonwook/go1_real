"""Phase-3 student 실물 Go1 배포 메인 루프.

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
  python3 deploy.py --mode hang  --policy model/P3-final/exported/policy_numpy.npz
  python3 deploy.py --mode walk  --policy model/P3-final/exported/policy_numpy.npz --vx 0.4
  python3 deploy.py --mode walk  --policy ... --injured-leg FR   # 부목 착용 시

부상 배포: --injured-leg 를 주면 (1) 관측의 peg_leg_one_hot 이 그 다리로 켜지고,
(2) 해당 calf 는 학습과 동일하게 action 이 마스킹되어 부목 고정각으로 유지됩니다.
물리적으로 부목을 채운 다리와 반드시 일치시켜야 합니다.
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
from observation import build_obs, clip_command, peg_leg_one_hot


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
    def __init__(self, robot, args):
        self.robot = robot
        self.args = args
        self.last_print = 0.0
        self.injured_leg = args.injured_leg
        self.one_hot = peg_leg_one_hot(self.injured_leg)
        if self.injured_leg is not None:
            print("[INJURY] %s 다리 부목 모드 — one_hot=%s, calf action 마스킹, "
                  "고정각 %.2f rad" % (C.LEG_NAMES[self.injured_leg],
                                      self.one_hot.astype(int).tolist(),
                                      C.SPLINT_CALF_ANGLE))

    @staticmethod
    def _aux_msg(policy):
        """정책 보조 헤드의 부목 길이 / 몸통 선속도 추정."""
        est = policy.estimate() if hasattr(policy, "estimate") else None
        if est is None:
            return None
        L, v = est
        return "L_hat=%.3f m  v_hat=(%+.2f,%+.2f,%+.2f)" % (L, v[0], v[1], v[2])

    # ---- 공통 루프 유틸 --------------------------------------------------

    def _contact_count(self, state):
        return int(((state.foot_force - C.FOOT_FORCE_BIAS)
                    > C.CONTACT_FORCE_THRESHOLD).sum())

    def _apply_splint(self, q_des, action):
        """부상 calf 를 학습과 같게 처리 — action 마스킹 + 고정각 유지.

        학습(go1_lab_env.step)은 부상 calf 의 action 을 0 으로 만든 뒤 관절 target
        을 lock angle 로 덮어씁니다. 관측의 last_action 도 마스킹 **후** 값이므로
        여기서 action 자체를 0 으로 되돌려 돌려줍니다.
        """
        if self.injured_leg is None:
            return q_des, action
        calf = 8 + self.injured_leg
        action = action.copy()
        action[calf] = 0.0
        q_des = q_des.copy()
        q_des[calf] = C.SPLINT_CALF_ANGLE
        return q_des, action

    def _tilt_ok(self, state) -> bool:
        roll, pitch = state.rpy[0], state.rpy[1]
        if abs(roll) > C.TILT_LIMIT_RAD or abs(pitch) > C.TILT_LIMIT_RAD:
            print(f"[GUARD] tilt roll={roll:+.2f} pitch={pitch:+.2f} "
                  f"> {C.TILT_LIMIT_RAD} rad — damping")
            return False
        return True

    def _telemetry(self, now, state, action=None, cmd=None, extra=None):
        if now - self.last_print < 1.0:
            return
        self.last_print = now
        msg = (f"rpy=({state.rpy[0]:+.2f},{state.rpy[1]:+.2f}) "
               f"contact={self._contact_count(state)}")
        if cmd is not None:
            msg += f" cmd=({cmd[0]:.2f},{cmd[2]:+.2f})"
        if action is not None:
            msg += f" |a|max={np.abs(action).max():.2f}"
        if extra:
            msg += "\n    [EST] " + extra
        print(msg, flush=True)

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
            if not self._tilt_ok(state):
                raise RuntimeError("tilt guard during hold")
            self.robot.send_positions(
                C.DEFAULT_JOINT_POS, self.args.stand_kp, self.args.stand_kd
            )
            self._telemetry(time.monotonic(), state)
            self._sleep_rest(t0)

    def lie_down(self, duration=C.LIE_DOWN_TIME):
        """정상 종료용: 현재 자세 → 엎드림 자세로 천천히 보간 후 damping.
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
                                   "ff", "quat", "gyro", "cmd", "h", "aux")}
        last_action = np.zeros(C.NUM_ACTIONS)
        n = int(duration / C.CONTROL_DT)
        for k in range(n):
            t0 = time.monotonic()
            if _stdin_pressed():
                print("[POLICY] 사용자 정지")
                break

            state = self.robot.read_state()
            if not self._tilt_ok(state):
                raise RuntimeError("tilt guard during policy")

            ramp = _smoothstep(k * C.CONTROL_DT / cmd_ramp)
            cmd = cmd_target * ramp
            if self.args.vx_floor > 0.0 and cmd_target[0] > 0.0:
                cmd[0] = max(cmd[0], self.args.vx_floor)
            obs = build_obs(state, cmd, last_action, self.one_hot)
            action = policy(obs)
            q_des = C.DEFAULT_JOINT_POS + C.ACTION_SCALE * action
            q_des, action = self._apply_splint(q_des, action)
            last_action = action
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
                log["cmd"].append(cmd.copy())
                log["h"].append(policy.hidden.copy())   # LSTM latent
                est = policy.estimate()
                if est is not None:
                    log["aux"].append(np.concatenate([[est[0]], est[1]]))

            self._telemetry(time.monotonic(), state, action, cmd,
                            extra=self._aux_msg(policy))
            elapsed = time.monotonic() - t0
            if elapsed > C.CONTROL_DT * (1.0 + C.LOOP_OVERRUN_LIMIT):
                print(f"[WARN] 루프 지연 {elapsed * 1000:.1f} ms")
            self._sleep_rest(t0)

        if log is not None:
            np.savez(self.args.log_npz,
                     **{k: np.asarray(v) for k, v in log.items() if v})
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
            now = time.monotonic()
            if now - self.last_print >= 1.0:
                self.last_print = now
                np.set_printoptions(precision=2, suppress=True)
                print(f"q={state.q}")
                print(f"  dq={state.dq}")
                print(f"  rpy={state.rpy} ff={state.foot_force} "
                      f"contact={self._contact_count(state)}", flush=True)
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
    ap.add_argument("--injured-leg", default=None,
                    choices=C.LEG_NAMES,
                    help="부목을 채운 다리. 관측의 peg_leg_one_hot 을 켜고 그 calf 의 "
                         "action 을 마스킹합니다. 생략하면 정상(one_hot 전부 0).")
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
    args.injured_leg = (C.LEG_NAMES.index(args.injured_leg)
                        if args.injured_leg else None)

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

    dep = Deployer(robot, args)
    _flush_stdin()
    try:
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
        # 정상 종료(시간 만료 / 정책 중 Enter)
        # 예외 경로(기울임 가드, Ctrl-C)는 아래 finally 의 즉시 damping 
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
