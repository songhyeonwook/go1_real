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
        self._aux_ema = None
        # stand / dry-run 용 상태 로그 (정책 구간은 run_policy 가 따로 기록)
        self._static_log = None
        if getattr(args, "log_npz", None):
            self._static_log = {k: [] for k in
                                ("t", "q", "dq", "q_des", "ff", "quat", "gyro")}
        if self.injured_leg is not None:
            print("[INJURY] %s 다리 부목 모드 — one_hot=%s, calf action 마스킹, "
                  "고정각 %.2f rad" % (C.LEG_NAMES[self.injured_leg],
                                      self.one_hot.astype(int).tolist(),
                                      C.SPLINT_CALF_ANGLE))

    def _aux_update(self, est, dt):
        """보조 헤드 추정치를 매 스텝 EMA(시정수 AUX_EMA_TAU)로 평활."""
        if est is None:
            return
        y = np.concatenate([[est[0]], est[1]]).astype(np.float64)
        if self._aux_ema is None:
            self._aux_ema = y
        else:
            self._aux_ema += min(1.0, dt / C.AUX_EMA_TAU) * (y - self._aux_ema)

    def _aux_msg(self, elapsed):
        """평활된 부목 길이 / 몸통 선속도 추정.

        인계 직후에는 LSTM 이 아직 이력을 쌓는 중이라 추정이 출렁입니다 (구 모델
        실측: 인계 구간에 가짜 부상 스파이크 후 소멸). 정책 시작 후 1초까지는
        값에 (수렴 중) 을 붙입니다.
        """
        if self._aux_ema is None:
            return None
        L, v = self._aux_ema[0], self._aux_ema[1:]
        msg = "v_hat=(%+.2f,%+.2f,%+.2f) m/s" % (v[0], v[1], v[2])
        if self.injured_leg is not None:
            # splint_head 는 부목 길이 0.3~0.4 m 라벨로만 학습됐습니다 (config.json
            # mse_norm.splint_mean 0.35 / std 0.029). 정상 로봇에서는 그 평균(0.35)
            # 근처를 그냥 출력하므로 의미가 없어 부목 모드에서만 표시합니다.
            msg = "L_hat=%.3f m  " % L + msg
        if elapsed < 1.0:
            msg += "  (수렴 중)"
        return msg

    # ---- 공통 루프 유틸 --------------------------------------------------

    def _static_log_step(self, t0, state, q_des):
        if self._static_log is None:
            return
        L = self._static_log
        L["t"].append(t0)
        L["q"].append(state.q.copy())
        L["dq"].append(state.dq.copy())
        L["q_des"].append(np.asarray(q_des, dtype=np.float64).copy())
        L["ff"].append(state.foot_force.copy())
        L["quat"].append(state.quat_wxyz.copy())
        L["gyro"].append(state.gyro.copy())

    def save_static_log(self):
        """stand / dry-run 의 상태 로그 저장 (hang/walk 는 run_policy 가 저장)."""
        L = self._static_log
        if L is None or not L["t"]:
            return
        np.savez(self.args.log_npz, **{k: np.asarray(v) for k, v in L.items()})
        print(f"[LOG] {len(L['t'])} steps -> {self.args.log_npz}")

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

    def state_warmup(self, seconds=1.0):
        """모터 명령 전에 zero-torque 패킷으로 상태 회신을 받아 둡니다.

        Go1 MCU 는 패킷을 보낸 클라이언트에게만 low-level 상태를 회신하므로,
        이걸 건너뛰면 stand_up() 이 시작 자세를 전부 0 으로 읽어 엎드린 로봇에
        다리를 완전히 뻗은 자세를 Kp 60 으로 명령합니다 (실기: 점프 후 전복).
        """
        for _ in range(int(seconds / C.CONTROL_DT)):
            self.robot.send_poll()
            self.robot.read_state()
            if not self.args.mock:
                time.sleep(C.CONTROL_DT)

    def stand_up(self, duration=3.0):
        """현재 자세 → DEFAULT_JOINT_POS 로 부드럽게 보간 (STAND_KP)."""
        print("[STAND] 기립 시퀀스 시작 (Enter = 중단)")
        q0 = self.robot.read_state().q.copy()
        if float(np.abs(q0).max()) < 1e-6:
            # 상태를 한 번도 못 받은 것 — 0 에서 보간을 시작하면 안 됩니다.
            raise RuntimeError(
                "stand_up: 관절 상태가 전부 0 (MCU 회신 없음). 하위제어 모드인지, "
                "state_warmup() 을 거쳤는지 확인하세요")
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
            self._static_log_step(t0, state, C.DEFAULT_JOINT_POS)
            self._telemetry(time.monotonic(), state)
            self._sleep_rest(t0)

    def blend_gains(self, duration):
        """정책 인계 전에 기본 자세를 홀드한 채 게인을 STAND_KP/KD -> KP/KD 로 내립니다.

        정책은 학습 내내 Kp 20 에서만 돌았습니다. 인계 순간의 "발작"은 보통 크기의
        보행 행동이 기립 강성(Kp 60)에서 실행돼 생기는 것이라(실측), 블렌딩을 정책
        실행 중이 아니라 그 전에 끝내서 정책이 첫 스텝부터 학습 조건(Kp 20, 기본
        자세, hidden 0, 100% 권한)에서 시작하게 합니다. 정책이 돌지 않는 구간이라
        LSTM 이 "명령해도 안 움직이는" 이력을 쌓을 일도 없습니다.
        기립 후 기본 자세는 Kp 20 으로도 버팁니다 (README 기술 사양 참고).
        """
        n = int(duration / C.CONTROL_DT)
        if n <= 0:
            return
        print("[HOLD] 게인 블렌딩 Kp %.0f/Kd %.1f -> %.0f/%.1f (%.1f s)" % (
            self.args.stand_kp, self.args.stand_kd, self.args.kp, self.args.kd, duration))
        for k in range(n):
            t0 = time.monotonic()
            if _stdin_pressed():
                raise KeyboardInterrupt
            state = self.robot.read_state()
            if not self._tilt_ok(state):
                raise RuntimeError("tilt guard during gain blend")
            b = _smoothstep((k + 1) / n)
            kp = self.args.stand_kp + (self.args.kp - self.args.stand_kp) * b
            kd = self.args.stand_kd + (self.args.kd - self.args.stand_kd) * b
            self.robot.send_positions(C.DEFAULT_JOINT_POS, kp, kd)
            self._static_log_step(t0, state, C.DEFAULT_JOINT_POS)
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
        print(f"[POLICY] 시작 cmd={cmd_target} Kp={self.args.kp:.0f}/Kd={self.args.kd:.1f} "
              "(Enter = 정지)")
        # recurrent(LSTM) 정책은 학습에서 에피소드가 hidden=0 으로 시작하므로,
        # 정책 인계 시점에 상태를 리셋합니다 (feed-forward / selftest 람다는 no-op).
        if hasattr(policy, "reset"):
            policy.reset()
        self._aux_ema = None  # 추정 평활도 에피소드마다 새로 시작
        log = None
        if getattr(self.args, "log_npz", None):
            log = {k: [] for k in ("t", "q", "dq", "q_des", "action",
                                   "ff", "quat", "gyro", "cmd", "h", "aux")}
        last_action = np.zeros(C.NUM_ACTIONS)
        n = int(duration / C.CONTROL_DT)
        try:
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
                est = policy.estimate() if hasattr(policy, "estimate") else None
                # 게인은 blend_gains() 가 인계 전에 이미 KP/KD 로 내려놓았습니다 —
                # 정책 행동이 기립 강성에서 실행되는 순간은 없습니다.
                self.robot.send_positions(q_des, self.args.kp, self.args.kd)

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
                    if est is not None:
                        log["aux"].append(np.concatenate([[est[0]], est[1]]))

                self._aux_update(est, C.CONTROL_DT)
                self._telemetry(time.monotonic(), state, action, cmd,
                                extra=self._aux_msg(k * C.CONTROL_DT))
                elapsed = time.monotonic() - t0
                if elapsed > C.CONTROL_DT * (1.0 + C.LOOP_OVERRUN_LIMIT):
                    print(f"[WARN] 루프 지연 {elapsed * 1000:.1f} ms")
                self._sleep_rest(t0)
        finally:
            # 기울임 가드 / Ctrl-C 로 루프가 끊겨도 로그는 남깁니다 — 낙상 직전
            # 데이터가 진단에 가장 중요합니다 (실측: 유실 사고 1회).
            if log is not None and log["t"]:
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
            self._static_log_step(t0, state, np.full(12, np.nan))
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
    ap.add_argument("--duration", type=float, default=20.0,
                    help="모드 본체의 실행 시간(s). dry-run=관측 출력, stand=기립 유지, "
                         "hang/walk=정책 실행. 기립(--stand-time)·lie_down 은 별도. "
                         "Enter 로 언제든 조기 종료. 처음엔 10 권장")
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
    ap.add_argument("--gain-blend", type=float, default=C.GAIN_BLEND_TIME,
                    help="정책 인계 *전* 기본 자세 홀드 중 STAND_KP->KP 블렌딩 시간(s). "
                         "정책은 첫 스텝부터 학습 게인(Kp20)에서 돕니다. 0=즉시 전환")
    ap.add_argument("--stand-kp", type=float, default=C.STAND_KP,
                    help="기립/유지 게인. 60=수평 기립(실측), 30=부드럽지만 뒷다리 처짐")
    ap.add_argument("--stand-kd", type=float, default=C.STAND_KD)
    ap.add_argument("--stand-time", type=float, default=5.0,
                    help="기립 보간 시간(s). 길수록 부드러움")
    ap.add_argument("--log-npz", default=None,
                    help="스텝별 상태를 .npz 로 저장. hang/walk 는 정책 구간(관측·행동 포함), "
                         "stand/dry-run 은 q/dq/ff/IMU. scripts/analyze_walk_log.py 로 요약")
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
        dep.state_warmup()
        dep.stand_up(duration=args.stand_time)
        dep.hold_default(1.0)
        if args.mode == "stand":
            dep.hold_default(args.duration)
        elif args.mode == "hang":
            dep.blend_gains(args.gain_blend)
            dep.run_policy(policy, np.zeros(3), args.duration)
        elif args.mode == "walk":
            cmd = clip_command(args.vx, args.vy, args.wz)
            dep.blend_gains(args.gain_blend)
            dep.run_policy(policy, cmd, args.duration)
        # 정상 종료(시간 만료 / 정책 중 Enter)
        # 예외 경로(기울임 가드, Ctrl-C)는 아래 finally 의 즉시 damping 
        dep.lie_down()
    except KeyboardInterrupt:
        print("[STOP] 사용자 중단")
    finally:
        if args.mode in ("stand", "dry-run"):
            dep.save_static_log()   # Ctrl-C / 가드로 끊겨도 저장
        if args.mode != "dry-run":
            print("[STOP] damping 모드로 종료")
            for _ in range(int(1.0 / C.CONTROL_DT)):
                robot.send_damping()
                if not args.mock:
                    time.sleep(C.CONTROL_DT)


if __name__ == "__main__":
    main()
