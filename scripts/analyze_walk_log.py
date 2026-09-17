#!/usr/bin/env python3
"""deploy.py --log-npz 로그를 요약합니다 (NX 의 numpy 1.13 에서 돌아감).

  python3 scripts/analyze_walk_log.py walk1.npz     # hang/walk: 정책 구간
  python3 scripts/analyze_walk_log.py stand1.npz    # stand/dry-run: 정지 상태

정지 로그는 발힘 분포(대각 비대칭), 자세 바이어스, 관절별 정상상태 오차, dq 센서
노이즈를 냅니다 — 무게중심/영점/바닥 문제를 가르는 용도입니다.

루프 타이밍, 자세(roll/pitch/yaw 드리프트), PD 토크 추정(Kp/Kd 는 학습 게인),
관절 추종 오차, 행동 스펙트럼, 발 접촉(duty / 스텝 주파수 / 스윙 길이 / 대각 동기),
보조 헤드 속도 적분을 출력합니다. 접촉은 config.FOOT_FORCE_BIAS 를 뺀 값으로
판정하므로 bias 가 오래됐으면 dry-run 으로 다시 재세요.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "sdk_deploy"))
import config as C  # noqa: E402

CALF_TORQUE_LIMIT = 35.55   # Go1 calf (1.5:1 링크) — hip/thigh 는 C.TORQUE_LIMIT


def quat_to_rpy(quat_wxyz):
    w, x, y, z = quat_wxyz.T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return roll, pitch, yaw


def dq_noise_rms(dq, cutoff_hz=10.0):
    D = dq - dq.mean(0)
    F = np.fft.rfft(D, axis=0)
    f = np.fft.rfftfreq(len(D), C.CONTROL_DT)
    return np.sqrt((np.abs(F[f > cutoff_hz]) ** 2).sum(0) * 2) / len(D)


def main_static(d):
    t, q, dq, qd, ff, quat, gyro = [d[k] for k in ("t", "q", "dq", "q_des", "ff", "quat", "gyro")]
    T = t - t[0]
    dts = np.diff(t)
    print("== 정지 로그: %d steps, %.2f s, dt mean %.1f ms max %.1f ms ==" % (len(t), T[-1], dts.mean() * 1e3, dts.max() * 1e3))
    roll, pitch, yaw = quat_to_rpy(quat)
    print("== 자세 (IMU) ==")
    print("  roll  %+.3f rad (%+.1f deg)  std %.3f" % (roll.mean(), np.degrees(roll.mean()), roll.std()))
    print("  pitch %+.3f rad (%+.1f deg)  std %.3f" % (pitch.mean(), np.degrees(pitch.mean()), pitch.std()))
    print("  gyro bias %s rad/s" % np.round(gyro.mean(0), 4).tolist())
    fm = ff.mean(0)
    print("== 발힘 raw (FL FR RL RR) ==")
    print("  mean %s  std %s" % (np.round(fm).tolist(), np.round(ff.std(0), 1).tolist()))
    tot = fm.sum()
    print("  분율  %s" % np.round(fm / tot, 2).tolist())
    print("  대각 FL+RR %.2f  vs  FR+RL %.2f   | 좌 %.2f 우 %.2f | 앞 %.2f 뒤 %.2f"
          % ((fm[0] + fm[3]) / tot, (fm[1] + fm[2]) / tot, (fm[0] + fm[2]) / tot, (fm[1] + fm[3]) / tot,
             (fm[0] + fm[1]) / tot, (fm[2] + fm[3]) / tot))
    print("  (센서 스케일이 발마다 다를 수 있음 — 같은 로봇을 180도 돌린 로그와 비교할 것)")
    if not np.isnan(qd).any():
        e = qd - q
        print("== 관절 정상상태 오차 q_des-q (FL FR RL RR) ==")
        for j, name in enumerate(("hip", "thigh", "calf")):
            v = e[:, 4 * j:4 * j + 4].mean(0)
            print("  %-5s %s   L-R 앞 %+.3f 뒤 %+.3f" % (name, np.round(v, 3).tolist(), v[0] - v[1], v[2] - v[3]))
    print("== 관절각 q 평균 (FL FR RL RR) ==")
    for j, name in enumerate(("hip", "thigh", "calf")):
        print("  %-5s %s" % (name, np.round(q[:, 4 * j:4 * j + 4].mean(0), 3).tolist()))
    print("== dq 센서 노이즈 (>10 Hz rms, rad/s; 학습 노이즈 std 0.02) ==")
    print("  hip %s  thigh %s  calf %s"
          % tuple(np.round(dq_noise_rms(dq)[i:i + 4], 3).tolist() for i in (0, 4, 8)))


def main(path):
    d = np.load(path)
    if "action" not in d.files:
        return main_static(d)
    t, q, dq, qd, a = d["t"], d["q"], d["dq"], d["q_des"], d["action"]
    ff, quat, gyro, cmd = d["ff"], d["quat"], d["gyro"], d["cmd"]
    T = t - t[0]
    dts = np.diff(t)
    print("== 루프 ==")
    print("  %d steps, %.2f s, dt mean %.1f ms max %.1f ms, >30 ms: %d"
          % (len(t), T[-1], dts.mean() * 1e3, dts.max() * 1e3, (dts > 0.03).sum()))

    roll, pitch, yaw = quat_to_rpy(quat)
    print("== 자세 ==")
    for name, v in (("roll", roll), ("pitch", pitch)):
        print("  %-5s mean %+.3f std %.3f  [%+.3f, %+.3f] rad"
              % (name, v.mean(), v.std(), v.min(), v.max()))
    print("  yaw drift %+.3f rad / %.1f s = %+.3f rad/s (gyro z mean %+.3f)  -> --wz %+.2f 로 보정"
          % (yaw[-1] - yaw[0], T[-1], (yaw[-1] - yaw[0]) / T[-1], gyro[:, 2].mean(),
             -(yaw[-1] - yaw[0]) / T[-1]))

    tau = C.KP * (qd - q) - C.KD * dq
    print("== PD 토크 추정 (Kp %.0f / Kd %.1f) ==" % (C.KP, C.KD))
    print("  |tau|max hip %.1f / thigh %.1f (한계 %.1f)  calf %.1f (한계 %.1f) Nm ; 99pct %.1f"
          % (np.abs(tau[:, :4]).max(), np.abs(tau[:, 4:8]).max(), C.TORQUE_LIMIT,
             np.abs(tau[:, 8:]).max(), CALF_TORQUE_LIMIT, np.percentile(np.abs(tau), 99)))
    err = np.abs(qd - q)
    print("  추종 오차 |q_des-q| mean %.3f max %.3f rad" % (err.mean(), err.max()))
    print("  관절별 mean (FL FR RL RR): hip %s  thigh %s  calf %s"
          % tuple(np.round(err.mean(0)[i:i + 4], 2).tolist() for i in (0, 4, 8)))

    s = int(np.argmax(cmd[:, 0] >= 0.95 * cmd[:, 0].max())) if cmd[:, 0].max() > 0 else 0
    span = T[-1] - T[s]
    print("== 행동 (cmd 도달 후 %.1f s) ==" % span)
    A = a[s:] - a[s:].mean(0)
    F = np.abs(np.fft.rfft(A, axis=0)) ** 2
    f = np.fft.rfftfreq(len(A), C.CONTROL_DT)
    print("  |a|max %.2f mean|a| %.2f  |Δa|/step max %.2f mean %.2f  ; >5 Hz 파워 %.0f%%  주파수 %.2f Hz"
          % (np.abs(a).max(), np.abs(a).mean(), np.abs(np.diff(a, axis=0)).max(),
             np.abs(np.diff(a, axis=0)).mean(),
             100 * F[f > 5].sum() / F[f > 0].sum(), f[1:][F[1:].sum(1).argmax()]))

    c = (ff[s:] - C.FOOT_FORCE_BIAS) > C.CONTACT_FORCE_THRESHOLD
    print("== 접촉 (bias %s, thr %.0f) ==" % (C.FOOT_FORCE_BIAS.tolist(), C.CONTACT_FORCE_THRESHOLD))
    print("  raw ff mean %s  min %s" % (np.round(ff.mean(0)).tolist(), np.round(ff.min(0)).tolist()))
    for i, name in enumerate(C.LEG_NAMES):
        ci = c[:, i].astype(int)
        td = int((np.diff(ci) == 1).sum())
        sw, run = [], 0
        for v in ci:
            if v == 0:
                run += 1
            elif run:
                sw.append(run)
                run = 0
        print("  %s duty %.2f  steps %2d (%.2f Hz)  swing mean %3.0f ms max %3.0f ms"
              % (name, ci.mean(), td, td / span if span else 0,
                 np.mean(sw) * C.CONTROL_DT * 1e3 if sw else 0,
                 max(sw) * C.CONTROL_DT * 1e3 if sw else 0))
    n_on = c.sum(1)
    print("  지면 발 수: mean %.2f ; <=1: %.2f  2: %.2f  3: %.2f  4: %.2f"
          % ((n_on.mean(),) + tuple((n_on == k).mean() for k in (1, 2, 3, 4))))
    print("  대각 동기 FL~RR %.2f  FR~RL %.2f  (트로트 -> 1 에 가까움)"
          % ((c[:, 0] == c[:, 3]).mean(), (c[:, 1] == c[:, 2]).mean()))

    if "aux" in d.files and len(d["aux"]):
        v = d["aux"][s:, 1:]
        print("== 보조 헤드 v_hat (제어 미사용) ==")
        print("  mean (%+.2f, %+.2f, %+.2f) m/s ; x 적분 %.2f m / %.1f s"
              % (v[:, 0].mean(), v[:, 1].mean(), v[:, 2].mean(), v[:, 0].sum() * C.CONTROL_DT, span))


if __name__ == "__main__":
    main(sys.argv[1])
