#!/usr/bin/env python3
"""Measure how well the robot reached its target standing pose.

Run while deploy_policy.py is holding the stand (enable_policy:=false):

    rosrun go1_real check_stand.py
    python3 check_stand.py            # works too, if ROS is sourced

stand_up() is open-loop PD with no integral term and no gravity feedforward, so the residual
error on each joint is exactly tau_required / Kp. This prints both, so a bad stand tells you
immediately whether the gain is too low (error tracks tauEst/Kp) or something else is wrong.
"""
import numpy as np
import rospy
from unitree_legged_msgs.msg import LowState

# Isaac order: FL, FR, RL, RR -- each (hip, thigh, calf)
DEFAULT_ISAAC = np.array([0.1, 0.8, -1.5,
                          -0.1, 0.8, -1.5,
                          0.1, 1.0, -1.5,
                          -0.1, 1.0, -1.5])
I2U = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
TARGET = DEFAULT_ISAAC[I2U]
NAMES = ["FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf",
         "RR_hip", "RR_thigh", "RR_calf", "RL_hip", "RL_thigh", "RL_calf"]

state = {}


def cb(msg):
    if "q" in state:
        return
    state["q"] = np.array([msg.motorState[i].q for i in range(12)])
    state["dq"] = np.array([msg.motorState[i].dq for i in range(12)])
    state["tau"] = np.array([msg.motorState[i].tauEst for i in range(12)])
    state["quat"] = list(msg.imu.quaternion)


def main():
    rospy.init_node("check_stand", anonymous=True, disable_signals=True)
    rospy.Subscriber("/low_state", LowState, cb)
    t0 = rospy.Time.now().to_sec()
    while "q" not in state and rospy.Time.now().to_sec() - t0 < 5.0:
        rospy.sleep(0.05)
    if "q" not in state:
        print("No /low_state received. Is the lowlevel bridge running?")
        return 1

    q, dq, tau = state["q"], state["dq"], state["tau"]
    err = TARGET - q

    print("%-10s %8s %8s %8s %8s %8s" % ("joint", "target", "actual", "err", "dq", "tauEst"))
    print("-" * 60)
    for i in range(12):
        flag = "  <<<" if abs(err[i]) > 0.15 else ""
        print("%-10s %8.3f %8.3f %8.3f %8.3f %8.2f%s"
              % (NAMES[i], TARGET[i], q[i], err[i], dq[i], tau[i], flag))

    worst = int(np.argmax(np.abs(err)))
    print("\nmax |err| = %.3f rad at %s" % (abs(err[worst]), NAMES[worst]))

    # err == tauEst / Kp for a pure-P hold, so this recovers the gain actually in effect.
    mask = np.abs(err) > 0.02
    if mask.any():
        implied = np.median(np.abs(tau[mask] / err[mask]))
        print("implied Kp (median of tauEst/err over loaded joints) = %.1f" % implied)

    w, x, y, z = state["quat"]
    gz = 2 * (x * x + y * y) - 1
    print("projected_gravity_z = %.3f  (upright = -1.0)" % gz)
    if abs(err[worst]) > 0.3:
        print("=> POOR STAND. Raise stand_up_Kp (hardware: Kp=20 sags 0.66 rad, Kp=60 -> 0.13).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
