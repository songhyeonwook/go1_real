#!/usr/bin/env python3
"""Measure whether the policy is actually walking, and with how many legs.

Run while deploy_policy.py is executing the policy and a velocity command is being published:

    rostopic pub -r 10 /cmd_vel geometry_msgs/Twist '{linear: {x: 0.3}}' &
    rosrun go1_real check_gait.py 5
    rostopic pub -1 /cmd_vel geometry_msgs/Twist '{}'      # cmd_vel has no watchdog!

The gait is a closed-loop limit cycle: the policy moves the joints and the resulting state
change drives the next phase. Below a certain action authority the cycle damps out and the
robot holds a fixed pose. Measured on hardware, action_scale_multiplier=0.2 gives 0/4 legs
moving with every joint under 0.003 rad peak-to-peak.
"""
import sys
import numpy as np
import rospy
from unitree_legged_msgs.msg import LowCmd

# Unitree hardware order: FR, FL, RR, RL -- each (hip, thigh, calf)
LEGS = [("FR", 0), ("FL", 3), ("RR", 6), ("RL", 9)]
MOVING_RAD = 0.02

samples = []


def main():
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    rospy.init_node("check_gait", anonymous=True, disable_signals=True)
    rospy.Subscriber("/low_cmd", LowCmd,
                     lambda m: samples.append([m.motorCmd[i].q for i in range(12)]))
    t0 = rospy.Time.now().to_sec()
    while rospy.Time.now().to_sec() - t0 < dur:
        rospy.sleep(0.02)

    if len(samples) < 20:
        print("only %d /low_cmd samples -- is deploy_policy.py running?" % len(samples))
        return 1

    a = np.array(samples)
    print("samples=%d over %.1fs (%.1f Hz)\n" % (len(a), dur, len(a) / dur))
    print("%-5s %9s %9s %9s   %s" % ("leg", "hip p-p", "thigh p-p", "calf p-p", "moving?"))
    print("-" * 52)
    moving = 0
    for name, i in LEGS:
        pp = [float(np.ptp(a[:, i + j])) for j in range(3)]
        is_mv = max(pp) > MOVING_RAD
        moving += is_mv
        print("%-5s %9.4f %9.4f %9.4f   %s" % (name, pp[0], pp[1], pp[2], "YES" if is_mv else "no"))

    print("\n%d / 4 legs moving (threshold %.2f rad p-p)" % (moving, MOVING_RAD))
    if moving == 0:
        print("=> fixed point, no gait. Raise action_scale_multiplier.")
    elif moving < 4:
        print("=> PARTIAL/ASYMMETRIC gait. Stop and reassess before putting the robot down.")
    else:
        print("=> all four legs active.")

    # Rough gait period from the largest-amplitude joint's zero crossings.
    j = int(np.argmax(np.ptp(a, axis=0)))
    sig = a[:, j] - a[:, j].mean()
    if np.ptp(sig) > MOVING_RAD:
        cross = np.where(np.diff(np.sign(sig)) != 0)[0]
        if len(cross) > 2:
            period = 2.0 * np.mean(np.diff(cross)) * (dur / len(a))
            print("dominant joint %d: ~%.2f s period (%.2f Hz)" % (j, period, 1.0 / period))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
