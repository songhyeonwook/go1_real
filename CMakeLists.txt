cmake_minimum_required(VERSION 3.0.2)
project(go1_real)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  rospy
  std_msgs
  unitree_legged_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/deploy_policy.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch model
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
