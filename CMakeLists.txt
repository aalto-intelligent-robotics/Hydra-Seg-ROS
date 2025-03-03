cmake_minimum_required(VERSION 3.0.2)
project(hydra_seg_ros)

find_package(catkin REQUIRED COMPONENTS
  hydra_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
)

catkin_python_setup()

catkin_package(
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
catkin_install_python(PROGRAMS
  app/yolo_ros_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
