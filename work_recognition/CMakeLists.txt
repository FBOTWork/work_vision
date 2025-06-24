cmake_minimum_required(VERSION 2.8.3)
project(work_vision)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  std_srvs
  sensor_msgs
  work_vision_msgs
)

catkin_python_setup()

catkin_package(
  INCLUDE_DIRS
  LIBRARIES 
  CATKIN_DEPENDS std_msgs std_srvs sensor_msgs work_vision_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
install(PROGRAMS
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)