cmake_minimum_required(VERSION 2.8.3)
project(work_vision)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  message_generation
)

catkin_python_setup()

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  Description.msg
  Recognitions.msg
)

## Generate services in the 'srv' folder
#  add_service_files(
#    FILES
#    TelegramMSG.srv
#  )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
 generate_messages(
   DEPENDENCIES
   std_msgs
   sensor_msgs
 )

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs sensor_msgs rospy
#  INCLUDE_DIRS include
#  LIBRARIES butia_behavior
#  CATKIN_DEPENDS rospy std_msgs
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)