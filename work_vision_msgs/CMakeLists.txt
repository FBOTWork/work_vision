cmake_minimum_required(VERSION 2.8.3)
project(work_vision_msgs)

find_package(catkin REQUIRED COMPONENTS std_msgs vision_msgs geometry_msgs sensor_msgs genmsg message_generation)

add_message_files(
  FILES
  Description2D.msg
  Description3D.msg
  Level.msg
  Line.msg
  Recognitions2D.msg
  Recognitions3D.msg
  ClassifierReload.msg
  Shelf.msg
  KeyPoint2D.msg
  KeyPoint3D.msg
  )

add_service_files(
  FILES
  FaceClassifierTraining.srv
  ListClasses.srv
  PeopleIntroducing.srv
  LookAtDescription3D.srv
  SetClass.srv
  VisualQuestionAnswering.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
  vision_msgs
)

catkin_package(CATKIN_DEPENDS geometry_msgs sensor_msgs vision_msgs std_msgs message_runtime)

include_directories()