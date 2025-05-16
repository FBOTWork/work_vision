# work_vision
## Overview

This is a group of ROS packages responsable for perform computer vision process of Fbot@Work industrial robot (Micky) in Robocup@Work league.

**Author: [Gabriel Torres], gabrieltlt721@gmail.com**
**Author: [Guilherme Costa], guilhermemano65667@gmail.com**

## Dependencies

This software is built on the Robotic Operating System (ROS Kinetic), which needs to be installed first. Additionally, the packages depends of a few libraries and frameworks:

...

## Packages
The vision system has main packages, helpers and 3rd helpers.

### Main Packages
- [work_behavior](https://github.com/FBOTWork/work_behavior)
- [work_manipulation](https://github.com/FBOTWork/work_manipulation)
- [apriltag_ros](https://github.com/FBOTWork/apriltag_ros)

### Helper Packages
- [apriltag](https://github.com/FBOTWork/apriltag)

### Helper 3rd Packages
- [realsense-ros](https://github.com/FBOTWork/realsense-ros)
- [ros_astra_camera](https://github.com/FBOTWork/ros_astra_camera)
- [iai_kinect2](https://github.com/FBOTWork/iai_kinect2)
- [libfreenect2](https://github.com/FBOTWork/libfreenect2)

## Clone

Clone this repository using the follow command:
```bash
git clone --recursive https://github.com/FBOTWork/work_vision.git
```

## Instalation

Run the follow commands:
```bash
source /opt/ros/$ROS_DISTRO/setup.bash
mkdir -p ~/work_ws/src
cd ~/work_ws/src
git clone https://github.com/FBOTWork/work_vision
cd ~/work_ws
rosdep install --from-paths src -iry
catkin build
```
