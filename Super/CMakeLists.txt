cmake_minimum_required(VERSION 3.0.2)
project(Super)

#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

# add_definitions(-DMP_EN)
# add_definitions(-DMP_PROC_NUM=4)
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  pcl_ros
  roscpp
  rospy
  std_msgs
  livox_ros_driver2
)

find_package(PCL REQUIRED)
find_package(OpenMP REQUIRED)
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES intensity_image
#  CATKIN_DEPENDS cv_bridge pcl_ros roscpp rospy std_msgs
#  DEPENDS system_lib
)

add_compile_options(${OpenMP_CXX_FLAGS})

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}  
  ${OpenCV_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
)


add_executable(intensity_image src/data_generate/intensity_image.cpp)
target_link_libraries(intensity_image ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenMP_CXX_LIBRARIES})

add_executable(record_rosbag src/data_generate/record_rosbag.cpp)
target_link_libraries(record_rosbag ${catkin_LIBRARIES})




