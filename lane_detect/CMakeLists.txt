cmake_minimum_required(VERSION 3.0.2)
project(lane_detect)

# 设置 C++17 标准和编译优化选项
set(CMAKE_BUILD_TYPE "Release")
add_compile_options(-std=c++17)
set(CMAKE_CXX_FLAGS "-std=c++17 -O3") 
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")

# libtorch: pass -DTorch_DIR=/path/to/libtorch/share/cmake/Torch when configuring
set(Torch_DIR "/home/gw/3rdPart/libtorch-cxx11-abi-shared-with-deps-2.4.0+cu124/libtorch/share/cmake/Torch")
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  pcl_ros
  roscpp
  rospy
  std_msgs
  livox_ros_driver2
)
find_package(PCL REQUIRED)
find_package(Torch REQUIRED)

catkin_package()


include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
)


set(NMS_SOURCES
  ${PROJECT_SOURCE_DIR}/src/nms/nms.cpp
  ${PROJECT_SOURCE_DIR}/src/nms/nms_kernel.cu
)


add_library(nms_lib SHARED ${NMS_SOURCES})
set_target_properties(nms_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(nms_lib PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(nms_lib "${TORCH_LIBRARIES}" ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})


# lane_detect 可执行文件，链接 NMS 库
# add_executable(laneAtt_detect src/lane_detect_static.cpp)
# target_link_libraries(laneAtt_detect ${catkin_LIBRARIES} ${PCL_LIBRARIES} "${TORCH_LIBRARIES}" nms_lib)

add_executable(laneAtt_detect src/lane_detect.cpp)
target_link_libraries(laneAtt_detect ${catkin_LIBRARIES} ${PCL_LIBRARIES} "${TORCH_LIBRARIES}" nms_lib)