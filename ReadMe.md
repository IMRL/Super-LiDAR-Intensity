
### Super-LiDAR-Intensity

This is the code repository for the IEEE-RAL'26 paper "Super LiDAR Intensity for Robotic Perception"
[![Super LiDAR Intensity for Robotic Perception](cover.png)](https://youtu.be/C5PahLDyoVY)

Coming soon...

# Dataset Display

![Dataset 1](img/dataset_1_compressed.png)

![Dataset 2](img/dataset2.png)


## 3. Prerequisites
### 3.1. Ubuntu and ROS
This repository has been tested on **Ubuntu 20.04 + ROS Noetic**.

### 3.2. Dependencies
- [Opencv 4.2](https://github.com/opencv/opencv)
- Eigen3
- [PCL 1.13.1](https://github.com/PointCloudLibrary/pcl)
- [FAST-LIO2](https://github.com/hku-mars/FAST_LIO)
- [livox_ros_driver2](https://github.com/Livox-SDK/livox_ros_driver2)
- [LibTorch 2.4.0（cxx11-abi-shared）](https://docs.pytorch.org/cppdocs/installing.html)
- CUDA 12.4
- OpenMP
- DBoW3


## Usage:
### Data Preparation
**Collect ROSbag**

We provide code for conveniently collecting static Livox data. Please refer to [record_rosbag.cpp](https://github.com/IMRL/Super-LiDAR-Intensity/blob/main/Super/src/data_generate/record_rosbag.cpp).

**Generate intensity images from ROS bag**

The `Super` package provides tools to convert LiDAR point clouds into panoramic/virtual-camera intensity images for training and evaluation. 

```
roslaunch Super intensity_image.launch
```





### Reference

```
@article{gao2025super,
  title={Super LiDAR Intensity for Robotic Perception},
  author={Gao, Wei and Zhang, Jie and Zhao, Mingle and Zhang, Zhiyuan and Kong, Shu and Ghaffari, Maani and Song, Dezhen and Xu, Cheng-Zhong and Kong, Hui},
  journal={arXiv preprint arXiv:2508.10398},
  year={2025}
}
```

