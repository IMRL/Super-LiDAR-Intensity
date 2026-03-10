
### Super-LiDAR-Intensity

This is the code repository for the IEEE-RAL'26 paper "Super LiDAR Intensity for Robotic Perception"
[![Super LiDAR Intensity for Robotic Perception](cover.png)](https://youtu.be/C5PahLDyoVY)

Coming soon...

# Dataset Display

![Dataset 1](img/dataset_1_compressed.png)

![Dataset 2](img/dataset2.png)


## Usage:
### Data Preparation

**Generate intensity / depth images from ROS bag**

The `Super` package provides tools to convert LiDAR point clouds into panoramic/virtual-camera intensity / depth images for training and evaluation. 
### Dependencies

This repository has been tested on **Ubuntu 20.04 + ROS Noetic**. It depends on two upstream ROS packages:

- **FAST-LIO2**: provides real-time LiDAR odometry and the registered point cloud / trajectory (`path_fast_lio`) used by `imaging_lidar_place_recognition` for loop detection and traffic lane detection.
- **livox_ros_driver2**: official Livox LiDAR ROS driver.


### Reference

```
@article{gao2025super,
  title={Super LiDAR Intensity for Robotic Perception},
  author={Gao, Wei and Zhang, Jie and Zhao, Mingle and Zhang, Zhiyuan and Kong, Shu and Ghaffari, Maani and Song, Dezhen and Xu, Cheng-Zhong and Kong, Hui},
  journal={arXiv preprint arXiv:2508.10398},
  year={2025}
}
```

