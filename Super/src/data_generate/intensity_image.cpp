
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <livox_ros_driver2/CustomMsg.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ros/ros.h>
#include <chrono>
#include <random>
#include <xmlrpcpp/XmlRpcValue.h>

#include <pcl/io/pcd_io.h> 

typedef pcl::PointXYZI PointType;

const float VERTICAL_FOV_MIN = -7.0f; 
const float VERTICAL_FOV_MAX = 52.0f;

int width_range = 1380;
int height_range = 240;
int width_four_view = 480;
int height_four_view = 240;
std::vector<int> frame_counts = {1, 5, 10, 500};
int num_groups = 5;
std::string livox_topic = "/livox/lidar";
/** Tilt angle of Livox LiDAR (deg), loaded from param rotation_angle_y_deg; rotation applied around Y axis before projection. */
float rotation_angle_y_deg = 25.0f;


void addNoiseToPointCloud(pcl::PointCloud<PointType>::Ptr& cloud) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> rotation_dist(-0.05, 0.05);  
    std::uniform_real_distribution<float> translation_dist(-0.02, 0.02); 

    Eigen::Affine3f rotation = Eigen::Affine3f::Identity();
    rotation.rotate(Eigen::AngleAxisf(rotation_dist(gen), Eigen::Vector3f::UnitX())); 
    rotation.rotate(Eigen::AngleAxisf(rotation_dist(gen), Eigen::Vector3f::UnitY())); 
    rotation.rotate(Eigen::AngleAxisf(rotation_dist(gen), Eigen::Vector3f::UnitZ())); 

    Eigen::Affine3f translation = Eigen::Affine3f::Identity();
    translation.translation() << translation_dist(gen), translation_dist(gen), translation_dist(gen);

    pcl::transformPointCloud(*cloud, *cloud, rotation * translation);
}


pcl::PointCloud<PointType>::Ptr rotatePointCloud(const pcl::PointCloud<PointType>::Ptr& cloud, double angle, const Eigen::Vector3f& axis) {
    pcl::PointCloud<PointType>::Ptr rotated_cloud(new pcl::PointCloud<PointType>());
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.rotate(Eigen::AngleAxisf(angle, axis));
    pcl::transformPointCloud(*cloud, *rotated_cloud, transform);
    return rotated_cloud;
}


cv::Mat generateRangeView(const pcl::PointCloud<PointType>::Ptr& cloud, int width, int height, cv::Mat& intensity_image, pcl::PointCloud<PointType>::Ptr& projected_cloud) 
{

    cv::Mat depth_image = cv::Mat::zeros(height, width, CV_32FC1);
    intensity_image = cv::Mat::zeros(height, width, CV_32FC1);

    cv::Mat min_depth_map = cv::Mat::ones(height, width, CV_32FC1) * std::numeric_limits<float>::max();

    float min_intensity = std::numeric_limits<float>::max();
    float max_intensity = std::numeric_limits<float>::lowest();

    float min_depth = std::numeric_limits<float>::max();
    float max_depth = std::numeric_limits<float>::lowest();
    
    projected_cloud->points.resize(width * height, PointType());


    for (const auto& point : cloud->points) {
        float depth = sqrt(point.x * point.x + point.y * point.y + point.z * point.z);

        float theta = atan2(point.y, point.x) * 180.0 / M_PI;  
        float phi = atan2(point.z, sqrt(point.x * point.x + point.y * point.y)) * 180.0 / M_PI; 

        int col = static_cast<int>((-theta + 180.0f) / 360.0f * width);  
        int row = static_cast<int>((VERTICAL_FOV_MAX - phi) / (VERTICAL_FOV_MAX - VERTICAL_FOV_MIN) * height); 

        if (col >= 0 && col < width && row >= 0 && row < height) {
            int idx = row * width + col; 

            if (depth < min_depth_map.at<float>(row, col)) {
      
                depth_image.at<float>(row, col) = depth;
                intensity_image.at<float>(row, col) = point.intensity;

                min_depth_map.at<float>(row, col) = depth;
                PointType& projected_point = projected_cloud->points[idx];
                projected_point.x = point.x;
                projected_point.y = point.y;
                projected_point.z = point.z;
                projected_point.intensity = point.intensity;

                if (point.intensity < min_intensity) min_intensity = point.intensity;
                if (point.intensity > max_intensity) max_intensity = point.intensity;

                if (depth < min_depth) min_depth = depth;
                if (depth > max_depth) max_depth = depth;
            }
        }
    }

    std::cout << "Min intensity: " << min_intensity << " Max intensity: " << max_intensity << std::endl;

    if (max_intensity > min_intensity) {
        cv::normalize(intensity_image, intensity_image, 0.0, 1.0, cv::NORM_MINMAX);
    } else {
        intensity_image.setTo(0.0f);
    }

    if (max_depth > min_depth) {
        cv::normalize(depth_image, depth_image, 0.0, 1.0, cv::NORM_MINMAX);
    } else {
        depth_image.setTo(0.0f);
    }

    return depth_image;
}


cv::Mat generateImage(const pcl::PointCloud<PointType>::Ptr& cloud, int width, int height, bool use_intensity)
{
    const float vertical_fov_min = -32.0f * M_PI / 180.0f;  
    const float vertical_fov_max = 27.0f * M_PI / 180.0f; 
    const float vertical_fov = vertical_fov_max - vertical_fov_min;

    float fx = width / (2 * tan(120 * M_PI / 360)); 
    float cx = width / 2;

    float fy = height / vertical_fov;
    float cy = height * (vertical_fov_max / vertical_fov);  
    float min_intensity = std::numeric_limits<float>::max();
    float max_intensity = std::numeric_limits<float>::lowest();


    cv::Mat image = cv::Mat::zeros(height, width, CV_32FC1);
    for (const auto& point : cloud->points)
    {
        if (point.x <= 0) continue;

        float value = use_intensity ? point.intensity : sqrt(point.x * point.x + point.y * point.y);


        int col = static_cast<int>(-fx * point.y / point.x + cx);  

        float theta_vertical = atan2(point.z, point.x); 
        int row = static_cast<int>((theta_vertical - vertical_fov_min) / vertical_fov * height); 
        row = height - 1 - row; 
        if (col >= 0 && col < width && row >= 0 && row < height)
        {
            image.at<float>(row, col) = value;
            // points_in_fov++;
            if (point.intensity < min_intensity) min_intensity = point.intensity;
            if (point.intensity > max_intensity) max_intensity = point.intensity;
        }
    }
    std::cout << "Min intensity: " << min_intensity << " Max intensity: " << max_intensity << std::endl;
    cv::normalize(image, image, 0.0, 1.0, cv::NORM_MINMAX);

    return image;
}

pcl::PointCloud<PointType>::Ptr extractPointCloud(const livox_ros_driver2::CustomMsg::ConstPtr& msg) {
    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
    for (const auto& point : msg->points) {
        PointType pcl_point;
        pcl_point.x = point.x;
        pcl_point.y = point.y;
        pcl_point.z = point.z;
        pcl_point.intensity = point.reflectivity;
        cloud->points.push_back(pcl_point);
    }
    return cloud;
}

std::vector<rosbag::MessageInstance*> randomlySelectFrames(const std::vector<rosbag::MessageInstance>& all_frames, int num_frames) {
    std::vector<rosbag::MessageInstance*> frame_pointers;

    for (const auto& frame : all_frames) {
        frame_pointers.push_back(const_cast<rosbag::MessageInstance*>(&frame));
    }

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(frame_pointers.begin(), frame_pointers.end(), g);

    frame_pointers.resize(num_frames);

    return frame_pointers;
}

pcl::PointCloud<PointType>::Ptr processFrames(const std::vector<rosbag::MessageInstance*>& frames, int frame_count, bool add_noise) {
    pcl::PointCloud<PointType>::Ptr accumulated_cloud(new pcl::PointCloud<PointType>);

    for (const auto* frame : frames) {
        livox_ros_driver2::CustomMsg::ConstPtr msg = frame->instantiate<livox_ros_driver2::CustomMsg>();
        if (msg != nullptr) {
            pcl::PointCloud<PointType>::Ptr cloud = extractPointCloud(msg);

            if (add_noise) {
                addNoiseToPointCloud(cloud);
            }

            *accumulated_cloud += *cloud;

            if (--frame_count <= 0) {
                break;
            }
        }
    }

    return accumulated_cloud;
}


void loadParams(const ros::NodeHandle& nh) {
    nh.param<int>("width_range", width_range, 1380);
    nh.param<int>("height_range", height_range, 240);
    nh.param<int>("width_four_view", width_four_view, 480);
    nh.param<int>("height_four_view", height_four_view, 240);
    nh.param<int>("num_groups", num_groups, 5);
    nh.param<std::string>("livox_topic", livox_topic, "/livox/lidar");
    nh.param<float>("rotation_angle_y_deg", rotation_angle_y_deg, 25.0f);
    XmlRpc::XmlRpcValue list;
    if (nh.getParam("frame_counts", list) && list.getType() == XmlRpc::XmlRpcValue::TypeArray) {
        frame_counts.clear();
        for (int i = 0; i < list.size(); ++i)
            frame_counts.push_back(static_cast<int>(list[i]));
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "livox_intensity_range_view");
    ros::NodeHandle nh("~");

    std::string bag_folder_path;
    if (!nh.getParam("bag_folder_path", bag_folder_path) || bag_folder_path.empty()) {
        ROS_ERROR("Need private param ~bag_folder_path (path to folder containing .bag files). Set in launch: <param name=\"bag_folder_path\" value=\"...\" />");
        return 1;
    }

    loadParams(nh);

    for (const auto& entry : boost::filesystem::directory_iterator(bag_folder_path)) {
        if (entry.path().extension() == ".bag") {
            std::string bag_file = entry.path().string();
            rosbag::Bag bag;
            bag.open(bag_file, rosbag::bagmode::Read);

            std::vector<std::string> topics = {livox_topic};
            rosbag::View view(bag, rosbag::TopicQuery(topics));

            std::vector<rosbag::MessageInstance> all_frames(view.begin(), view.end());

        for (int frame_count : frame_counts) {
            bool is_sparse = (frame_count == 5 || frame_count == 10);

            for (int group_idx = 0; group_idx < num_groups; ++group_idx) {
                std::vector<rosbag::MessageInstance*> selected_frames;

                if (is_sparse) {
                    selected_frames = randomlySelectFrames(all_frames, frame_count);
                } else {
                    for (size_t i = 0; i < std::min(frame_count, static_cast<int>(all_frames.size())); ++i) {
                        selected_frames.push_back(&all_frames[i]);
                    }
                }

                pcl::PointCloud<PointType>::Ptr accumulated_cloud = processFrames(selected_frames, frame_count, is_sparse);

                if (accumulated_cloud->points.empty()) {
                    ROS_WARN("No points accumulated for frame count %d in group %d", frame_count, group_idx);
                    continue;
                }

                std::string base_output_dir = bag_folder_path + "/frames_" + std::to_string(frame_count);
                std::string depth_output_dir = base_output_dir + "/depth_view";
                std::string intensity_output_dir = base_output_dir + "/intensity_view";

                std::string front_view_output_dir = base_output_dir + "/front_view";
                std::string left_view_output_dir = base_output_dir + "/left_view";
                std::string right_view_output_dir = base_output_dir + "/right_view";
                std::string back_view_output_dir = base_output_dir + "/back_view";

                boost::filesystem::create_directories(depth_output_dir);
                boost::filesystem::create_directories(intensity_output_dir);
                boost::filesystem::create_directories(front_view_output_dir);
                boost::filesystem::create_directories(left_view_output_dir);
                boost::filesystem::create_directories(right_view_output_dir);
                boost::filesystem::create_directories(back_view_output_dir);

                std::string file_prefix = boost::filesystem::path(bag_file).stem().string() + "_group_" + std::to_string(group_idx);

                pcl::PointCloud<PointType>::Ptr projected_cloud(new pcl::PointCloud<PointType>);
                cv::Mat intensity_image;
                cv::Mat depth_image = generateRangeView(accumulated_cloud, width_range, height_range, intensity_image, projected_cloud);

                std::string depth_image_file = depth_output_dir + "/" + file_prefix + "_depth_range_view.png";
                depth_image.convertTo(depth_image, CV_8UC1, 255.0);
                cv::imwrite(depth_image_file, depth_image);

                std::string intensity_image_file = intensity_output_dir + "/" + file_prefix + "_intensity_range_view.png";
                intensity_image.convertTo(intensity_image, CV_8UC1, 255.0);
                cv::imwrite(intensity_image_file, intensity_image);

                float angle_rad = rotation_angle_y_deg * static_cast<float>(M_PI) / 180.0f;
                Eigen::Affine3f rotate_y = Eigen::Affine3f::Identity();
                rotate_y.rotate(Eigen::AngleAxisf(angle_rad, Eigen::Vector3f::UnitY()));
                pcl::PointCloud<PointType>::Ptr rotated_cloud(new pcl::PointCloud<PointType>);
                pcl::transformPointCloud(*accumulated_cloud, *rotated_cloud, rotate_y);

                pcl::PointCloud<PointType>::Ptr left_cloud = rotatePointCloud(rotated_cloud, -M_PI_2, Eigen::Vector3f::UnitZ());
                pcl::PointCloud<PointType>::Ptr right_cloud = rotatePointCloud(rotated_cloud, M_PI_2, Eigen::Vector3f::UnitZ());
                pcl::PointCloud<PointType>::Ptr back_cloud = rotatePointCloud(rotated_cloud, M_PI, Eigen::Vector3f::UnitZ());

                cv::Mat intensity_front_view = generateImage(rotated_cloud, width_four_view, height_four_view, true);
                cv::Mat intensity_left_view = generateImage(left_cloud, width_four_view, height_four_view, true);
                cv::Mat intensity_right_view = generateImage(right_cloud, width_four_view, height_four_view, true);
                cv::Mat intensity_back_view = generateImage(back_cloud, width_four_view, height_four_view, true);


                intensity_front_view.convertTo(intensity_front_view, CV_8UC1, 255.0 ); 
                intensity_left_view.convertTo(intensity_left_view, CV_8UC1, 255.0 );
                intensity_right_view.convertTo(intensity_right_view, CV_8UC1, 255.0 );
                intensity_back_view.convertTo(intensity_back_view, CV_8UC1, 255.0);

                cv::imwrite(front_view_output_dir + "/" + file_prefix + "_intensity_front.png", intensity_front_view);
                cv::imwrite(left_view_output_dir + "/" + file_prefix + "_intensity_left.png", intensity_left_view);
                cv::imwrite(right_view_output_dir + "/" + file_prefix + "_intensity_right.png", intensity_right_view);
                cv::imwrite(back_view_output_dir + "/" + file_prefix + "_intensity_back.png", intensity_back_view);

                std::cout << "Generated range view images for " << bag_file << " with " << frame_count << " frames in group " << group_idx << "." << std::endl;
            }
        }

            bag.close();
        }
    }

    return 0;
}