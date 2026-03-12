#include <ros/ros.h>
#include <ros/package.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <livox_ros_driver2/CustomMsg.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <Eigen/Geometry>
#include <deque>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <tuple>
#include <iostream>

#include <omp.h>
#include "lane_detect/nms.h"


using PointType = pcl::PointXYZI;

struct Pose6D {
  double x;
  double y;
  double z;
  double roll;
  double pitch;
  double yaw;
};
struct Lane {
    std::vector<std::pair<double, double>> points;
    double start_x;
    double start_y;
    double conf;
};
const float HORIZONTAL_FOV = 120.0f;
const float VERTICAL_FOV = 60.0f;

bool image_process = true;
bool saveImg = true;

int frame_window = 5;

int frameCount = 0;
int width = 480;
int height = 240;

int img_w = 640;
int n_strips = 71;
torch::Tensor anchor_ys;
std::deque<std::pair<livox_ros_driver2::CustomMsg, nav_msgs::Odometry>> cloud_odom_queue; 

std::string lidar_topic, odom_topic, camera_topic;
ros::Publisher pubIntensityImage, pubLeftImage, pubRightImage, pubBackImage,pubDepthImage,pubIntensityImageDense,pubIntensityComImageDense,pubDepthImageDense,pubLaneImage;

torch::jit::script::Module model;
std::string model_path;
std::string imagesavePath;

torch::jit::script::Module lane_detect_model;
std::string lane_model_path;


void load_model()
{
    try {
        model = torch::jit::load(model_path);
        if (torch::cuda::is_available()) {
            model.to(torch::kCUDA);
            std::cout << "GPU Availble!!!!!!" <<std::endl;
        }
        model.eval();
        std::cout << "Model loaded successfully." << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        exit(-1);
    }
}


void load_lane_detect_model() {
    try {
        lane_detect_model = torch::jit::load(lane_model_path);
        if (torch::cuda::is_available()) {
            lane_detect_model.to(torch::kCUDA);
            std::cout << "Lane detection model loaded on GPU!" << std::endl;
        } else {
            lane_detect_model.to(torch::kCPU);
            std::cout << "Lane detection model loaded on CPU!" << std::endl;
        }
        lane_detect_model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the lane detection model: " << e.what() << std::endl;
        exit(-1);
    }
}
cv::Mat sharp_laplacian(const cv::Mat& img) {
    cv::Mat laplacian, sharpened_image;
    cv::Laplacian(img, laplacian, CV_32F, 3);
    cv::Mat abs_laplacian;
    cv::convertScaleAbs(laplacian, abs_laplacian);
    cv::addWeighted(img, 1.0, abs_laplacian, 0.0, 0.0, sharpened_image,CV_32F);
    cv::Mat output;
    cv::cvtColor(sharpened_image, output, cv::COLOR_GRAY2RGB);
    std::cout<<"Output Channel: " << output.channels()  <<" Output Size" << output.size() << " output type: " << cv::typeToString(output.type()) <<std::endl;
    return output;
}


void filterBrightness(cv::Mat& img, cv::Size& window_size)
{
    cv::Mat brightness;
    cv::blur(img, brightness, window_size);
    brightness += 1;
    cv::Mat normalized_img = (140. * img / brightness);
    cv::Mat smoothed;
    cv::GaussianBlur(normalized_img, img, cv::Size(3, 3), 0);
}

pcl::PointCloud<PointType>::Ptr global2local(const pcl::PointCloud<PointType>::Ptr &cloudIn, const Pose6D& pose)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
    Eigen::Affine3f transToLocal = pcl::getTransformation(pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw);
    pcl::transformPointCloud(*cloudIn, *cloudOut, transToLocal.inverse());

    return cloudOut;
}
pcl::PointCloud<PointType>::Ptr local2global(const pcl::PointCloud<PointType>::Ptr &cloud, const Pose6D &pose)
{
    pcl::PointCloud<PointType>::Ptr cloud_global(new pcl::PointCloud<PointType>());
    Eigen::Affine3f trans_to_global = pcl::getTransformation(pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw);
    pcl::transformPointCloud(*cloud, *cloud_global, trans_to_global);
    return cloud_global;
}
Pose6D getOdom(nav_msgs::Odometry _odom)
{
    auto tx = _odom.pose.pose.position.x;
    auto ty = _odom.pose.pose.position.y;
    auto tz = _odom.pose.pose.position.z;

    double roll, pitch, yaw;
    geometry_msgs::Quaternion quat = _odom.pose.pose.orientation;
    tf::Matrix3x3(tf::Quaternion(quat.x, quat.y, quat.z, quat.w)).getRPY(roll, pitch, yaw);

    return Pose6D{tx, ty, tz, roll, pitch, yaw};
}


cv::Mat generateImage(const pcl::PointCloud<PointType>::Ptr& cloud, int width, int height, bool use_intensity)
{
    float fx = width / (2 * tan(120 * M_PI / 360)); 
    float cx = width / 2;
    const float vertical_fov_min = -32.0f * M_PI / 180.0f;
    const float vertical_fov_max = 27.0f * M_PI / 180.0f;
    const float vertical_fov = vertical_fov_max - vertical_fov_min;
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
            if (point.intensity < min_intensity) min_intensity = point.intensity;
            if (point.intensity > max_intensity) max_intensity = point.intensity;
        }
    }
    cv::normalize(image, image, 0.0, 1.0, cv::NORM_MINMAX);

    if (image_process)
    {
        cv::Size window_size_ = cv::Size(10, 10);
        filterBrightness(image, window_size_);
    }
    return image;
}

pcl::PointCloud<PointType>::Ptr rotatePointCloud(const pcl::PointCloud<PointType>::Ptr &cloudIn, float angle, const Eigen::Vector3f &axis)
{
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.rotate(Eigen::AngleAxisf(angle, axis));
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
    pcl::transformPointCloud(*cloudIn, *cloudOut, transform);
    return cloudOut;
}
torch::Tensor preprocess_images(const cv::Mat& depth_image, const cv::Mat& intensity_image) 
{
    cv::Mat resized_depth, resized_intensity;
    cv::resize(depth_image, resized_depth, cv::Size(480, 240));
    cv::resize(intensity_image, resized_intensity, cv::Size(480, 240));

    cv::Mat float_depth, float_intensity;
    resized_depth.convertTo(float_depth, CV_32F);
    resized_intensity.convertTo(float_intensity, CV_32F);

    double min_val, max_val;
    cv::minMaxLoc(float_depth, &min_val, &max_val);
    float_depth = (float_depth - min_val) / (max_val - min_val);

    cv::minMaxLoc(float_intensity, &min_val, &max_val);
    float_intensity = (float_intensity - min_val) / (max_val - min_val);

    torch::Tensor tensor_depth = torch::from_blob(float_depth.data, {240, 480}, torch::kFloat32).clone();
    torch::Tensor tensor_intensity = torch::from_blob(float_intensity.data, {240, 480}, torch::kFloat32).clone();

    torch::Tensor tensor_image = torch::stack({tensor_intensity, tensor_depth}, 0).unsqueeze(0);

    return tensor_image;
}

std::tuple<cv::Mat, cv::Mat, cv::Mat> run_model(const cv::Mat& depth_image, const cv::Mat& intensity_image) 
{
    torch::Tensor input_tensor = preprocess_images(depth_image, intensity_image);
    if (torch::cuda::is_available()) {
        input_tensor = input_tensor.to(torch::kCUDA);
        std::cout << "Running on GPU!!!!!!" << std::endl;
    } else {
        input_tensor = input_tensor.to(torch::kCPU);
        std::cout << "Running on CPU!!!!!!" << std::endl;
    }

    auto output_tuple = model.forward({input_tensor}).toTuple();

    if (output_tuple->elements().size() != 2) {
        throw std::runtime_error("Model output Tuple does not contain exactly 2 elements!");
    }

    torch::Tensor output1 = output_tuple->elements()[0].toTensor().to(torch::kCPU);
    torch::Tensor output2 = output_tuple->elements()[1].toTensor().to(torch::kCPU);
    cv::Mat depth_out1(cv::Size(480, 240), CV_32FC1, output1[0][1].data_ptr<float>());
    cv::Mat intensity_out1(cv::Size(480, 240), CV_32FC1, output1[0][0].data_ptr<float>());
    cv::Mat intensity_out2(cv::Size(480, 240), CV_32FC1, output2[0][0].data_ptr<float>());
    return {depth_out1.clone(), intensity_out1.clone(), intensity_out2.clone()};
}

cv::Mat normalizeAndConvertTo8UC1(const cv::Mat& input) {
    cv::Mat normalized;
    cv::normalize(input, normalized, 0, 255, cv::NORM_MINMAX); 
    normalized.convertTo(normalized, CV_8UC1);
    return normalized;
}

torch::Tensor preprocess_lane_image(const cv::Mat& intensity_image, cv::Mat& float_image) {
    if (intensity_image.empty()) {
        throw std::runtime_error("Input image is empty!");
    }

    cv::Mat blurred_image;
    cv::blur(intensity_image, blurred_image, cv::Size(5, 5));
    cv::Mat sharpened_image = sharp_laplacian(blurred_image);
    cv::Mat resized_image;
    cv::resize(sharpened_image, resized_image, cv::Size(640, 360));
    resized_image.convertTo(float_image, CV_32F);
    torch::Tensor tensor_img = torch::from_blob(
        float_image.clone().data, {float_image.rows, float_image.cols, 3}, torch::TensorOptions().dtype(torch::kFloat32)
    );
    tensor_img = tensor_img.permute({2, 0, 1});
    tensor_img = tensor_img.unsqueeze(0);
    return tensor_img.clone();
}
std::vector<at::Tensor> nms_forward(
    at::Tensor boxes,
    at::Tensor scores,
    float thresh,
    unsigned long top_k);

std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> nms_wrapper(
    const torch::Tensor& batch_proposals, 
    const torch::Tensor& batch_attention_matrix, 
    const torch::Tensor& anchors, 
    float nms_thres, 
    int nms_topk, 
    float conf_threshold) {

    std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> proposals_list;

    for (int i = 0; i < batch_proposals.size(0); ++i) {
        auto proposals = batch_proposals[i];
        auto attention_matrix = batch_attention_matrix[i];
        auto anchor_inds = torch::arange(batch_proposals.size(1), proposals.device());

        auto scores = torch::softmax(proposals.index({torch::indexing::Slice(), torch::indexing::Slice(0, 2)}), 1).select(1, 1);
        if (conf_threshold > 0.0) {
            auto above_threshold = scores > conf_threshold;
            proposals = proposals.index({above_threshold});
            scores = scores.index({above_threshold});
            anchor_inds = anchor_inds.index({above_threshold});
        }

        if (proposals.size(0) == 0) {
            proposals_list.emplace_back(
                torch::empty({0}, proposals.options()),
                torch::empty({0}, anchors.options()),
                torch::empty({0}, attention_matrix.options()),
                torch::empty({0}, anchor_inds.options())
            );
            continue;
        }

        auto nms_output = nms_forward(proposals, scores, nms_thres, nms_topk);
        auto keep = nms_output[0].index({torch::indexing::Slice(0, nms_output[1].item<int>())});
        proposals = proposals.index({keep});
        anchor_inds = anchor_inds.index({keep});
        attention_matrix = attention_matrix.index({anchor_inds});

        proposals_list.emplace_back(proposals, anchors.index({keep}), attention_matrix, anchor_inds);
    }
    return proposals_list;
}
std::vector<std::pair<double, double>> to_array(
    const Lane& lane, 
    int img_w = 640, 
    int img_h = 360) 
{
 
    std::vector<double> sample_y;
    for (int y = 710; y >= 150; y -= 10) {
        sample_y.push_back(static_cast<double>(y));
    }

    std::vector<double> ys(sample_y.size());
    for (size_t i = 0; i < sample_y.size(); ++i) {
        ys[i] = sample_y[i] / static_cast<double>(img_h);  
    }


    std::vector<double> lane_xs(ys.size(), -2.0); 
    for (size_t i = 0; i < ys.size(); ++i) {
        double y = ys[i];
        for (size_t j = 1; j < lane.points.size(); ++j) {
            double y1 = lane.points[j - 1].second;
            double y2 = lane.points[j].second;
            if (y1 <= y && y <= y2) {

                double x1 = lane.points[j - 1].first;
                double x2 = lane.points[j].first;
                lane_xs[i] = x1 + (x2 - x1) * (y - y1) / (y2 - y1);
                break;
            }
        }
    }

    std::vector<std::pair<double, double>> valid_points;
    for (size_t i = 0; i < lane_xs.size(); ++i) {
        if (lane_xs[i] >= 0.0 && lane_xs[i] <= 1.0) {
            double x_pixel = lane_xs[i] * img_w;  
            double y_pixel = sample_y[i];       
            valid_points.emplace_back(x_pixel, y_pixel);
        }
    }

    return valid_points;
}

std::vector<Lane> proposals_to_pred(const torch::Tensor& proposals) {
    std::vector<Lane> lanes;
    auto anchor_ys_device = anchor_ys.to(proposals.device()).to(torch::kDouble);

    for (int i = 0; i < proposals.size(0); ++i) {
        auto lane = proposals[i];
        auto lane_xs = lane.index({torch::indexing::Slice(5, torch::indexing::None)}) / img_w;
        int start = static_cast<int>(std::round(lane[2].item<double>() * n_strips));
        int length = static_cast<int>(std::round(lane[4].item<double>()));

        int end = start + length - 1;
        end = std::min(end, static_cast<int>(anchor_ys_device.size(0)) - 1);

        if (end >= lane_xs.size(0)) {
            std::cerr << "Error: End index out of range. End: " << end
                    << ", Lane_xs size: " << lane_xs.size(0) << std::endl;
            continue;
        }
        lane_xs.index_put_({torch::indexing::Slice(end + 1, torch::indexing::None)}, -2);
        if (start > lane_xs.size(0) || start < 0) {
            std::cerr << "Error: Start index out of range. Start: " << start
                    << ", Lane_xs size: " << lane_xs.size(0) << std::endl;
            continue;
        }
        auto slice = lane_xs.index({torch::indexing::Slice(0, start)});

        auto mask = (~(((slice >= 0) & (slice <= 1)))
                        .flip(0)
                        .cumprod(0)
                        .flip(0)
                        .to(torch::kBool));

        if (mask.dim() != 1 || mask.size(0) != slice.size(0)) {
            std::cerr << "Error: Mask dimension or size mismatch. "
                    << "Mask dim: " << mask.dim()
                    << ", Mask size: " << mask.size(0)
                    << ", Slice size: " << slice.size(0) << std::endl;
            continue;
        }

        slice.index_put_({mask}, -2);
        lane_xs.index_put_({torch::indexing::Slice(0, start)}, slice);

        auto valid_mask = lane_xs >= 0;
        auto valid_lane_xs = lane_xs.index({valid_mask});
        auto valid_lane_ys = anchor_ys_device.index({valid_mask});

        valid_lane_xs = valid_lane_xs.flip(0).to(torch::kDouble);
        valid_lane_ys = valid_lane_ys.flip(0).to(torch::kDouble);
        if (valid_lane_xs.size(0) <= 1) {
            continue;
        }

        std::vector<std::pair<double, double>> points;
        for (int j = 0; j < valid_lane_xs.size(0); ++j) {
            points.emplace_back(valid_lane_xs[j].item<double>(), valid_lane_ys[j].item<double>());
        }
        lanes.push_back({
            points,
            lane[3].item<double>(),
            lane[2].item<double>(),
            lane[1].item<double>()
        });
    }

    return lanes;
}

std::vector<std::vector<Lane>> get_lanes(const std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>& proposals_list) {
    std::vector<std::vector<Lane>> all_lanes;

    for (size_t i = 0; i < proposals_list.size(); ++i) {
        const auto& proposals = std::get<0>(proposals_list[i]);

        if (proposals.size(0) == 0) {
            all_lanes.emplace_back();
            continue;
        }
        auto updated_proposals = proposals.clone();
        auto scores = torch::softmax(updated_proposals.index({torch::indexing::Slice(), torch::indexing::Slice(0, 2)}), 1);
        updated_proposals.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, 2)}, scores);
        updated_proposals.index_put_({torch::indexing::Slice(), 4}, torch::round(updated_proposals.index({torch::indexing::Slice(), 4})));

        auto lanes = proposals_to_pred(updated_proposals);
        all_lanes.push_back(lanes);
    }
    return all_lanes;
}

void imshow_lanes(cv::Mat& img, const std::vector<Lane>& lanes) {
    for (const auto& lane : lanes) {
        for (const auto& [x, y] : lane.points) {
            if (x <= 0 || y <= 0) continue;
            cv::circle(img, cv::Point(static_cast<int>(x), static_cast<int>(y)), 4, cv::Scalar(0, 0, 255), 2);
        }
    }
}

cv::Mat run_lane_detect_model(const cv::Mat& intensity_image) {
    cv::Mat preprocess_image;
    torch::Tensor img_tensor = preprocess_lane_image(intensity_image, preprocess_image);

    if (torch::cuda::is_available()) {
        img_tensor = img_tensor.to(torch::kCUDA);
    } else {
        img_tensor = img_tensor.to(torch::kCPU);
    }

    c10::Dict<std::string, torch::Tensor> batch;
    batch.insert("img", img_tensor);
    torch::IValue input_dict = batch;

    auto output = lane_detect_model.forward({input_dict}).toTuple();

    torch::Tensor reg_proposals = output->elements()[0].toTensor();
    torch::Tensor attention_matrix = output->elements()[1].toTensor();
    torch::Tensor anchors = output->elements()[2].toTensor();

    float nms_thres = 45;
    int nms_topk = 5;
    float conf_threshold = 0.2;

    auto proposals_list = nms_wrapper(reg_proposals, attention_matrix, anchors, nms_thres, nms_topk, conf_threshold);
    auto all_lanes = get_lanes(proposals_list);

    std::vector<Lane> pixel_lane_objects;

    for (size_t i = 0; i < all_lanes.size(); ++i) {
        const auto& lanes = all_lanes[i];
        for (const auto& lane : lanes) {
            Lane pixel_lane;
            pixel_lane.points = to_array(lane);
            pixel_lane_objects.push_back(pixel_lane);
        }
    }

    cv::Mat result_image = preprocess_image.clone();
    imshow_lanes(result_image, pixel_lane_objects);
    return result_image;
}




void cloud_handler(const nav_msgs::Odometry::ConstPtr &odom_msg, const livox_ros_driver2::CustomMsg::ConstPtr &livox_msg, const sensor_msgs::CompressedImageConstPtr &image_msg_camera)
{

    cloud_odom_queue.push_back(std::make_pair(*livox_msg, *odom_msg));

    while (cloud_odom_queue.size() > frame_window)
    {
        cloud_odom_queue.pop_front();
    }

    pcl::PointCloud<PointType>::Ptr accumulated_cloud_global(new pcl::PointCloud<PointType>);
    for (const auto &pair : cloud_odom_queue)
    {
        const auto &livox_msg = pair.first;

        pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
        for (const auto &point : livox_msg.points)
        {
            PointType pcl_point;
            pcl_point.x = point.x;
            pcl_point.y = point.y;
            pcl_point.z = point.z;
            pcl_point.intensity = point.reflectivity; 
            cloud->points.push_back(pcl_point);
        }

        *accumulated_cloud_global += *cloud;
    }

    pcl::PointCloud<PointType>::Ptr accumulated_cloud_local(new pcl::PointCloud<PointType>);
    Pose6D odom_first= getOdom(cloud_odom_queue.front().second);; 
    auto start_time = std::chrono::high_resolution_clock::now();
    *accumulated_cloud_local = *global2local(accumulated_cloud_global, odom_first);

    Eigen::Affine3f rotate_y = Eigen::Affine3f::Identity();
    rotate_y.rotate(Eigen::AngleAxisf(M_PI / 180 * 25, Eigen::Vector3f::UnitY()));

    pcl::transformPointCloud(*accumulated_cloud_local, *accumulated_cloud_local, rotate_y);


    cv::Mat intensity_front_view = generateImage(accumulated_cloud_local, width, height, true);
    sensor_msgs::ImagePtr image_msg = cv_bridge::CvImage(std_msgs::Header(), "32FC1", intensity_front_view).toImageMsg();
    image_msg->header.frame_id = "map";
    image_msg->header.stamp = ros::Time::now(); 
    pubIntensityImage.publish(image_msg);
    cv::Mat depth_front_view = generateImage(accumulated_cloud_local, width, height, false);
    sensor_msgs::ImagePtr depth_image_msg = cv_bridge::CvImage(std_msgs::Header(), "32FC1", depth_front_view).toImageMsg();
    depth_image_msg->header.frame_id = "map";
    depth_image_msg->header.stamp = ros::Time::now();
    pubDepthImage.publish(depth_image_msg);
    auto [output1, output2,output3] = run_model(depth_front_view, intensity_front_view);


    cv::Mat lane_image = run_lane_detect_model(output2);

    cv::Mat lane_image_8u;
    lane_image.convertTo(lane_image_8u, CV_8UC3, 255.0);


    sensor_msgs::ImagePtr lane_image_msg = cv_bridge::CvImage(std_msgs::Header(), "rgb8", lane_image_8u).toImageMsg();
    lane_image_msg->header.frame_id = "map";
    lane_image_msg->header.stamp = ros::Time::now();
    pubLaneImage.publish(lane_image_msg);

    sensor_msgs::ImagePtr output1_image_msg = cv_bridge::CvImage(std_msgs::Header(), "32FC1", output1).toImageMsg();
    output1_image_msg->header.frame_id = "map";
    output1_image_msg->header.stamp = ros::Time::now();
    pubDepthImageDense.publish(output1_image_msg);

    sensor_msgs::ImagePtr output2_image_msg = cv_bridge::CvImage(std_msgs::Header(), "32FC1", output2).toImageMsg();
    output2_image_msg->header.frame_id = "map";
    output2_image_msg->header.stamp = ros::Time::now();
    pubIntensityImageDense.publish(output2_image_msg);

    sensor_msgs::ImagePtr output3_image_msg = cv_bridge::CvImage(std_msgs::Header(), "32FC1", output3).toImageMsg();
    output3_image_msg->header.frame_id = "map";
    output3_image_msg->header.stamp = ros::Time::now();
    pubIntensityComImageDense.publish(output3_image_msg);
    // 左向图像
    pcl::PointCloud<PointType>::Ptr left_cloud = rotatePointCloud(accumulated_cloud_local, -M_PI_2, Eigen::Vector3f::UnitZ());
    cv::Mat intensity_left_view = generateImage(left_cloud, width, height, true);
    sensor_msgs::ImagePtr left_image_msg = cv_bridge::CvImage(std_msgs::Header(), "32FC1", intensity_left_view).toImageMsg();
    left_image_msg->header.frame_id = "map";
    left_image_msg->header.stamp = ros::Time::now();
    pubLeftImage.publish(left_image_msg);

    // 右向图像
    pcl::PointCloud<PointType>::Ptr right_cloud = rotatePointCloud(accumulated_cloud_local, M_PI_2, Eigen::Vector3f::UnitZ());
    cv::Mat intensity_right_view = generateImage(right_cloud, width, height, true);
    sensor_msgs::ImagePtr right_image_msg = cv_bridge::CvImage(std_msgs::Header(), "32FC1", intensity_right_view).toImageMsg();
    right_image_msg->header.frame_id = "map";
    right_image_msg->header.stamp = ros::Time::now();
    pubRightImage.publish(right_image_msg);

    // 后向图像 
    pcl::PointCloud<PointType>::Ptr back_cloud = rotatePointCloud(accumulated_cloud_local, M_PI, Eigen::Vector3f::UnitZ());
    cv::Mat intensity_back_view = generateImage(back_cloud, width, height, true);
    sensor_msgs::ImagePtr back_image_msg = cv_bridge::CvImage(std_msgs::Header(), "32FC1", intensity_back_view).toImageMsg();
    back_image_msg->header.frame_id = "map";
    back_image_msg->header.stamp = ros::Time::now();
    pubBackImage.publish(back_image_msg);


    auto end_time_ = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time_ - start_time;
    std::cout << "all image generate time: " << duration.count() << " seconds" << std::endl;
    cv::Mat image_camera = cv::imdecode(cv::Mat(image_msg_camera->data), cv::IMREAD_COLOR);

    if (saveImg && frameCount % 5 == 0)
    {
        std::string intensity_save_path_front = imagesavePath + "intensity_front/front_" + std::to_string(frameCount) + ".png";
        cv::imwrite(intensity_save_path_front, normalizeAndConvertTo8UC1(output2));

        std::string camera_save_path = imagesavePath + "camera/camera" + std::to_string(frameCount) + ".png";
        cv::imwrite(camera_save_path, image_camera);

        std::string lane_save_path = imagesavePath + "lane_detect/lane_" + std::to_string(frameCount) + ".png";
        cv::imwrite(lane_save_path, lane_image_8u);

    }

    frameCount++;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "SubMap");
    ros::NodeHandle nh;
    ros::NodeHandle nhPrivate = ros::NodeHandle("~");

    std::string pkg_path = ros::package::getPath("lane_detect");
    nhPrivate.param<std::string>("model_path", model_path, pkg_path + "/model/four_view_afm_s2d_epoch_9.ts");
    nhPrivate.param<std::string>("lane_model_path", lane_model_path, pkg_path + "/model/laneatt/test.ts");
    nhPrivate.param<std::string>("imagesave_path", imagesavePath, pkg_path + "/image_save/");
    nhPrivate.param<std::string>("lidar_topic", lidar_topic, "/livox/lidar");
    nhPrivate.param<std::string>("odom_topic", odom_topic, "/Odometry");
    nhPrivate.param<std::string>("camera_topic", camera_topic, "/camera/color/image_raw/compressed");
    nhPrivate.param("frame_window", frame_window, 5);
    nhPrivate.param("image_process", image_process, true);
    nhPrivate.param("saveImg", saveImg, false);
    nhPrivate.param("width", width, 480);
    nhPrivate.param("height", height, 240);
    nhPrivate.param("img_w", img_w, 640);

    load_model();
    load_lane_detect_model();
    anchor_ys = torch::linspace(1, 0, 72, torch::TensorOptions().dtype(torch::kFloat32));

    message_filters::Subscriber<nav_msgs::Odometry> subOdometry(nh, odom_topic, 1);
    message_filters::Subscriber<livox_ros_driver2::CustomMsg> subLaserCloud(nh, lidar_topic, 1);
    message_filters::Subscriber<sensor_msgs::CompressedImage> subCameraImage(nh, camera_topic, 1);

    typedef message_filters::sync_policies::ApproximateTime<nav_msgs::Odometry, livox_ros_driver2::CustomMsg, sensor_msgs::CompressedImage> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), subOdometry, subLaserCloud, subCameraImage);

    sync.registerCallback(boost::bind(&cloud_handler, _1, _2, _3));

    pubIntensityImage = nh.advertise<sensor_msgs::Image>("front_view", 1);
    pubDepthImage = nh.advertise<sensor_msgs::Image>("depth_view", 1);
    pubDepthImageDense = nh.advertise<sensor_msgs::Image>("dense_depth_view", 1);
    pubIntensityImageDense = nh.advertise<sensor_msgs::Image>("dense_intensity_view", 1);
    pubIntensityComImageDense = nh.advertise<sensor_msgs::Image>("com_dense_intensity_view", 1);
    pubLeftImage = nh.advertise<sensor_msgs::Image>("left_view", 1);
    pubRightImage = nh.advertise<sensor_msgs::Image>("right_view", 1);
    pubBackImage = nh.advertise<sensor_msgs::Image>("back_view", 1);
    pubLaneImage = nh.advertise<sensor_msgs::Image>("lane_detect_view", 1);

    ros::spin();

    return 0;
}


