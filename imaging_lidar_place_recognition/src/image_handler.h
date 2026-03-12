#include "parameters.h"
#include "livox_ros_driver2/CustomMsg.h"
#include <torch/script.h>
#include <torch/torch.h>
#include <deque>
#include <utility>
#include <ros/package.h>

class ImageHandler
{
public:

    ros::NodeHandle nh;

    ros::Publisher pub_image;
    ros::Publisher pub_intensity;


    cv::Mat image_range;
    cv::Mat image_intensity;
    cv::Mat image_intensity_com;
    cv::Mat image_intensity_sparse;

    Pose6D position;
    const float VERTICAL_FOV_MIN = -7.0f;
    const float VERTICAL_FOV_MAX = 52.0f;


    int frame_count;
    std::deque<std::pair<livox_ros_driver2::CustomMsg, nav_msgs::Odometry>> cloud_odom_queue; 

    Pose6D odom_first;
    int frame_window = 5;
    bool save_file = false;

    pcl::PointCloud<PointType>::Ptr cloud_track;

    torch::jit::script::Module model;
    std::string model_path;
    std::string single_frame_cloud_path;
    std::string global_accumulated_cloud_path;
    std::string local_accumulated_cloud_path;
    std::string sparse_image_dir;
    std::string dense_image_dir;


    ImageHandler()
    {
        cloud_track.reset(new pcl::PointCloud<PointType>());
        cloud_track->resize(IMAGE_HEIGHT * IMAGE_WIDTH);

        pub_image  = nh.advertise<sensor_msgs::Image>("loop_detector/image_stack", 1);
        pub_intensity = nh.advertise<sensor_msgs::Image>("loop_detector/image_intensity", 1);
        frame_count =0;

        ros::NodeHandle pnh("~");
        pnh.param("frame_window", frame_window, frame_window);
        pnh.param("save_file", save_file, save_file);
        std::string default_model_path = ros::package::getPath("imaging_lidar_place_recognition") + "/scripts/weight/super_panoramic.ts";
        pnh.param("model_path", model_path, default_model_path);
        pnh.param("single_frame_cloud_path", single_frame_cloud_path, std::string(""));
        pnh.param("global_accumulated_cloud_path", global_accumulated_cloud_path, std::string(""));
        pnh.param("local_accumulated_cloud_path", local_accumulated_cloud_path, std::string(""));
        pnh.param("sparse_image_dir", sparse_image_dir, std::string(""));
        pnh.param("dense_image_dir", dense_image_dir, std::string(""));

        load_model();
    }
    
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

    cv::Mat generateIntensityView(const pcl::PointCloud<PointType>::Ptr& cloud, int width, int height, int &points_in_fov) 
    {
        points_in_fov = 0;

        cv::Mat image = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, cv::Scalar(0));

        for (const auto& point : cloud->points) {
            float value = point.intensity;
            float theta = atan2(point.y, point.x) * 180.0 / M_PI;
            float phi = atan2(point.z, sqrt(point.x * point.x + point.y * point.y)) * 180.0 / M_PI;

            int col = (-theta + 180.0f) / 360.0f * width;
            int row = (VERTICAL_FOV_MAX - phi) / (VERTICAL_FOV_MAX - VERTICAL_FOV_MIN) * height;

            if (col >= 0 && col < width && row >= 0 && row < height) {

                uint8_t pixel_value = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, value)));
                image.at<uint8_t>(row, col) = pixel_value;
                points_in_fov++;
            }
        }
        return image;
    }

    cv::Mat generateRangeView(const pcl::PointCloud<PointType>::Ptr& cloud, int width, int height, int &points_in_fov) 
    {
        points_in_fov = 0;
        
        cv::Mat image = cv::Mat(height, width, CV_32FC1, cv::Scalar(0));

        float min_value = std::numeric_limits<float>::max();
        float max_value = std::numeric_limits<float>::lowest();

        for (const auto& point : cloud->points) {
            float value = sqrt(point.x * point.x + point.y * point.y);

            if (value < min_value) min_value = value;
            if (value > max_value) max_value = value;
        }
        for (const auto& point : cloud->points) {
            float value = sqrt(point.x * point.x + point.y * point.y);
            float theta = atan2(point.y, point.x) * 180.0 / M_PI;
            float phi = atan2(point.z, sqrt(point.x * point.x + point.y * point.y)) * 180.0 / M_PI;

            int col = (-theta + 180.0f) / 360.0f * width;
            int row = (VERTICAL_FOV_MAX - phi) / (VERTICAL_FOV_MAX - VERTICAL_FOV_MIN) * height;

            if (col >= 0 && col < width && row >= 0 && row < height) {
                float normalized_value = 255.0f * (value - min_value) / (max_value - min_value);
                image.at<float>(row, col) = normalized_value;

                points_in_fov++;
            }
        }

        cv::Mat image_uint8;
        image.convertTo(image_uint8, CV_8UC1);

        return image_uint8;
    }

    void savePointCloud(const pcl::PointCloud<PointType>::Ptr& cloud, const std::string& folder_path, const std::string& prefix, int frame_idx)
    {
        std::string filename = folder_path + prefix + "_" + std::to_string(frame_idx) + ".pcd";
        pcl::io::savePCDFileBinary(filename, *cloud);
        std::cout << "Saved point cloud: " << filename << std::endl;
    }
    void filterBrightness(cv::Mat& img, cv::Size& window_size) {
        cv::Mat brightness;
        cv::blur(img, brightness, window_size);
        brightness += 1;
        cv::Mat normalized_img = (140.0f * img / brightness);
        cv::GaussianBlur(normalized_img, img, cv::Size(3, 3), 0);
    }

    void cloud_handler(const nav_msgs::Odometry::ConstPtr &odom_msg, const livox_ros_driver2::CustomMsg::ConstPtr &livox_msg)
    {
        cloud_odom_queue.push_back(std::make_pair(*livox_msg, *odom_msg));

        while (cloud_odom_queue.size() > frame_window)
        {
            cloud_odom_queue.pop_front();
        }

        odom_first = getOdom(cloud_odom_queue.front().second);
        position = odom_first;
        std::cout << "Set first frame pose as reference coordinate system." << std::endl;


        pcl::PointCloud<PointType>::Ptr accumulated_cloud_global(new pcl::PointCloud<PointType>);
        for (const auto &pair : cloud_odom_queue)
        {
            const auto &livox_msg = pair.first;
            const auto &odom = pair.second;

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

            Pose6D odom_cur = getOdom(odom);

            pcl::PointCloud<PointType>::Ptr cloud_global = local2global(cloud, odom_cur);

            *accumulated_cloud_global += *cloud_global;
        }

        pcl::PointCloud<PointType>::Ptr accumulated_cloud_local(new pcl::PointCloud<PointType>);
        auto start_time = std::chrono::high_resolution_clock::now();
        *accumulated_cloud_local = *global2local(accumulated_cloud_global, odom_first);

        int points_in_fov_intensity = 0;
        int points_in_fov_range = 0;
        image_range = generateRangeView(accumulated_cloud_local, IMAGE_WIDTH, IMAGE_HEIGHT, points_in_fov_range);
        image_intensity = generateIntensityView(accumulated_cloud_local, IMAGE_WIDTH, IMAGE_HEIGHT, points_in_fov_intensity);

        image_intensity_sparse = image_intensity;
        auto [output1, output2] = run_model(image_range, image_intensity);
        image_intensity = output1;
        image_intensity_com = output2;


        auto end_time_ = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time_ - start_time;
        std::cout << "All image generation time: " << duration.count() << " seconds" << std::endl;

        std::cout << "Points in FOV (intensity view): " << points_in_fov_intensity << std::endl;
        std::cout << "Points in FOV (range view): " << points_in_fov_range << std::endl;

        frame_count++;
        if (save_file && frame_count % 1 == 0)
        {
            if (!global_accumulated_cloud_path.empty())
            {
                // savePointCloud(accumulated_cloud_global, global_accumulated_cloud_path, "global_accumulated", frame_count);
            }
            if (!local_accumulated_cloud_path.empty())
            {
                // savePointCloud(accumulated_cloud_local, local_accumulated_cloud_path, "local_accumulated", frame_count);
            }
            if (!sparse_image_dir.empty())
            {
                std::string sparse_filename = sparse_image_dir + "/intensity_sparse_" + std::to_string(frame_count) + ".png";
                cv::imwrite(sparse_filename, image_intensity_sparse);
            }
            if (!dense_image_dir.empty())
            {
                std::string dense_filename = dense_image_dir + "/intensity_dense_" + std::to_string(frame_count) + ".png";
                cv::imwrite(dense_filename, image_intensity);
            }
            std::cout << "Saved intensity views at frame " << frame_count << std::endl;
        }
        

        if (pub_image.getNumSubscribers() != 0)
        {
            cv::Mat image_visualization;
            cv::vconcat(image_intensity_sparse, image_intensity, image_visualization);
            cv::cvtColor(image_visualization, image_visualization, CV_GRAY2RGB);

            cv::putText(image_visualization, "Sparse", 
                        cv::Point2f(5, 20 + IMAGE_HEIGHT * 0), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 255), 2);
            cv::putText(image_visualization, "Dense", 
                        cv::Point2f(5, 20 + IMAGE_HEIGHT * 1), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 255), 2);

            pubImage(&pub_image, image_visualization, livox_msg->header, "bgr8");
        }
    }


    void pubImage(ros::Publisher *this_pub, const cv::Mat& this_image, std_msgs::Header this_header, string image_format)
    {
        static cv_bridge::CvImage bridge;
        bridge.header = this_header;
        bridge.encoding = image_format;
        bridge.image = this_image;
        this_pub->publish(bridge.toImageMsg());
    }


    torch::Tensor preprocess_images(const cv::Mat& depth_image, const cv::Mat& intensity_image) 
    {

        cv::Mat resized_depth, resized_intensity;
        cv::resize(depth_image, resized_depth, cv::Size(1376, 240));
        cv::resize(intensity_image, resized_intensity, cv::Size(1376, 240));

        cv::Mat float_depth, float_intensity;
        resized_depth.convertTo(float_depth, CV_32F);
        resized_intensity.convertTo(float_intensity, CV_32F);

        double min_val, max_val;
        cv::minMaxLoc(float_depth, &min_val, &max_val);
        float_depth = (float_depth - min_val) / (max_val - min_val);

        cv::minMaxLoc(float_intensity, &min_val, &max_val);
        float_intensity = (float_intensity - min_val) / (max_val - min_val);

        torch::Tensor tensor_depth = torch::from_blob(float_depth.data, {240, 1376}, torch::kFloat32).clone();
        torch::Tensor tensor_intensity = torch::from_blob(float_intensity.data, {240, 1376}, torch::kFloat32).clone();

        torch::Tensor tensor_image = torch::stack({tensor_intensity, tensor_depth}, 0).unsqueeze(0);

        return tensor_image;
    }

    std::pair<cv::Mat, cv::Mat> run_model(const cv::Mat& depth_image, const cv::Mat& intensity_image) 
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

        cv::Mat intensity_out1(cv::Size(1376, 240), CV_32FC1, output1[0][0].data_ptr<float>());
        cv::Mat depth_out1(cv::Size(1376, 240), CV_32FC1, output1[0][1].data_ptr<float>());

        cv::Mat intensity_out2(cv::Size(1376, 240), CV_32FC1, output2[0][0].data_ptr<float>());

        cv::Mat intensity_out1_uint8, depth_out1_uint8, intensity_out2_uint8;
        cv::normalize(intensity_out1, intensity_out1, 0, 255, cv::NORM_MINMAX);
        cv::normalize(depth_out1, depth_out1, 0, 255, cv::NORM_MINMAX);
        cv::normalize(intensity_out2, intensity_out2, 0, 255, cv::NORM_MINMAX);

        intensity_out1.convertTo(intensity_out1_uint8, CV_8UC1);
        depth_out1.convertTo(depth_out1_uint8, CV_8UC1);
        intensity_out2.convertTo(intensity_out2_uint8, CV_8UC1);

        return {intensity_out1_uint8, intensity_out2_uint8};
    }
};
