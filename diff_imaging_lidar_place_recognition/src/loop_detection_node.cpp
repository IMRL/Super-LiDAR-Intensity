#include "parameters.h"
#include "keyframe.h"
#include "loop_detection.h"
#include "image_handler.h"
#include "livox_ros_driver2/CustomMsg.h"
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <torch/script.h>
#include <Eigen/Dense>
#include <tf/transform_datatypes.h>
#include <pcl/common/transforms.h>

#include <Eigen/Dense>

std::string PROJECT_NAME;
std::string CLOUD_TOPIC_1;
std::string ODOM_TOPIC_1;
std::string PATH_TOPIC_1;
std::string CLOUD_TOPIC_2;
std::string ODOM_TOPIC_2;
std::string PATH_TOPIC_2;
int IMAGE_WIDTH;
int IMAGE_HEIGHT;
int IMAGE_CROP;
int USE_BRIEF;
int USE_ORB;
int NUM_BRI_FEATURES;
int NUM_ORB_FEATURES;
int MIN_LOOP_FEATURE_NUM;
int MIN_LOOP_SEARCH_GAP;
double MIN_LOOP_SEARCH_TIME;
float MIN_LOOP_BOW_TH;
double SKIP_TIME = 0;
int NUM_THREADS;
int DEBUG_IMAGE;
double MATCH_IMAGE_SCALE;
cv::Mat MASK;
map<int, int> index_match_container;

map<int, int> index_poseindex_container_bag1;
map<int, int> index_poseindex_container_bag2;
pcl::PointCloud<PointType>::Ptr cloud_traj_1(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr cloud_traj_2(new pcl::PointCloud<PointType>());

ros::Publisher pub_match_img;
ros::Publisher pub_match_msg;
ros::Publisher pub_bow_img;
ros::Publisher pub_prepnp_img;
ros::Publisher pub_marker;
ros::Publisher pub_index;
ros::Publisher pub_marker_10;
ros::Publisher pub_index_10;
ros::Publisher re_pub_odometry;
ros::Publisher re_pub_cloud_registered;

ros::Publisher re_pub_path_2;

BriefExtractor briefExtractor;

ImageHandler *image_handler_1;
ImageHandler *image_handler_2;

LoopDetector loopDetector;

float TX,TY,TZ,ROLL,PITCH,YAW;

float radar_tilt_angle = 25.0 * M_PI / 180.0;

Eigen::Affine3f T_2_to_1 = Eigen::Affine3f::Identity();

void initializeTransform()
{

    Eigen::Affine3f rotate_to_radar = Eigen::Affine3f(Eigen::AngleAxisf(radar_tilt_angle, Eigen::Vector3f::UnitY()));

    Eigen::Affine3f user_transform = Eigen::Affine3f(Eigen::Translation3f(TX, TY, TZ)) *
                                     Eigen::Affine3f(Eigen::AngleAxisf(ROLL * M_PI / 180.0, Eigen::Vector3f::UnitX())) *
                                     Eigen::Affine3f(Eigen::AngleAxisf(PITCH * M_PI / 180.0, Eigen::Vector3f::UnitY())) *
                                     Eigen::Affine3f(Eigen::AngleAxisf(YAW * M_PI / 180.0, Eigen::Vector3f::UnitZ()));

    Eigen::Affine3f rotate_back = Eigen::Affine3f(Eigen::AngleAxisf(-radar_tilt_angle, Eigen::Vector3f::UnitY()));
    T_2_to_1 = rotate_back * user_transform * rotate_to_radar;

    std::cout << "tx ty tz " << TX << TY <<TZ <<ROLL <<PITCH <<YAW<< std::endl;
    std::cout << "T_2_to_1 matrix:\n" << T_2_to_1.matrix() << std::endl;
}
void transformOdometryToFrame1(boost::shared_ptr<nav_msgs::Odometry>& odom_ptr)
{
    Eigen::Vector3f position(odom_ptr->pose.pose.position.x,
                             odom_ptr->pose.pose.position.y,
                             odom_ptr->pose.pose.position.z);

    position = T_2_to_1 * position;
    odom_ptr->pose.pose.position.x = position.x();
    odom_ptr->pose.pose.position.y = position.y();
    odom_ptr->pose.pose.position.z = position.z();
    tf::Quaternion q_orig;
    tf::quaternionMsgToTF(odom_ptr->pose.pose.orientation, q_orig);

    tf::Matrix3x3 rotation_matrix(q_orig);
    Eigen::Matrix3f eigen_rotation;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            eigen_rotation(i, j) = rotation_matrix[i][j];

    eigen_rotation = T_2_to_1.rotation() * eigen_rotation;

    Eigen::Quaternionf eigen_quaternion(eigen_rotation);
    tf::Quaternion q_new(eigen_quaternion.x(), eigen_quaternion.y(), eigen_quaternion.z(), eigen_quaternion.w());
    q_new.normalize();
    tf::quaternionTFToMsg(q_new, odom_ptr->pose.pose.orientation);
}

void cloud_registered_handler(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
    pcl::fromROSMsg(*cloud_msg, *cloud);

    pcl::transformPointCloud(*cloud, *cloud, T_2_to_1);
   
    sensor_msgs::PointCloud2 cloud_out;
    pcl::toROSMsg(*cloud, cloud_out);
    cloud_out.header = cloud_msg->header; 
    re_pub_cloud_registered.publish(cloud_out);

    ROS_INFO("Re-published cloud_registered with rotation and translation offset.");
}



void visualizeLoopClosure(ros::Publisher *pub_m, ros::Publisher *pub_i, 
                          ros::Time timestamp, 
                          const pcl::PointCloud<PointType>::Ptr cloud_traj_1, 
                          const pcl::PointCloud<PointType>::Ptr cloud_traj_2)
{
    static visualization_msgs::MarkerArray markerArray; 
    static std_msgs::Int64MultiArray indexArray;        

    std::cout << "!!!!!!!!!!!!visualizing!!!!!!!!!!!!!!!!!!!" << std::endl;
    if (cloud_traj_1->empty() || cloud_traj_2->empty())
        return;

 
    visualization_msgs::Marker markerEdge;
    markerEdge.header.frame_id = "camera_init";
    markerEdge.header.stamp = timestamp;
    markerEdge.action = visualization_msgs::Marker::ADD;
    markerEdge.type = visualization_msgs::Marker::LINE_LIST;
    markerEdge.ns = "loop_edges";
    markerEdge.id = 1; 
    markerEdge.pose.orientation.w = 1;
    markerEdge.scale.x = 0.1;
    markerEdge.color.r = 0.9; 
    markerEdge.color.g = 0.9; 
    markerEdge.color.b = 0.0; 
    markerEdge.color.a = 1.0;

    visualization_msgs::Marker markerNode;
    markerNode.header.frame_id = "camera_init";
    markerNode.header.stamp = timestamp;
    markerNode.action = visualization_msgs::Marker::ADD;
    markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
    markerNode.ns = "loop_nodes";
    markerNode.id = 0; 
    markerNode.pose.orientation.w = 1;
    markerNode.scale.x = 0.3; 
    markerNode.scale.y = 0.3; 
    markerNode.scale.z = 0.3; 
    markerNode.color.r = 0.0; 
    markerNode.color.g = 1.0; 
    markerNode.color.b = 0.0; 
    markerNode.color.a = 1.0;

    const PointType& new_point = cloud_traj_2->points.back();

    geometry_msgs::Point p1, p2;
    p2.x = new_point.x;
    p2.y = new_point.y;
    p2.z = new_point.z;

    double min_distance = std::numeric_limits<double>::max();
    int nearest_index = -1;

    for (size_t i = 0; i < cloud_traj_1->size(); ++i)
    {
        const PointType& candidate_point = cloud_traj_1->points[i];
        double distance = std::sqrt(
            std::pow(candidate_point.x - p2.x, 2) +
            std::pow(candidate_point.y - p2.y, 2) +
            std::pow(candidate_point.z - p2.z, 2)
        );

        if (distance < 3.0 && distance < min_distance)
        {
            min_distance = distance;
            nearest_index = i;
        }
    }

    if (nearest_index != -1)
    {
        const PointType& nearest_point = cloud_traj_1->points[nearest_index];
        p1.x = nearest_point.x;
        p1.y = nearest_point.y;
        p1.z = nearest_point.z;

        markerEdge.points.push_back(p1);
        markerEdge.points.push_back(p2);

        markerNode.points.push_back(p1);
        markerNode.points.push_back(p2);

        indexArray.data.push_back(nearest_index);
        indexArray.data.push_back(cloud_traj_2->size() - 1);
    }


    markerArray.markers.clear();
    markerArray.markers.push_back(markerNode);
    markerArray.markers.push_back(markerEdge);

    pub_m->publish(markerArray);
    pub_i->publish(indexArray);

}

void path_handler_1(const nav_msgs::PathConstPtr& path_msg)
{
    cloud_traj_1->clear();

    for (size_t i = 0; i < path_msg->poses.size(); ++i)
    {
        PointType p;
        p.x = path_msg->poses[i].pose.position.x;
        p.y = path_msg->poses[i].pose.position.y;
        p.z = path_msg->poses[i].pose.position.z;
        cloud_traj_1->push_back(p);
    }
    std::cout << " cloud_traj_1 size is : " << cloud_traj_1->size() << std::endl;
}



void path_handler_2(const nav_msgs::PathConstPtr& path_msg)
{

    cloud_traj_2->clear();

    nav_msgs::Path transformed_path;
    transformed_path.header = path_msg->header; 

    for (size_t i = 0; i < path_msg->poses.size(); ++i)
    {
        Eigen::Vector3f position(path_msg->poses[i].pose.position.x,
                                 path_msg->poses[i].pose.position.y,
                                 path_msg->poses[i].pose.position.z);

        position = T_2_to_1 * position;

        PointType p;
        p.x = position.x();
        p.y = position.y();
        p.z = position.z();
        cloud_traj_2->push_back(p); 

        tf::Quaternion q_orig;
        tf::quaternionMsgToTF(path_msg->poses[i].pose.orientation, q_orig);

        tf::Matrix3x3 rotation_matrix(q_orig);
        Eigen::Matrix3f eigen_rotation;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                eigen_rotation(r, c) = rotation_matrix[r][c];

        
        eigen_rotation = T_2_to_1.rotation() * eigen_rotation;

        Eigen::Quaternionf eigen_quaternion(eigen_rotation);
        tf::Quaternion q_new(eigen_quaternion.x(), eigen_quaternion.y(), eigen_quaternion.z(), eigen_quaternion.w());
        q_new.normalize();

        geometry_msgs::PoseStamped transformed_pose;
        transformed_pose.header = path_msg->poses[i].header; 
        transformed_pose.pose.position.x = position.x();
        transformed_pose.pose.position.y = position.y();
        transformed_pose.pose.position.z = position.z();
        tf::quaternionTFToMsg(q_new, transformed_pose.pose.orientation);

        transformed_path.poses.push_back(transformed_pose);
    }

    std::cout << " cloud_traj_2 size is : " << cloud_traj_2->size() << std::endl;

    re_pub_path_2.publish(transformed_path);
}

void cloud_handler_1(const nav_msgs::Odometry::ConstPtr& odom_msg_1, const livox_ros_driver2::CustomMsg::ConstPtr& cloud_msg_1)
{
    image_handler_1->cloud_handler(odom_msg_1,cloud_msg_1,1);
    std::cout<<"cloud_handle!!!!!!!!!!" <<std::endl;
    double cloud_time = cloud_msg_1->header.stamp.toSec();
    static double last_skip_time = -1;
    if (cloud_time - last_skip_time < SKIP_TIME)
        return;
    else
        last_skip_time = cloud_time;

    static int global_frame_index_1 = 0;
    KeyFrame* keyframe = new KeyFrame(cloud_time,
                                      global_frame_index_1,
                                      image_handler_1->image_intensity,
                                      image_handler_1->cloud_track,
                                      image_handler_1->position,
                                      0);

    loopDetector.addKeyFrame(keyframe, 0);

    index_poseindex_container_bag1[global_frame_index_1] = std::max((int)cloud_traj_1->size() - 1, 0);

    global_frame_index_1++;
    ROS_INFO("Bag1 Keyframe Processed.");
}

void cloud_handler_2(const nav_msgs::Odometry::ConstPtr& odom_msg_2, const livox_ros_driver2::CustomMsg::ConstPtr& cloud_msg_2)
{
    ROS_INFO("\033[1;32m----> Bag2 Keyframe Processed!!!!!!!!!!!.\033[0m");

    boost::shared_ptr<nav_msgs::Odometry> odom_out(new nav_msgs::Odometry(*odom_msg_2));

    transformOdometryToFrame1(odom_out);        
    re_pub_odometry.publish(odom_out);          
    ROS_INFO("Re-published odometry_2 with rotation and translation offset.");

    image_handler_2->cloud_handler(odom_out, cloud_msg_2,2);
    double cloud_time = cloud_msg_2->header.stamp.toSec();
    static double last_skip_time = -1;
    if (cloud_time - last_skip_time < SKIP_TIME)
        return;
    else
        last_skip_time = cloud_time;

    static int global_frame_index_2 = 0;
    KeyFrame* keyframe = new KeyFrame(cloud_time,
                                      global_frame_index_2,
                                      image_handler_2->image_intensity,
                                      image_handler_2->cloud_track,
                                      image_handler_2->position,
                                      1);

    loopDetector.addKeyFrame(keyframe, 1);

    index_poseindex_container_bag2[global_frame_index_2] = std::max((int)cloud_traj_2->size() - 1, 0);

    visualizeLoopClosure(&pub_marker, &pub_index, cloud_msg_2->header.stamp,cloud_traj_1, cloud_traj_2);
    global_frame_index_2++;
    ROS_INFO("Bag2 Keyframe Processed.");
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "loop_detection");
    ros::NodeHandle n;
    
    std::string config_file;
    n.getParam("lio_loop_config_file", config_file);
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    usleep(100);
    
    int LOOP_CLOSURE;
    fsSettings["loop_closure"] >> LOOP_CLOSURE;
    if (!LOOP_CLOSURE)
    {
        ros::spin();
        return 0;
    }

    fsSettings["project_name"] >> PROJECT_NAME;

    fsSettings["cloud_topic_1"]  >> CLOUD_TOPIC_1;
    fsSettings["odom_topic_1"]    >> ODOM_TOPIC_1;

    fsSettings["cloud_topic_2"]  >> CLOUD_TOPIC_2;
    fsSettings["odom_topic_2"]    >> ODOM_TOPIC_2;

    fsSettings["path_topic_1"]   >> PATH_TOPIC_1;
    fsSettings["path_topic_2"]   >> PATH_TOPIC_2;
    fsSettings["image_width"]  >> IMAGE_WIDTH;
    fsSettings["image_height"] >> IMAGE_HEIGHT;
    fsSettings["image_crop"]   >> IMAGE_CROP;
    fsSettings["use_brief"]    >> USE_BRIEF;
    fsSettings["use_orb"]      >> USE_ORB;
    fsSettings["num_bri_features"] >> NUM_BRI_FEATURES;
    fsSettings["num_orb_features"] >> NUM_ORB_FEATURES;
    fsSettings["min_loop_feature_num"] >> MIN_LOOP_FEATURE_NUM;
    fsSettings["min_loop_search_gap"]  >> MIN_LOOP_SEARCH_GAP;
    fsSettings["min_loop_search_time"] >> MIN_LOOP_SEARCH_TIME;
    fsSettings["min_loop_bow_th"]      >> MIN_LOOP_BOW_TH;
    fsSettings["skip_time"]    >> SKIP_TIME;
    fsSettings["num_threads"]  >> NUM_THREADS;
    fsSettings["debug_image"]  >> DEBUG_IMAGE;
    fsSettings["match_image_scale"] >> MATCH_IMAGE_SCALE;

    fsSettings["tx"] >> TX;
    fsSettings["ty"] >> TY;
    fsSettings["tz"] >> TZ;
    fsSettings["roll"] >> ROLL;
    fsSettings["pitch"] >> PITCH;
    fsSettings["yaw"] >> YAW;

    string pkg_path = ros::package::getPath(PROJECT_NAME);

    string vocabulary_file;
    fsSettings["vocabulary_file"] >> vocabulary_file;  
    vocabulary_file = pkg_path + vocabulary_file;
    loopDetector.loadVocabulary(vocabulary_file);

    string brief_pattern_file;
    fsSettings["brief_pattern_file"] >> brief_pattern_file;  
    brief_pattern_file = pkg_path + brief_pattern_file;
    briefExtractor = BriefExtractor(brief_pattern_file);
    initializeTransform();

 

    MASK = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255)); 

    MASK(cv::Range(0, 25), cv::Range::all()) = 0;
    MASK(cv::Range(IMAGE_HEIGHT - IMAGE_CROP, IMAGE_HEIGHT), cv::Range::all()) = 0;



    message_filters::Subscriber<nav_msgs::Odometry> subOdometry_1(n, ODOM_TOPIC_1, 1);
    message_filters::Subscriber<livox_ros_driver2::CustomMsg> subLaserCloud_1(n, CLOUD_TOPIC_1, 1);

    typedef message_filters::sync_policies::ApproximateTime<nav_msgs::Odometry, livox_ros_driver2::CustomMsg> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync_1(MySyncPolicy(10), subOdometry_1, subLaserCloud_1);
    sync_1.registerCallback(boost::bind(&cloud_handler_1, _1, _2));


    message_filters::Subscriber<nav_msgs::Odometry> subOdometry_2(n, ODOM_TOPIC_2, 1);
    message_filters::Subscriber<livox_ros_driver2::CustomMsg> subLaserCloud_2(n, CLOUD_TOPIC_2, 1);

    typedef message_filters::sync_policies::ApproximateTime<nav_msgs::Odometry, livox_ros_driver2::CustomMsg> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync_2(MySyncPolicy(10), subOdometry_2, subLaserCloud_2);
    sync_2.registerCallback(boost::bind(&cloud_handler_2, _1, _2));


    ros::Subscriber sub_path_1  = n.subscribe(PATH_TOPIC_1,  1, path_handler_1);
    ros::Subscriber sub_path_2  = n.subscribe(PATH_TOPIC_2,  1, path_handler_2);
    re_pub_path_2 = n.advertise<nav_msgs::Path>("/bag2/path_trans", 1);
    ros::Subscriber sub_cloud_registered = n.subscribe("/bag2/cloud_registered", 1, cloud_registered_handler);
    re_pub_cloud_registered = n.advertise<sensor_msgs::PointCloud2>("cloud_registered_offset", 1);
    re_pub_odometry = n.advertise<nav_msgs::Odometry>("odometry_offset", 1);

    pub_match_img  = n.advertise<sensor_msgs::Image>                ("loop_detector/image", 1);
    pub_match_msg  = n.advertise<std_msgs::Float64MultiArray>       ("loop_detector/time", 1);
    pub_bow_img    = n.advertise<sensor_msgs::Image>                ("loop_detector/bow", 1);
    pub_prepnp_img = n.advertise<sensor_msgs::Image>                ("loop_detector/prepnp", 1);
    pub_marker     = n.advertise<visualization_msgs::MarkerArray>   ("loop_detector/marker", 1);
    pub_index      = n.advertise<std_msgs::Int64MultiArray>         ("loop_detector/index", 1);
    pub_marker_10     = n.advertise<visualization_msgs::MarkerArray>   ("loop_detector/marker_10", 1);
    pub_index_10      = n.advertise<std_msgs::Int64MultiArray>         ("loop_detector/index_10", 1);

    image_handler_1 = new ImageHandler();
    image_handler_2 = new ImageHandler();

    ROS_INFO("\033[1;32m----> Imaging Lidar Place Recognition Started.\033[0m");

    ros::spin();

    return 0;
}