#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <rosbag/recorder.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <thread>
#include <chrono>
#include <sstream>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

bool isMoving = false;
ros::Time last_motion_time;
std::string rosbag_directory = "/tmp"; 
int rosbag_index = 0;

void cmdVelCallback(const geometry_msgs::Twist::ConstPtr& msg) {
    if (msg->linear.x != 0 || msg->linear.y != 0 || msg->linear.z != 0 ||
        msg->angular.x != 0 || msg->angular.y != 0 || msg->angular.z != 0) {
        isMoving = true;
        last_motion_time = ros::Time::now();
    } else {
        isMoving = false;
    }
}

bool fileExists(const std::string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

void recordRosbag() {

    std::this_thread::sleep_for(std::chrono::seconds(3));
    
    if (isMoving) {
        return; 
    }

    std::stringstream ss;
    ss << rosbag_directory << "/rosbag_" << rosbag_index << ".bag";
    std::string bag_name = ss.str();
    rosbag::RecorderOptions options;
    options.prefix = bag_name;
    options.topics.push_back("/topic1");
    options.topics.push_back("/topic2");
    options.max_duration = ros::Duration(60); 
    options.quiet = false;

    rosbag::Recorder recorder(options);
    recorder.run();

    if (fileExists(bag_name + "_0.bag")) {
        rosbag::Bag bag;
        bag.open(bag_name + "_0.bag", rosbag::bagmode::Read);
        rosbag::View view(bag);
        if (view.size() == 0 || view.getEndTime() - view.getBeginTime() < ros::Duration(60)) {
            bag.close();
            remove((bag_name + "_0.bag").c_str()); 
            ROS_WARN_STREAM("Rosbag " << bag_name << "_0.bag is empty or too short, deleting it.");
        } else {
            bag.close();
            rosbag_index++; 
            ROS_INFO_STREAM("Rosbag " << bag_name << "_0.bag recorded successfully.");
        }
    }
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "record_rosbag");
    ros::NodeHandle nh;

    ros::Subscriber sub = nh.subscribe("cmd_vel", 1000, cmdVelCallback);
    last_motion_time = ros::Time::now();

    ros::Rate rate(10); 
    while (ros::ok()) {
        if (!isMoving && (ros::Time::now() - last_motion_time).toSec() >= 3) {
            std::thread(recordRosbag).detach();
        }
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}