#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <boost/filesystem.hpp>
#include <Eigen/Dense>
#include <fstream>
#include <cmath>
#include <algorithm>  
#include <limits>    

// Optical Parameters: for eta_R function
const double r_d = 0.001;  
const double d = 0.05;     
const double D = 0.1;      
const double S = 0.1;      

// Compensate Parameters: you need to calibrate the parameters by yourself data
// const double a = -0.22501790662094903;
// const double b = 9.586593531978117;
// const double c = -85.444089785139182;
const double a = -0.08608821299703262;
const double b = 2.163142332692553;
const double c = 0;
const double k = 1.0306961986774714;


double calculateDistance(const pcl::PointXYZINormal& point) {
    double distance = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
    if (!std::isfinite(distance) || distance <= 0) {
        ROS_WARN("Invalid distance: %f", distance);
        return -1.0; 
    }
    return distance;
}

double calculateIncidenceAngle(const pcl::PointXYZINormal& point) {
    Eigen::Vector3d laser_direction(point.x, point.y, point.z);
    laser_direction.normalize();

    Eigen::Vector3d normal_vector(point.normal_x, point.normal_y, point.normal_z);
    normal_vector.normalize();

    if (laser_direction.hasNaN() || normal_vector.hasNaN()) {
        ROS_WARN("NaN detected in laser direction or normal vector.");
        return -1.0; 
    }

    double cos_theta = laser_direction.dot(normal_vector);
    if (std::isnan(cos_theta) || cos_theta < -1.0 || cos_theta > 1.0) {
        ROS_WARN("Invalid cosine of angle: %f", cos_theta);
        return -1.0; 
    }

    double theta = std::acos(cos_theta) * 180.0 / M_PI;

    if (theta > 90.0) {
        theta = 180.0 - theta;
    }

    if (theta > 60)
    {
        theta = 60;
    }

    return theta;
}

double eta_R(double R, double r_d, double d, double D, double S) {
    return 1 - std::exp(-2 * (r_d * r_d) * (R + d) * (R + d) / (D * D * S * S));
}

int applyIntensityCorrection(double intensity, double distance, double incidence_angle) {
    double eta = eta_R(distance, r_d, d, D, S);
    

    double corrected_intensity = intensity;

    double cos_alpha = std::cos(incidence_angle * M_PI / 180.0);  

    corrected_intensity -= (a * distance * distance + b * distance + c);

    int rounded_intensity = static_cast<int>(std::round(corrected_intensity));
    return std::max(0, std::min(255, rounded_intensity));
}
int main(int argc, char** argv) {
    ros::init(argc, argv, "pcd_intensity_correction");

    if (argc < 2) {
        ROS_ERROR("Please provide the input PCD file path.");
        return -1;
    }

    std::string input_pcd_file = argv[1];
    std::string output_pcd_file = "./test_pcd/output/corrected_" + boost::filesystem::path(input_pcd_file).filename().string();

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
    if (pcl::io::loadPCDFile<pcl::PointXYZINormal>(input_pcd_file, *cloud) == -1) {
        ROS_ERROR("Couldn't read file %s", input_pcd_file.c_str());
        return -1
    }

    ROS_INFO("Loaded PCD file with %lu points", cloud->size());

    int min_intensity = std::numeric_limits<int>::max();  
    int max_intensity = std::numeric_limits<int>::min();  

    for (auto& point : cloud->points) {
        double distance = calculateDistance(point);
        if (distance < 0) {
            continue; 
        }

        double incidence_angle = calculateIncidenceAngle(point);
        if (incidence_angle < 0) {
            continue;
        }

        point.intensity = applyIntensityCorrection(point.intensity, distance, incidence_angle);

        min_intensity = std::min(min_intensity, static_cast<int>(point.intensity));
        max_intensity = std::max(max_intensity, static_cast<int>(point.intensity));
    }

    ROS_INFO("Corrected Intensity Range: Min = %d, Max = %d", min_intensity, max_intensity);

    if (pcl::io::savePCDFile(output_pcd_file, *cloud) == -1) {
        ROS_ERROR("Couldn't write output PCD file %s", output_pcd_file.c_str());
        return -1;
    }

    ROS_INFO("Saved corrected PCD file to %s", output_pcd_file.c_str());

    return 0;
}