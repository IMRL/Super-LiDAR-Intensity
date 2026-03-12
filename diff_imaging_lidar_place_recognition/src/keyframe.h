#pragma once

#include "parameters.h"

using namespace Eigen;
using namespace std;
using namespace DVision;

class KeyFrame
{
public:

    double time_stamp; 
    int index;
    int bag_id;

    cv::Mat image;
    cv::Mat image_intensity;
    cv::Mat thumbnail;
    pcl::PointCloud<PointType>::Ptr cloud;

    vector<cv::Point3f> brief_point_3d;
    vector<cv::Point2f> brief_point_2d_uv;
    vector<cv::Point2f> brief_point_2d_norm;
    vector<cv::KeyPoint> brief_window_keypoints;
    vector<BRIEF::bitset> brief_window_descriptors;

    vector<cv::Point3f> search_brief_point_3d;
    vector<cv::Point2f> search_brief_point_2d_uv;
    vector<cv::Point2f> search_brief_point_2d_norm;
    vector<cv::KeyPoint> search_brief_keypoints;
    vector<BRIEF::bitset> search_brief_descriptors;

    vector<cv::Point3f> orb_point_3d;
    vector<cv::Point2f> orb_point_2d_uv;
    vector<cv::Point2f> orb_point_2d_norm;
    vector<cv::KeyPoint> orb_window_keypoints;
    cv::Mat orb_window_descriptors;

    vector<cv::Point3f> search_orb_point_3d;
    vector<cv::Point2f> search_orb_point_2d_uv;
    vector<cv::Point2f> search_orb_point_2d_norm;
    vector<cv::KeyPoint> search_orb_keypoints;
    cv::Mat search_orb_descriptors;


    float fx = IMAGE_WIDTH / 360.0f;
    float fy = IMAGE_HEIGHT / 59.0f;
    float cx = IMAGE_WIDTH / 2.0;
    float cy = IMAGE_HEIGHT / 2.0;
    vector<cv::Mat> bow_descriptors;

    Pose6D position;

    KeyFrame(double _time_stamp, 
        int _index,
        const cv::Mat &_image_intensity, 
        const pcl::PointCloud<PointType>::Ptr _cloud,
        const Pose6D &_position,
        int _bag_id);

    bool findConnection(KeyFrame* old_kf);
    void computeWindowOrbPoint();
    void computeWindowBriefPoint();
    void computeSearchOrbPoint();
    void computeSearchBriefPoint();
    void computeBoWPoint();

    int HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b);

    bool searchInAera(const BRIEF::bitset window_descriptor,
                      const std::vector<BRIEF::bitset> &descriptors_old,
                      const std::vector<cv::Point2f> &keypoints_old,
                      const std::vector<cv::Point2f> &keypoints_old_norm,
                      cv::Point2f &best_match,
                      cv::Point2f &best_match_norm);

    void searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
                          std::vector<cv::Point2f> &matched_2d_old_norm,
                          std::vector<uchar> &status,
                          const std::vector<BRIEF::bitset> &descriptors_now,
                          const std::vector<BRIEF::bitset> &descriptors_old,
                          const std::vector<cv::Point2f> &keypoints_old,
                          const std::vector<cv::Point2f> &keypoints_old_norm);


    void PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
                   const std::vector<cv::Point3f> &matched_3d,
                   std::vector<uchar> &status);

    void estimatePose2D2D(const std::vector<cv::Point2f> &matched_2d_cur,
                            const std::vector<cv::Point2f> &matched_2d_old,
                            cv::Mat &R, cv::Mat &t);

    void extractPoints(const vector<cv::Point2f>& in_point_2d_uv, 
                        vector<cv::Point3f>& out_point_3d,
                        vector<cv::Point2f>& out_point_2d_norm,
                        vector<uchar>& out_status);

    void extractNormalizedPoints(const vector<cv::Point2f>& in_point_2d_uv, 
                                       vector<cv::Point2f>& out_point_2d_norm,
                                       vector<uchar>& out_status);
    bool distributionValidation(const vector<cv::Point2f>& new_point_2d_uv, 
                          const vector<cv::Point2f>& old_point_2d_uv);

    void freeMemory();
};

