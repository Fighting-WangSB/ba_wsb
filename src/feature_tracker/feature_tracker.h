#pragma once

#include <cstdio>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <queue>
#include <execinfo.h>
#include <csignal>
#include <eigen3/Eigen/Dense>

//#include "camodocal/camera_models/CameraFactory.h"
//#include "camodocal/camera_models/CataCamera.h"
//#include "camodocal/camera_models/PinholeCamera.h"

using namespace std;
using namespace Eigen;

class FeatureTracker
{
public:
    FeatureTracker(int max_features = 300, double quality_level = 0.05, double min_distance = 30.0)
        : MAX_FEATURES(max_features), QUALITY_LEVEL(quality_level), MIN_DISTANCE(min_distance) {};
    void processMeasurements2d2d(cv::Mat &K,cv::Mat &R_mat,cv::Mat &R,cv::Mat &t);
    void processMeasurements3d2d(cv::Mat &K,cv::Mat &R_mat,cv::Mat &R,cv::Mat &t);
    void extractFeaturesimg(const cv::Mat& image,const cv::Mat& depth_image);
    // void extractFeaturesimg2(const cv::Mat& image);


    int row, col;
    cv::Mat prev_img, cur_img;
    vector<cv::Point2f> n_pts;   //特征点输出容器
    vector<cv::Point2f> prev_pts, cur_pts;
    vector<cv::Point2f> prev_un_pts, cur_un_pts;
    vector<cv::Point3f> prev_depth_pts, cur_depth_pts;


private:
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    int MAX_FEATURES;              // 最大特征点数量
    double QUALITY_LEVEL;          // 角点质量水平
    double MIN_DISTANCE;           // 特征点间最小距离



};
