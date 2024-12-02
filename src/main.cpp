#include <iostream>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include "feature_tracker/feature_tracker.h"
#include "estimator/estimator.h"

#define MAX_FRAME_COUNT 2

int main(int argc, char** argv) {

    int imagecount = 0;

    // 读取输入图像
    //cv::Mat img1 = cv::imread("/home/wang/wsb_slam/photo/1.png", cv::IMREAD_GRAYSCALE);
    //cv::Mat img2 = cv::imread("/home/wang/wsb_slam/photo/2.png", cv::IMREAD_GRAYSCALE);

    //const std::string config_file = "/home/wang/vins-fusion/src/VINS-Fusion/config/euroc/euroc_mono_imu_config.yaml";
    //readParameters(config_file);

    // if (img1.empty() || img2.empty()) {
    //     std::cerr << "Error opening images!" << std::endl;
    //     return -1;
    // }
 
    Estimator estimator;
    while(imagecount < MAX_FRAME_COUNT) {
        std::ostringstream oss;
        std::string imagename;
        oss << "/home/wang/wsb_slam/photo/" << imagecount+1 << ".png";
        // oss << "/home/wang/wsb_slam/tum_photo/" << imagecount+1 << ".png";
        imagename = oss.str();
        cv::Mat img = cv::imread(imagename, cv::IMREAD_GRAYSCALE);

        std::ostringstream oss1;
        std::string imagename1;
        oss1 << "/home/wang/wsb_slam/photo/" << imagecount+1 << "_depth.png";
        // oss1 << "/home/wang/wsb_slam/tum_photo/" << imagecount+1 << "_depth.png";
        imagename1 = oss1.str();
        cv::Mat img_depth = cv::imread(imagename1, cv::IMREAD_UNCHANGED);

        estimator.inputImage(imagecount,img,img_depth);

        ++imagecount;
    }

    return 0;
}

