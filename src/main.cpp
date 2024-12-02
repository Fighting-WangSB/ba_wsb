#include <iostream>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include "feature_tracker/feature_tracker.h"
#include "estimator/estimator.h"

#define MAX_FRAME_COUNT 2

int main(int argc, char** argv) {

    int imagecount = 0;
 
    Estimator estimator;
    while(imagecount < MAX_FRAME_COUNT) {
        std::ostringstream oss;
        std::string imagename;
        oss << "../photo/" << imagecount+1 << ".png";
        // oss << "/home/wang/wsb_slam/tum_photo/" << imagecount+1 << ".png";
        imagename = oss.str();
        cv::Mat img = cv::imread(imagename, cv::IMREAD_GRAYSCALE);

        std::ostringstream oss1;
        std::string imagename1;
        oss1 << "../photo/" << imagecount+1 << "_depth.png";
        // oss1 << "/home/wang/wsb_slam/tum_photo/" << imagecount+1 << "_depth.png";
        imagename1 = oss1.str();
        cv::Mat img_depth = cv::imread(imagename1, cv::IMREAD_UNCHANGED);

        estimator.inputImage(imagecount,img,img_depth);

        ++imagecount;
    }

    return 0;
}

