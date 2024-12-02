#include "feature_tracker.h"
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

//2d-2d
void FeatureTracker::processMeasurements2d2d(cv::Mat &K,cv::Mat &R_mat,cv::Mat &R,cv::Mat &t) {

    // 1. 计算本质矩阵
    cv::Mat E = cv::findEssentialMat(prev_pts, cur_pts, K ,CV_RANSAC);

    // 2. 从本质矩阵恢复相机之间的位姿
    cv::recoverPose(E, prev_pts, cur_pts, K, R, t, CV_RANSAC);

    // 3输出旋转矩阵和位移向量
    std::cout << "Rotation Matrix (R): " << std::endl << R << std::endl;
    std::cout << "Translation Vector (t): " << std::endl << t << std::endl;

    // 1. 平移向量
    double tx = t.at<double>(0);
    double ty = t.at<double>(1);
    double tz = t.at<double>(2);

    // 2. 将旋转矩阵转换为四元数
    cv::Rodrigues(R, R_mat);  // 将旋转矩阵 R 转换为旋转向量 rvec

    // 确保 rvec 是一个 3x1 矩阵
    CV_Assert(R_mat.rows == 3 && R_mat.cols == 1);

    double angle = cv::norm(R_mat);  // 旋转角度
    cv::Vec3d axis(R_mat.at<double>(0) / angle, R_mat.at<double>(1) / angle, R_mat.at<double>(2) / angle);  // 归一化轴

    double qw = cos(angle / 2.0);
    double qx = axis[0] * sin(angle / 2.0);
    double qy = axis[1] * sin(angle / 2.0);
    double qz = axis[2] * sin(angle / 2.0);

    // 3. 输出结果
    std::cout << "Translation Vector: (" << tx << ", " << ty << ", " << tz << ")" << std::endl;
    std::cout << "Quaternion: (" << qx << ", " << qy << ", " << qz << ", " << qw << ")" << std::endl;
}

//3d-2d pnp
void FeatureTracker::processMeasurements3d2d(cv::Mat &K,cv::Mat &R_mat,cv::Mat &R,cv::Mat &t) {

    // std::cout<<"2dcount:" << cur_pts.size()<<std::endl;
    // std::cout<<"3dcount:" << prev_depth_pts.size()<<std::endl;

    cv::solvePnP(prev_depth_pts,cur_pts,K,cv::Mat(),R_mat,t,false,cv::SOLVEPNP_EPNP);

    cv::Rodrigues(R_mat,R);  // 将旋转向量转换为旋转矩阵

    // 输出旋转和平移
    std::cout << "Rotation Matrix (R): " << std::endl << R << std::endl;
    std::cout << "Translation Vector (t): " << std::endl << t << std::endl;
}

//可视化特征匹配
void visualizeFeatureMatches(const cv::Mat& prev_img, const cv::Mat& cur_img,
                             const std::vector<cv::Point2f>& prev_pts,
                             const std::vector<cv::Point2f>& cur_pts) {
    // 检查输入的特征点是否匹配
    if (prev_pts.size() != cur_pts.size() || prev_pts.empty()) {
        std::cerr << "Feature points are not valid or do not match in size!" << std::endl;
        return;
    }

    cv::Mat prev_img_color, cur_img_color;
    cv::cvtColor(prev_img, prev_img_color, cv::COLOR_GRAY2BGR);
    cv::cvtColor(cur_img, cur_img_color, cv::COLOR_GRAY2BGR);

    // 创建一张新的图像，显示左右两帧图像拼接在一起
    cv::Mat output_img;
    cv::hconcat(prev_img_color, cur_img_color, output_img);  // 将前一帧和当前帧水平拼接

    // 绘制特征点对应的线
    for (size_t i = 0; i < prev_pts.size(); ++i) {
        // 计算每个点的绘制位置 (在拼接的图像中，当前图像的点需要加上一个偏移量)
        cv::Point2f prev_pt = prev_pts[i];
        cv::Point2f cur_pt = cur_pts[i] + cv::Point2f(static_cast<float>(prev_img.cols), 0); // 偏移当前帧点的位置

        // 绘制连接线（细线）
        cv::line(output_img, prev_pt, cur_pt, cv::Scalar(0, 255, 0), 1, 8, 0);  // 绿色细线连接

        // 可选：可以在特征点上绘制小圆圈
        cv::circle(output_img, prev_pt, 3, cv::Scalar(0, 0, 255), -1); // 前一帧特征点
        cv::circle(output_img, cur_pt, 3, cv::Scalar(255, 0, 0), -1);  // 当前帧特征点
    }

    // 显示结果
    cv::imshow("Feature Matches", output_img);
    cv::waitKey(0);  // 按键退出
}

// Shi-Tomasi特征提取
void FeatureTracker::extractFeaturesimg(const cv::Mat& image,const cv::Mat& depth_image) {

    if (prev_img.empty()) {
        prev_img = image.clone();
        // 使用 Shi-Tomasi 方法提取特征点作为初始 prev_pts
        cv::goodFeaturesToTrack(prev_img, prev_pts, MAX_FEATURES, QUALITY_LEVEL, MIN_DISTANCE);

        for (const auto& pt : prev_pts) {
            ushort d = depth_image.ptr<ushort>(static_cast<int>(pt.y))[static_cast<int>(pt.x)];

            float depth = d / 5000.0;
            // cout << "depth:" << depth << endl;
            float X = (pt.x - K.at<double>(0, 2)) * depth / K.at<double>(0, 0);
            float Y = (pt.y - K.at<double>(1, 2)) * depth / K.at<double>(1, 1);
            float Z = depth;
            prev_depth_pts.emplace_back(X, Y, Z);
            // cout << X << "," << Y << "," << Z << std::endl;
        }

        // cout << "count of 3d point:" << prev_depth_pts.size() << endl;
        // cout << "count of 2d point:" << prev_pts.size() << endl;

        return;  // 跳过光流计算，等待下一帧
    }

    // 清空之前的特征点
    cur_pts.clear();

    cur_img = image.clone();

    // 初始化当前帧特征点
    std::vector<uchar> status;
    std::vector<float> err;

    // 使用PyrLK方法跟踪特征点
    cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err);

    // 删除跟踪失败的特征点，并且确保 prev_pts 和 cur_pts 是一一对应的
    std::vector<cv::Point2f> prev_pts_valid;
    std::vector<cv::Point2f> cur_pts_valid;
    std::vector<cv::Point3f> prev3d_pts_valid;

    for (size_t i = 0; i < prev_pts.size(); ++i) {
        if (status[i] && prev_depth_pts[i].z) {  // 如果光流跟踪成功
            prev_pts_valid.push_back(prev_pts[i]);  // 将成功的 prev_pts 加入
            cur_pts_valid.push_back(cur_pts[i]);    // 将对应的 cur_pts 加入
            prev3d_pts_valid.push_back(prev_depth_pts[i]);  // 将对应的 prev_depth_pts 加入
        }
    }

    // 更新 prev_pts 和 cur_pts,prev_depth_pts
    prev_pts = prev_pts_valid;
    cur_pts = cur_pts_valid;
    prev_depth_pts = prev3d_pts_valid;

    //特征点的数量
    std::cout << cur_pts.size() << std::endl;

    prev_pts_valid.clear();
    cur_pts_valid.clear();
    prev3d_pts_valid.clear();
    status.clear();
    err.clear();

    cv::calcOpticalFlowPyrLK(cur_img,prev_img, cur_pts,prev_pts,  status, err);

    for (size_t i = 0; i < cur_pts.size(); ++i) {
        if (status[i] && prev_depth_pts[i].z) {  // 如果光流跟踪成功
            prev_pts_valid.push_back(prev_pts[i]);  // 将成功的 prev_pts 加入
            cur_pts_valid.push_back(cur_pts[i]);    // 将对应的 cur_pts 加入
            prev3d_pts_valid.push_back(prev_depth_pts[i]);  // 将对应的 prev_depth_pts 加入
        }
    }

    // 更新 prev_pts 和 cur_pts,prev_depth_pts
    prev_pts = prev_pts_valid;
    cur_pts = cur_pts_valid;
    prev_depth_pts = prev3d_pts_valid;

    //特征点的数量
    std::cout << cur_pts.size() << std::endl;

    // 可视化特征点匹配
    visualizeFeatureMatches(prev_img, cur_img, prev_pts, cur_pts);
}


