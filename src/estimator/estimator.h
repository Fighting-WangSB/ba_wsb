#pragma once

#include <thread>
#include <mutex>
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include "../feature_tracker/feature_tracker.h"

class Estimator
{
  public:
    Estimator();

    void inputImage(const int imagecount,const cv::Mat &_img1,const cv::Mat& depth_image);

    queue<pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > > > > featureBuf;

    cv::Mat R_mat;
    cv::Mat K;
    cv::Mat R, t;
    FeatureTracker featureTracker;
};

// 定义重投影误差类
class ReprojectionError {
public:
    ReprojectionError(double observed_x, double observed_y, const cv::Point3d& point_3d)
        : observed_x_(observed_x), observed_y_(observed_y), point_3d_(point_3d) {}

    template <typename T>
    bool operator()(const T* const rotation, const T* const translation, T* residuals) const {
        // 1. 将 3D 点转换到相机坐标系
        T p[3];
        p[0] = T(point_3d_.x);
        p[1] = T(point_3d_.y);
        p[2] = T(point_3d_.z);

        T p_rotated[3];
        ceres::AngleAxisRotatePoint(rotation, p, p_rotated);

        // 2. 添加平移
        p_rotated[0] += translation[0];
        p_rotated[1] += translation[1];
        p_rotated[2] += translation[2];

        // 3. 投影到相机坐标系
        // 使用相机内参矩阵进行投影，假设内参矩阵 K 已经被传递到这个类的构造函数中
        T predicted_x = p_rotated[0] / p_rotated[2];  // x' = X / Z
        T predicted_y = p_rotated[1] / p_rotated[2];  // y' = Y / Z

        // 4. 使用内参矩阵 K 将归一化的图像坐标转换为像素坐标
        // 内参矩阵 K: [fx 0 cx; 0 fy cy; 0 0 1]
        T fx = T(520.9);  // 焦距 x
        T fy = T(521.0);  // 焦距 y
        T cx = T(325.1);  // 主点 x
        T cy = T(249.7);  // 主点 y

        T u = fx * predicted_x + cx;  // 预测的像素坐标 x
        T v = fy * predicted_y + cy;  // 预测的像素坐标 y

        // 5. 计算残差（预测的 2D 点与实际观测的 2D 点之间的差异）
        residuals[0] = u - T(observed_x_);  // x 方向的残差
        residuals[1] = v - T(observed_y_);  // y 方向的残差

        // cout << "xerror" << residuals[0] << endl;
        // cout << "yerror" << residuals[1] << endl;
        return true;
    }

private:
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    double observed_x_;
    double observed_y_;
    cv::Point3d point_3d_;
};

struct AnalyticReprojectionError : public ceres::SizedCostFunction<2, 3, 3> {
    AnalyticReprojectionError(double observed_x, double observed_y, const cv::Point3d& point_3d)
        : observed_x_(observed_x), observed_y_(observed_y), point_3d_(point_3d) {}

    virtual ~AnalyticReprojectionError() {}

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        const double* rotation = parameters[0];  // 旋转向量
        const double* translation = parameters[1];  // 平移向量

        // 1. 将3D点从世界坐标转换到相机坐标系
        double p[3];
        double p_rotated[3] = { point_3d_.x, point_3d_.y, point_3d_.z };
        ceres::AngleAxisRotatePoint(rotation, p_rotated, p);  // 使用旋转向量旋转3D点
        p[0] += translation[0];  // 加上平移
        p[1] += translation[1];
        p[2] += translation[2];

        cv::Mat K_= (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);  // 相机内参矩阵

        // 2. 投影到2D像素平面（使用内参矩阵 K）
        double predicted_x = p[0] / p[2];  // x' = X / Z
        double predicted_y = p[1] / p[2];  // y' = Y / Z
        double u = K_.at<double>(0, 0) * predicted_x +  K_.at<double>(0, 2);  // 预测的像素坐标 x
        double v = K_.at<double>(1, 1) * predicted_y + K_.at<double>(1, 2);  // 预测的像素坐标 y

        // 3. 计算残差
        residuals[0] = u - observed_x_;
        residuals[1] = v - observed_y_;

        // 4. 计算雅可比矩阵
        if (jacobians) {
            // 计算相对于旋转向量的雅可比
            if (jacobians[0]) {
                jacobians[0][0] = -K_.at<double>(0, 0) * p[0] * p[1] / (p[2] * p[2]);
                jacobians[0][1] = K_.at<double>(0, 0) + K_.at<double>(0, 0) * (p[0] * p[0]) / (p[2] * p[2]);
                jacobians[0][2] = -K_.at<double>(0, 0) * p[1] / p[2];

                jacobians[0][3] =  - K_.at<double>(1, 1) - K_.at<double>(1, 1) * (p[1] * p[1]) / (p[2] * p[2]);
                jacobians[0][4] = K_.at<double>(1, 1) * p[0] * p[1] / (p[2] * p[2]);
                jacobians[0][5] = K_.at<double>(1, 1) * p[0] / p[2];

            }

            // 计算相对于平移向量的雅可比
            if (jacobians[1]) {
                jacobians[1][0] = K_.at<double>(0, 0) / p[2];
                jacobians[1][1] = 0.0;
                jacobians[1][2] = -K_.at<double>(0, 0) * p[0] / (p[2] * p[2]);

                jacobians[1][3] = 0.0;
                jacobians[1][4] = K_.at<double>(1, 1) / p[2];
                jacobians[1][5] = -K_.at<double>(1, 1) * p[1] / (p[2] * p[2]);
            }
        }

        return true;
    }

private:
    double observed_x_;
    double observed_y_;
    cv::Point3d point_3d_;

};
