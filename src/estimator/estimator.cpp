#include "estimator.h"

Estimator::Estimator() {

}

void BundleAdjustment(const std::vector<cv::Point3f>& points_3d, const std::vector<cv::Point2f>& points_2d,
                      const cv::Mat& K, cv::Mat& R, cv::Mat& t) {

    // 提取旋转和平移参数
    cv::Mat rotation_vector;  // 旋转向量
    cv::Rodrigues(R, rotation_vector);  // 将旋转矩阵转换为旋转向量

    double rotation[3] = {rotation_vector.at<double>(0), rotation_vector.at<double>(1), rotation_vector.at<double>(2)};
    double translation[3] = {t.at<double>(0), t.at<double>(1), t.at<double>(2)};

    // 创建 Ceres 问题
    ceres::Problem problem;

    // 添加每个3D点到优化问题
    for (size_t i = 0; i < points_3d.size(); ++i) {
        const cv::Point3d& pt_3d = points_3d[i];
        const cv::Point2d& pt_2d = points_2d[i];

        // 为每个点添加残差块，计算雅克比矩阵  (自动求导)
        // ceres::CostFunction* cost_function =
        //     new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 3>(
        //         new ReprojectionError(pt_2d.x, pt_2d.y, pt_3d));

        // 为每个点添加残差块，计算雅克比矩阵  (解析求导)
        ceres::CostFunction* cost_function = new AnalyticReprojectionError(pt_2d.x, pt_2d.y, pt_3d);

        // 定义损失函数，使用 Huber 损失函数，阈值设置为 1.0
        double huber_threshold = 1.0;
        ceres::LossFunction* loss_function = new ceres::HuberLoss(huber_threshold);

        //管理和存储所有的残差块和变量
        problem.AddResidualBlock(cost_function, loss_function, rotation, translation);
    }

    // 配置 Ceres Solver 的选项
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.function_tolerance = 1e-6;                           // 目标函数收敛容忍度
    options.gradient_tolerance = 1e-6;                           // 梯度收敛容忍度
    options.max_num_iterations = 50;  // 设置最大迭代次数

    // 解决优化问题
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 输出优化结果
    std::cout << summary.BriefReport() << std::endl;

    // 将优化结果更新回旋转和平移
    cv::Rodrigues(cv::Mat(3, 1, CV_64F, rotation), R);
    t = (cv::Mat_<double>(3, 1) << translation[0], translation[1], translation[2]);
}

//图像输入
void Estimator::inputImage(const int imagecount,const cv::Mat &_img1,const cv::Mat& depth_image)
{
    featureTracker.extractFeaturesimg(_img1,depth_image);

    K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    if(imagecount) {
        //直接处理数据
        featureTracker.processMeasurements2d2d(K,R_mat,R,t);
        featureTracker.processMeasurements3d2d(K,R_mat,R,t);

        // 进行 BA 优化
        BundleAdjustment(featureTracker.prev_depth_pts, featureTracker.cur_pts, K, R_mat, t);

        std::cout << "Optimized Rotation Matrix: " << std::endl << R << std::endl;
        std::cout << "Optimized Translation Vector: " << std::endl << t << std::endl;

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

        // 更新 prev_img 为当前帧，以便下一次使用
        featureTracker.prev_img = featureTracker.cur_img.clone();
        featureTracker.prev_pts = std::vector<cv::Point2f>(featureTracker.cur_pts);  // 深拷贝
    }
    else {

    }

}

