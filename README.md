ceres solver 2.0.0 \
eigen 3 \
opencv 3.2.0 \
ros melodic \
 \
mkdir -p catkin_ws/src \
cd catkin_ws \
catkin_make \
cd src \
git clone  .. \
修改main.cpp中图像的路径 \
cd .. \
catkin_make \
source devel/setup.bash \
rosrun feature_matching feature_matching
