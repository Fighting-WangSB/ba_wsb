ceres solver 2.0.0
eigen 3
opencv 3.2.0
ros melodic

mkdir -p catkin_ws/src
cd catkin_ws
catkin_make
cd src
git clone  ..
cd ..
catkin_make
source devel/setup.bash
rosrun 
