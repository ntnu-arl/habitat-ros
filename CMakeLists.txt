cmake_minimum_required(VERSION 3.16)

project(habitat_ros)

find_package(catkin REQUIRED COMPONENTS
    geometry_msgs
    nav_msgs
    rospy
    sensor_msgs
    std_msgs
)

catkin_python_setup()

catkin_package(
    CATKIN_DEPENDS
        geometry_msgs
        message_runtime
        nav_msgs
        rospy
        sensor_msgs
        std_msgs
)