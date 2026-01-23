cmake_minimum_required(VERSION 3.16)
project(habitat_ros)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

install(PROGRAMS app/habitat_node.py
                 app/teleop.py
        DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

ament_python_install_package(${PROJECT_NAME})
ament_package()
