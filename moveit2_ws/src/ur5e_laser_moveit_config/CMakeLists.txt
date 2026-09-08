cmake_minimum_required(VERSION 3.8)
project(ur5e_laser_moveit_config)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch urdf meshes DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/check_sensor_transform.py DESTINATION lib/${PROJECT_NAME})

ament_package()
