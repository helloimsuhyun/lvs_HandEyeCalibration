cmake_minimum_required(VERSION 3.8)
project(ur5e_laser_visualization)

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf launch rviz meshes
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
