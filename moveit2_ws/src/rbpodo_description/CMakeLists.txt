cmake_minimum_required(VERSION 3.8)
project(rbpodo_description)
find_package(ament_cmake REQUIRED)
set(RB5_ASSET_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../mujoco_handeye_sim/assets/rb5_850e/rbpodo_description")
install(DIRECTORY "${RB5_ASSET_ROOT}/meshes" "${RB5_ASSET_ROOT}/robots"
  DESTINATION share/${PROJECT_NAME})
ament_package()
