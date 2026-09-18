cmake_minimum_required(VERSION 3.18)
project(sanchez_wpca LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

find_path(FLANN_INCLUDE_DIR flann/flann.hpp)
find_library(FLANN_CPP_LIBRARY NAMES flann_cpp)
find_library(LZ4_LIBRARY NAMES lz4)

if(NOT FLANN_INCLUDE_DIR OR NOT FLANN_CPP_LIBRARY)
  message(FATAL_ERROR "FLANN not found. On Ubuntu: sudo apt install libflann-dev")
endif()

if(NOT LZ4_LIBRARY)
  message(FATAL_ERROR "LZ4 not found. On Ubuntu: sudo apt install liblz4-dev")
endif()

set(MEPP2_WPCA_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/MEPP2/FEVV/Filters/Generic/PointCloud/WeightedPCANormals"
    CACHE PATH "Path to MEPP2 WeightedPCANormals directory")

if(NOT EXISTS "${MEPP2_WPCA_DIR}/core.h" OR NOT EXISTS "${MEPP2_WPCA_DIR}/core.inl")
  message(FATAL_ERROR
    "MEPP2 WeightedPCANormals source not found at ${MEPP2_WPCA_DIR}. "
    "Run scripts/install.sh or clone MEPP2 into third_party/MEPP2.")
endif()

pybind11_add_module(sanchez_wpca src/sanchez_wpca.cpp)

target_include_directories(sanchez_wpca PRIVATE
  "${MEPP2_WPCA_DIR}"
  "${FLANN_INCLUDE_DIR}"
)

target_link_libraries(sanchez_wpca PRIVATE
  Eigen3::Eigen
  "${FLANN_CPP_LIBRARY}"
  "${LZ4_LIBRARY}"
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(sanchez_wpca PRIVATE -O3 -Wall -Wextra -Wno-unused-parameter)
endif()
