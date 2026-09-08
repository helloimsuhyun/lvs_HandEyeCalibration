cmake_minimum_required(VERSION 3.18)
project(pclpybridge LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(PCL REQUIRED COMPONENTS
    common
    filters
    features
    keypoints
    registration
    search
    kdtree
)

pybind11_add_module(_core src/bindings.cpp)

target_include_directories(_core SYSTEM PRIVATE ${PCL_INCLUDE_DIRS})
target_link_directories(_core PRIVATE ${PCL_LIBRARY_DIRS})
target_link_libraries(_core PRIVATE ${PCL_LIBRARIES})

# PCL_DEFINITIONS may contain compiler switches (e.g. -D...).
if(PCL_DEFINITIONS)
    target_compile_options(_core PRIVATE ${PCL_DEFINITIONS})
endif()

install(TARGETS _core
    LIBRARY DESTINATION pclpybridge
    RUNTIME DESTINATION pclpybridge
    ARCHIVE DESTINATION pclpybridge
)
