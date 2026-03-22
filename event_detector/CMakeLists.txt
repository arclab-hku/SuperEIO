cmake_minimum_required(VERSION 2.8.3)
project(event_detector)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Add TensorRTBuffer as a subdirectory
add_subdirectory(${PROJECT_SOURCE_DIR}/src/Thirdparty/TensorRTBuffer)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    sensor_msgs
    cv_bridge
    camera_model
    )

find_package(OpenCV REQUIRED)

catkin_package()

include_directories(
    ${catkin_INCLUDE_DIRS}
    )

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
find_package(CUDA REQUIRED)# new line
find_package(yaml-cpp REQUIRED)
find_package(Boost REQUIRED)
find_package(Gflags REQUIRED)
find_package(Glog REQUIRED)# new line
include_directories(
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${GFLAGS_INCLUDE_DIRS} 
    ${GLOG_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src
    /home/cpy/TensorRT-8.4.1.5/include
    /home/cpy/catkin_supereio/devel/include
)

# Link directories for TensorRT
link_directories(
    /home/cpy/TensorRT-8.4.1.5/lib
)

add_executable(event_detector
    src/event_detector_node.cpp
    src/parameters.cpp
    src/event_detector.cpp
    src/TS_generator/TS_generator.cc
    src/utility/visualization.cpp
    src/super_eventpoint.cpp
    src/utils.cc
    )

target_link_libraries(event_detector
    ${catkin_LIBRARIES} 
    ${OpenCV_LIBS} 
    nvinfer
    nvonnxparser
    ${Boost_LIBRARIES}
    ${GFLAGS_LIBRARIES} 
    ${GLOG_LIBRARIES}
    yaml-cpp
    TensorRTBuffer
    ${CUDA_LIBRARIES}
    )


