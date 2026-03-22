cmake_minimum_required(VERSION 2.8.3)
project(loop_closure)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

# Add TensorRTBuffer as a subdirectory
add_subdirectory(${PROJECT_SOURCE_DIR}/src/ThirdParty/TensorRTBuffer)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    nav_msgs
    camera_model
    cv_bridge
    roslib
    )

find_package(OpenCV)


find_package(Ceres REQUIRED)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)

include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS}  ${EIGEN3_INCLUDE_DIR})

catkin_package()

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

add_executable(loop_closure
    src/loop_closure_node.cpp
    src/loop_closure.cpp
    src/keyframe.cpp
    src/utility/CameraPoseVisualization.cpp
    src/ThirdParty/DBoW/BowVector.cpp
    src/ThirdParty/DBoW/FBrief.cpp
    src/ThirdParty/DBoW/FeatureVector.cpp
    src/ThirdParty/DBoW/QueryResults.cpp
    src/ThirdParty/DBoW/ScoringObject.cpp
    src/ThirdParty/DUtils/Random.cpp
    src/ThirdParty/DUtils/Timestamp.cpp
    src/ThirdParty/DVision/BRIEF.cpp
    src/ThirdParty/VocabularyBinary.cpp
    src/super_eventpoint.cpp
    src/super_eventmatch.cpp
    src/utils.cc
    src/point_matching.cc
    )

target_link_libraries(loop_closure 
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
        ${CERES_LIBRARIES}) 
# message("catkin_lib  ${catkin_LIBRARIES}")
