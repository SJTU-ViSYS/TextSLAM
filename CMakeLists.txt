cmake_minimum_required(VERSION 2.8)

project(TextSLAM)
set(CMAKE_CXX_STANDARD 11)

# ceres
find_package(Ceres REQUIRED)
# opencv
#set(OpenCV_DIR /opt/ros/kinetic/share/OpenCV-3.3.1-dev)
find_package(OpenCV REQUIRED)
# Sophus
find_package(Sophus REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/ceres
    ${CERES_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS})


add_executable(${PROJECT_NAME} main.cpp
    src/ORBextractor.cc
    src/frame.cc
    src/initializer.cc
    src/keyframe.cc
    src/map.cc
    src/mapPts.cc
    src/mapText.cc
    src/optimizer.cc
    src/setting.cc
    src/system.cc
    src/tool.cc
    src/tracking.cc
    src/loopClosing.cc
    src/Sim3Solver.cc
    src/Random.cpp
    src/Timestamp.cpp)

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${CERES_LIBRARIES} ${OpenCV_LIBS} ${QT_LIBRARIES} ${Sophus_LIBRARIES}
    ${EIGEN3_LIBS})
