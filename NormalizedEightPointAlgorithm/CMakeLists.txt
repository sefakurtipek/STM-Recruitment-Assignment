cmake_minimum_required(VERSION 3.10)

# Project Name
project(NormalizedFundamentalMatrixEstimation)

# C++ Standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include Directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("/usr/include/eigen3") # Adjust this path to where Eigen is located in your system

# Add executable
add_executable(NormalizedFundamentalMatrixEstimation normalizedEightPointAlgorithm.cpp)

# Link libraries
target_link_libraries(NormalizedFundamentalMatrixEstimation ${OpenCV_LIBS})
