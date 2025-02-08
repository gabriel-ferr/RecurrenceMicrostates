cmake_minimum_required(VERSION 3.30)
project(RecurrenceMicrostates)

set(CMAKE_CXX_STANDARD 26)

#   Set the correct directory of Eigen
set(EIGEN3_INCLUDE_DIR "/usr/local/include/eigen3")
include_directories(${EIGEN3_INCLUDE_DIR})

#   Build the executable
add_executable(RecurrenceMicrostates main.cpp
        src/settings.cpp
        src/settings.h
        src/microstates.h)