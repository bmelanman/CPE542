cmake_minimum_required(VERSION 3.24)
project(CPE542)

set(CMAKE_CXX_STANDARD 14)

include_directories(./)

find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(CPE542
        main.cpp
        main.hpp
)

target_link_libraries(CPE542 ${OpenCV_LIBS})