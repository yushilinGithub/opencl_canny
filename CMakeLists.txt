cmake_minimum_required(VERSION 3.0)
project(imgpro)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/kernel DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/build/)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/image DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/build)

find_package(OpenCL REQUIRED)
find_package(OpenCV REQUIRED)


add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(imgpro main.cpp)


target_include_directories(imgpro PUBLIC ${OpenCV_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS})
target_link_libraries(imgpro imageProcessor ${OpenCV_LIBS} ${OpenCL_LIBRARY})
