cmake_minimum_required(VERSION 3.0)
file(GLOB HEADER_FILE *.hpp)
file(GLOB SOURCE_FILE *.cpp)

include_directories(${OpenCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY} ${OpenCV_LIBS})

add_library(imageProcessor STATIC ${HEADER_FILE} ${SOURCE_FILE})
#target_link_libraries(imageProcessor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

