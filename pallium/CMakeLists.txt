# CMake 最低版本号要求
cmake_minimum_required (VERSION 2.8)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# 项目信息
project (testbench)

FIND_PACKAGE( OpenCV REQUIRED )
if(OpenCV_FOUND)
    message("OPENCV FOUND==")
    MESSAGE(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
else()
    message("OPENCV NOT FOUND")
    MESSAGE(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
endif()

FIND_PACKAGE( OpenMP REQUIRED)

if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


# 指定生成目标
add_executable(testbench testbench.cpp opticnerve.cpp celeb.cpp neu.cpp Timer.cpp)
add_executable(tbceleb tbceleb.cpp opticnerve.cpp celeb.cpp)
add_executable(tbceleb8d tbceleb8d.cpp opticnerve.cpp celeb.cpp)
add_executable(testthread testthread.cpp thread_pool.cpp Timer.cpp)

TARGET_LINK_LIBRARIES(testbench ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(tbceleb ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(tbceleb8d ${OpenCV_LIBS})