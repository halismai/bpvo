cmake_minimum_required(VERSION 2.8.11 FATAL_ERROR)
project(bpvo CXX)

set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  ${CMAKE_SOURCE_DIR}/cmake
  ${CMAKE_SOURCE_DIR}/cmake/modules)

set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Installation directory")
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Config" FORCE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops -ftree-vectorize")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops -ftree-vectorize")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=auto")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -mtune=native -march=native -mfpmath=sse")
set(CMAKE_EXE_LINKER_FLAGS "-lpthread -lm -lrt -lgomp")


option(BUILD_STATIC   "Build static libraries" OFF)

if(BUILD_STATIC)
  set(LIBRARY_TYPE STATIC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
else()
  set(LIBRARY_TYPE SHARED)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui contrib)
find_package(TBB REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Eigen REQUIRED)

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC_RUNTIME ON)
set(Boost_USE_MULTITHREAD ON)
find_package(Boost 1.57.0 REQUIRED COMPONENTS program_options)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS} ${TBB_INCLUDE_DIRS} ${EIGEN_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
list(APPEND MY_LIBRARIES ${OpenCV_LIBS} ${TBB_LIBRARIES} ${Boost_LIBRARIES})

#pkg_check_modules(TCMALLOC libtcmalloc)
#if(TCMALLOC_FOUND)
#  list(APPEND MY_LIBRARIES ${TCMALLOC_LIBRARIES})
#endif()

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lprofiler")

add_subdirectory(bpvo)
add_subdirectory(test)

