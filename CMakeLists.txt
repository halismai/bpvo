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
option(WITH_TBB       "with Intel TBB"         OFF)
option(WITH_PROFILER  "With Google Profiler"   OFF)
option(WITH_TCMALLOC  "With Google TCMalloc"   OFF)
option(WITH_SIMD      "Use SIMD instructions"  ON)
option(WITH_BOOST     "Use boost program_options & circular_buffer"  ON)

if(BUILD_STATIC)
  set(LIBRARY_TYPE STATIC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
else()
  set(LIBRARY_TYPE SHARED)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui contrib)
find_package(Eigen REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN_INCLUDE_DIRS})
list(APPEND MY_LIBRARIES ${OpenCV_LIBS})

if(WITH_BOOST)
  set(Boost_USE_STATIC_LIBS     ON)
  set(Boost_USE_STATIC_RUNTIME  ON)
  set(Boost_USE_MULTITHREAD     ON)
  find_package(Boost 1.59.0 REQUIRED COMPONENTS program_options)
  if(Boost_FOUND)
    add_definitions(-DWITH_BOOST)
    include_directories(${Boost_INCLUDE_DIRS})
    list(APPEND MY_LIBRARIES ${Boost_LIBRARIES})
  else()
    message(STATUS "could not find boost")
  endif()
endif()

if(WITH_TBB)
  find_package(TBB)
  if(TBB_FOUND)
    include_directories(${TBB_INCLUDE_DIRS})
    list(APPEND MY_LIBRARIES ${TBB_LIBRARIES})
    add_definitions(-DWITH_TBB)
  endif()
endif()

find_package(GooglePerfTools)
if(WITH_PROFILER)
  if(GOOGLE_PERFTOOLS_FOUND)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${PROFILER_LIBRARIES}")
    add_definitions(-DWITH_PROFILER)
  endif()
endif()

if(WITH_TCMALLOC)
  if(GOOGLE_PERFTOOLS_FOUND)
    list(APPEND MY_LIBRARIES ${TCMALLOC_LIBRARY})
  endif()
endif()

if(WITH_TCMALLOC OR WITH_PROFILER)
  include_directories(${GOOGLE_PERFTOOLS_INCLUDE_DIR})
endif()


if(WITH_SIMD)
  add_definitions(-DWITH_SIMD)
endif()

add_subdirectory(bpvo)
add_subdirectory(test)

