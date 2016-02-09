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

option(BUILD_STATIC   "Build static libraries" OFF)
option(WITH_TBB       "with Intel TBB"         OFF)
option(WITH_PROFILER  "With Google Profiler"   OFF)
option(WITH_TCMALLOC  "With Google TCMalloc"   OFF)
option(WITH_SIMD      "Use SIMD instructions"  ON)
option(WITH_BOOST     "Use boost program_options & circular_buffer"  ON)
option(WITH_OPENMP    "use OpenMP"  ON)

option(ENABLE_PROFILING "enable profiling" OFF)
option(ENABLE_FAST_MATH "enable fast math" OFF)
option(ENABLE_OMIT_FRAME_POINTER "omit frame pointer" OFF)

option(BUILD_TEST "build the testing stuff" ON)

# auto decide release configuration if not set
if(CMAKE_GENERATOR MATCHES "Makefiles|Ninja" AND "${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

# Set the compiler options
include(cmake/SetCompilerOptions.cmake)

if(BUILD_STATIC)
  # -fPIC is set from SetCompilerOptions
  set(LIBRARY_TYPE STATIC)
else()
  set(LIBRARY_TYPE SHARED)
endif()

# REQUIRED packages
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui contrib)
find_package(Eigen  REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN_INCLUDE_DIRS})
list(APPEND MY_LIBRARIES ${OpenCV_LIBS})

set(Boost_USE_STATIC_LIBS     ON)
set(Boost_USE_STATIC_RUNTIME  ON)
set(Boost_USE_MULTITHREAD     ON)
find_package(Boost 1.59.0 REQUIRED COMPONENTS program_options)
add_definitions(-DWITH_BOOST)
include_directories(${Boost_INCLUDE_DIRS})
list(APPEND MY_LIBRARIES ${Boost_LIBRARIES})

# optional stuff
find_package(TBB)
find_package(GooglePerfTools)

if(TBB_FOUND)
  include_directories(${TBB_INCLUDE_DIRS})
  list(APPEND MY_LIBRARIES ${TBB_LIBRARIES})
  if(WITH_TBB)
    add_definitions(-DWITH_TBB)
  endif()
endif()

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

if(BUILD_TEST)
  add_subdirectory(test)
endif()

