# This file is part of bpvo.
#
# bpvo is free software: you can redistribute it and/or modify
# it under the terms of the Lesser GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# bpvo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# Lesser GNU General Public License for more details.

# You should have received a copy of the Lesser GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(bpvo C CXX)

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
option(BUILD_MATLAB   "Build Matlab bindings"  OFF)
option(WITH_PROFILER  "With Google Profiler"   OFF)
option(WITH_TCMALLOC  "With Google TCMalloc"   OFF)
option(WITH_SIMD      "Use SIMD instructions"  ON)
option(WITH_TBB       "with Intel TBB"         OFF)
option(WITH_OPENMP    "use OpenMP"             OFF)
# boost is needed for the utils library
option(WITH_BOOST     "Use boost program_options & circular_buffer"     ON)
option(WITH_GPL_CODE  "Use external code with GPL licenses or similar"  OFF)

option(ENABLE_PROFILING "enable profiling" OFF)
option(ENABLE_FAST_MATH "enable fast math" OFF)
option(ENABLE_OMIT_FRAME_POINTER "omit frame pointer" ON)

option(BUILD_TEST "build the testing stuff" OFF)
option(BUILD_APPS "build apps" ON)


option(WITH_DISPARITY_SPACE_WARP "use disparity space" OFF)

# auto decide release configuration if not set
if(CMAKE_GENERATOR MATCHES "Makefiles|Ninja" AND "${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

include(cmake/MakeVersion.cmake)
MakeVersion(BPVO 1 3 0)

set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Direct Visual Odometry")
set(CPACK_PACKAGE_VENDOR              "halismai")
set(CPACK_PACKAGE_CONTACT             "halismai@cs.cmu.edu")
set(CPACK_PACKAGE_DESCRIPTION_FILE    "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
set(CPACK_RESOURCE_FILE_LICENSE       "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_SOURCE_GENERATOR            "TBZ2")
set(CPACK_SOURCE_PACKAGE_FILE_NAME    "${CMAKE_PROJECT_NAME}-${BPVO_VERSION}")

set(CPACK_SOURCE_IGNORE_FILES
  ${CPACK_SOURCE_IGNORE_FILES}
  "/build/"
  "/.hg/"
  "/.git/"
  # TODO this regex could be wrong
  "/*.swp"
  #"/*.in"
  "${PROJECT_SOURCE_DIR}/.hgignore"
  "${PROJECT_SOURCE_DIR}/.gitignore")

message(STATUS "ignore ${CPACK_SOURCE_IGNORE_FILES}")

include(CPack)
add_custom_target(srcdist COMMAND ${CMAKE_MAKE_PROGRAM} package_source)
message(STATUS "dist file: ${CPACK_SOURCE_PACKAGE_FILE_NAME}")
message(STATUS "version  : ${BPVO_VERSION}")

if("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
  add_definitions(-DIS_64BIT)
endif()

If(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(IS_OSX ON)
  add_definitions(-DIS_OSX)
  message(STATUS "building for OS X")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  message(STATUS "Building for linux")
  set(IS_LINUX ON)
  add_definitions(-DIS_LINUX)
endif()


# add a 'tuning' build type
set(CMAKE_CXX_FLAGS_TUNE "-O3 -g -DNDEBUG"
  CACHE STRING "Flags for code tuning" FORCE)
set(CMAKE_C_FLAGS_TUNE "-O3 -g -DNDEBUG"
  CACHE STRING "Flags for code tuning" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_TUNE ""
  CACHE STRING "Flags for code tuning" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_TUNE ""
  CACHE STRING "Flags for code tuning" FORCE)
mark_as_advanced(CMAKE_CXX_FLAGS_TUNE CMAKE_C_FLAGS_TUNE
  CMAKE_EXE_LINKER_FLAGS_TUNE CMAKE_SHARED_LINKER_FLAGS_TUNE)

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
list(APPEND MY_LIBRARIES_CORE ${OpenCV_LIBS})

set(Boost_USE_STATIC_LIBS     ON)
set(Boost_USE_STATIC_RUNTIME  ON)
set(Boost_USE_MULTITHREAD     ON)
find_package(Boost 1.59.0 REQUIRED COMPONENTS program_options)
add_definitions(-DWITH_BOOST)
include_directories(${Boost_INCLUDE_DIRS})
list(APPEND MY_LIBRARIES ${Boost_LIBRARIES})

if(WITH_DISPARITY_SPACE_WARP)
  add_definitions(-DWITH_DISPARITY_SPACE_WARP)
endif()

# optional stuff
find_package(TBB)
find_package(GooglePerfTools)

if(TBB_FOUND)
  include_directories(${TBB_INCLUDE_DIRS})
  list(APPEND MY_LIBRARIES ${TBB_LIBRARIES})
  list(APPEND MY_LIBRARIES_CORE ${TBB_LIBRARIES})
  if(WITH_TBB)
    add_definitions(-DWITH_TBB)
  endif()
endif()

if(WITH_GPL_CODE)
  add_definitions(-DWITH_GPL_CODE)
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
  if(GOOGLE_PERFTOOLS_FOUND)
    include_directories(${GOOGLE_PERFTOOLS_INCLUDE_DIR})
  endif()
endif()

if(WITH_SIMD)
  add_definitions(-DWITH_SIMD)
endif()


configure_file(
  "${PROJECT_SOURCE_DIR}/bpvo_config.h.in"
  "${PROJECT_SOURCE_DIR}/bpvo/config.h")

add_subdirectory(bpvo)
add_subdirectory(utils)

if(BUILD_MATLAB)
  add_subdirectory(matlab)
endif()

if(BUILD_APPS)
  add_subdirectory(apps)
endif()

if(BUILD_TEST)
  add_subdirectory(test)
endif()

