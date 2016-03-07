file(GLOB src vo_mex.cc)
include_directories(/home/halismai/code/mexmat/)
add_custom_command(
  OUTPUT vo_mex.mexa64
  COMMAND mex
  ARGS -cxx -output vo_mex.mexa64 ${src} CXX_FLAGS='${CMAKE_CXX_FLAGS}'
  DEPENDS bpvo bpvo_utils
  COMMENT "building mex")

add_custom_target(vo_mex ALL DEPENDS vo_mex.mexa64 SOURCES ${src})
