file(GLOB src "*.cc")
add_library(bpvo_utils ${LIBRARY_TYPE} ${src})

if(BUILD_MATLAB)
  add_library(bpvo_utils_matlab STATIC ${src})
endif()

set_target_properties(bpvo_utils PROPERTIES LINKER_LANGUAGE CXX)

