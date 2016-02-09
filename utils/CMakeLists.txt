file(GLOB src "*.cc")
add_library(bpvo_utils ${LIBRARY_TYPE} ${src})
set_target_properties(bpvo_utils PROPERTIES LINKER_LANGUAGE CXX)

