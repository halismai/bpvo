file(GLOB src "*.cc")
add_library(bpvo_dataset ${LIBRARY_TYPE} ${src})
set_target_properties(bpvo_dataset PROPERTIES LINKER_LANGUAGE CXX)

