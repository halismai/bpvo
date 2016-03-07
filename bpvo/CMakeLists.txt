file(GLOB src "*.cc")
add_library(bpvo ${LIBRARY_TYPE} ${src})

if(BUILD_MATLAB)
  add_library(bpvo_matlab STATIC ${src})
endif()

set_target_properties(bpvo PROPERTIES LINKER_LANGUAGE CXX)

install(TARGETS bpvo DESTINATION lib)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/bpvo"
  DESTINATION include/bpvo FILES_MATCHING PATTERN "*.h")

