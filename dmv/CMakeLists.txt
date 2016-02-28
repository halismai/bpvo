file(GLOB src "*.cc")
add_library(dmv ${LIBRARY_TYPE} ${src})
set_target_properties(dmv PROPERTIES LINKER_LANGUAGE CXX)

install(TARGETS dmv DESTINATION lib)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/dmv"
  DESTINATION include/bpvo FILES_MATCHING PATTERN "*.h")

