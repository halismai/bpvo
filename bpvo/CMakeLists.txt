file(GLOB src "*.cc")
add_library(bpvo ${LIBRARY_TYPE} ${src})
set_target_properties(bpvo PROPERTIES LINKER_LANGUAGE CXX)

if(IS_OSX)
  target_link_libraries(bpvo ${OpenCV_LIBS})
endif()

install(TARGETS bpvo DESTINATION lib)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/bpvo"
  DESTINATION include/bpvo FILES_MATCHING PATTERN "*.h")

