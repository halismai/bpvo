add_library(bpvo_apps ${LIBRARY_TYPE} vo_app.cc)

if(TRUE OR IS_OSX)
  target_link_libraries(bpvo_apps ${MY_LIBRARIES_CORE} bpvo_utils bpvo)
endif()

file(GLOB src "*.cc")
list(REMOVE_ITEM src "${CMAKE_CURRENT_SOURCE_DIR}/vo_app.cc")

foreach(f ${src})
  get_filename_component(bname ${f} NAME_WE)
  add_executable(${bname} ${f})
  target_link_libraries(${bname} bpvo_apps bpvo_utils bpvo ${MY_LIBRARIES})
endforeach()

