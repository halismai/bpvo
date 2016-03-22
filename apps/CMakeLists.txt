add_library(bpvo_apps ${LIBRARY_TYPE} vo_app.cc)

file(GLOB src "*.cc")
list(REMOVE_ITEM src vo_app.cc)

foreach(f ${src})
  get_filename_component(bname ${f} NAME_WE)
  add_executable(${bname} ${f})
  target_link_libraries(${bname} bpvo_utils bpvo_apps bpvo ${MY_LIBRARIES})
endforeach()
