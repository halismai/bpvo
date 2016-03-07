file(GLOB src "test*.cc")
foreach(f ${src})
  get_filename_component(bname ${f} NAME_WE)
  add_executable(${bname} ${f})

  if(WITH_DMV)
    target_link_libraries(${bname} dmv bpvo_utils bpvo ${MY_LIBRARIES})
  else()
    target_link_libraries(${bname} bpvo_utils bpvo ${MY_LIBRARIES})
  endif()

endforeach()
