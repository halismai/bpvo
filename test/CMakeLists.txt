add_library(bpvo_test ${LIBRARY_TYPE}
  data_loader.cc
  program_options.cc)

set_target_properties(bpvo_test PROPERTIES LINKER_LANGUAGE CXX)

file(GLOB src "test*.cc")
foreach(f ${src})
  get_filename_component(bname ${f} NAME_WE)
  add_executable(${bname} ${f})
  target_link_libraries(${bname} bpvo_test bpvo ${MY_LIBRARIES})
endforeach()
