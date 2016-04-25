
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
separate_arguments(MEX_FLAGS UNIX_COMMAND "${CMAKE_CXX_FLAGS}")

foreach(d ${dirs})
  list(APPEND MEX_FLAGS "-I${d}")
endforeach()


## Need to set the memxat directory
## TODO integerate this in the build system
list(APPEND MEX_FLAGS "-I/home/halismai/code/mexmat")

list(APPEND MEX_FLAGS "-DMEXMAT_WITH_EIGEN")
list(APPEND MEX_FLAGS "-DMEXMAT_WITH_OPENCV")
list(APPEND MEX_FLAGS "-DMEXMAT_NO_INIT_ARRAY=0")

set(FLAGS -O -largeArrayDims CXXFLAGS='-fPIC ${MEX_FLAGS}')

set(MEX_OUT_DIR ${CMAKE_BINARY_DIR}/bin)
set(MEXEXT mexa64)

set(MEX_COMPILER /usr/local/MATLAB/R2015a/bin/mex)

# will only append the ones needed by the bpvo core
list(APPEND MEX_LIBS -lopencv_core -lopencv_imgproc)

if(WITH_TBB)
  list(APPEND MEX_LIBS ${TBB_LIBRARIES})
endif()

if(WITH_OPENMP)
  list(APPEND MEX_LIBS ${OPENMP})
endif()

file(GLOB src "*_mex.cc")
foreach(f ${src})
  get_filename_component(bname ${f} NAME_WE)

  add_custom_command(
    OUTPUT ${MEX_OUT_DIR}/${bname}.${MEXEXT}
    COMMAND ${MEX_COMPILER}
    ARGS -output ${MEX_OUT_DIR}/${bname}.${MEXEXT} ${FLAGS} ${src}
    -L$<TARGET_FILE_DIR:bpvo> $<TARGET_FILE:bpvo> ${MEX_LIBS}
    DEPENDS bpvo ${f}
    COMMENT "building MEX file ${f}")

  add_custom_target(${bname} ALL DEPENDS ${MEX_OUT_DIR}/${bname}.${MEXEXT} SOURCES ${src})

endforeach()
