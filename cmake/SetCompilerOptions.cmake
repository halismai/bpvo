set(EXTRA_FLAGS "")
set(EXTRA_CXX_FLAGS "-std=c++11")
set(EXTRA_C_FLAGS "")
set(EXTRA_FLAGS_DEBUG "")
set(EXTRA_FLAGS_RELEASE "")
set(EXTRA_EXE_LINKER_FLAGS "")
set(EXTRA_EXE_LINKER_FLAGS_DEBUG "")
set(EXTRA_EXE_LINKER_FLAGS_RELEASE "")

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_COMPILER_IS_GNUCXX 1)
  set(CMAKE_COMPILER_IS_CLANGCXX 1)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  message(STATUS "Using Intel compilers")
  set(CMAKE_COMPILER_IS_INTEL 1)
endif()

macro(addExtraCompilerOptions option)
  set(EXTRA_CXX_FLAGS "${EXTRA_CXX_FLAGS} ${option}")
  set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} ${option}")
endmacro()

macro(addExtraLinkerOption option)
  set(EXTRA_EXE_LINKER_FLAGS "${EXTRA_EXE_LINKER_FLAGS} ${option}")
endmacro()

addExtraCompilerOptions(-W)
addExtraCompilerOptions(-Wall)
addExtraCompilerOptions(-Wextra)
addExtraCompilerOptions(-ftree-vectorize)
addExtraCompilerOptions(-funroll-loops)

if(CMAKE_COMPILER_IS_GNUCXX)
  addExtraCompilerOptions(-W)
  addExtraCompilerOptions(-Wall)
  addExtraCompilerOptions(-Werror=sequence-point)
  addExtraCompilerOptions(-Werror=format-security -Wformat)
  addExtraCompilerOptions(-Wundef)
  addExtraCompilerOptions(-Winit-self)
  addExtraCompilerOptions(-Wpointer-arith)
  addExtraCompilerOptions(-Wsign-promo)
  #Warn whenever a pointer is cast such that the required alignment of the
  #target is increased. For example, warn if a char * is cast to an int * on
  #machines where integers can only be accessed at two- or four-byte boundaries.
  addExtraCompilerOptions(-Wcast-align)
  addExtraCompilerOptions(-fdiagnostics-show-option)
  addExtraCompilerOptions(-fdiagnostics-color=auto)
  addExtraCompilerOptions(-pthread)

  if(ENABLE_OMIT_FRAME_POINTER)
    addExtraCompilerOptions(-fomit-frame-pointer)
  else()
    addExtraCompilerOptions(-fno-omit-frame-pointer)
  endif()

  if(ENABLE_FAST_MATH)
    addExtraCompilerOptions(-ffast-math)
  endif()

  if(ENABLE_SSE)
    addExtraCompilerOptions(-msse)
  endif()

  if(ENABLE_SSE2)
    addExtraCompilerOptions(-msse2)
  endif()

  if(ENABLE_SSE3)
    addExtraCompilerOptions(-msse3)
  endif()

  if(ENABLE_SSSE3)
    addExtraCompilerOptions(-mssse3)
  endif()

  if(ENABLE_SSE41)
    addExtraCompilerOptions(-msse4.1)
  endif()

  if(ENABLE_SSE42)
    addExtraCompilerOptions(-msse4.2)
  endif()

  if(ENABLE_AVX)
    addExtraCompilerOptions(-mavx)
  endif()

  if(ENABLE_POPCNT)
    addExtraCompilerOptions(-mpopcnt)
  endif()

  if(EXTRA_CXX_FLAGS MATCHES "-m(sse2|avx)")
    addExtraCompilerOptions(-mfpmath=sse)
  endif()

  if(ENABLE_PROFILING)
    addExtraCompilerOptions("-pg -g")
  endif()

endif()

if(CMAKE_COMPILER_IS_INTEL)
  addExtraCompilerOptions(-qopt-report-phase=vec)
  addExtraCompilerOptions(-opt-report=4)
  addExtraCompilerOptions(-ipo)
  addExtraCompilerOptions(-finline)
  if(ENABLE_AVX)
    addExtraCompilerOptions(-xAVX)
    addExtraCompilerOptions(-axAVX)
  endif()
  add_definitions(-DNOALIAS)
endif()

if(WITH_OPENMP)
  addExtraCompilerOptions(-fopenmp)
  addExtraLinkerOption(-lgomp)
  add_definitions(-DWITH_OPENMP)
endif()


if(BUILD_STATIC AND CMAKE_COMPILER_IS_GNUCXX)
  set(EXTRA_FLAGS "-fPIC ${EXTRA_CXX_FLAGS}")
endif()


set(EXTRA_FLAGS                     "${EXTRA_FLAGS}"                    CACHE INTERNAL "Extra flags")
set(EXTRA_C_FLAGS                   "${EXTRA_C_FLAGS}"                  CACHE INTERNAL "Extra C flags")
set(EXTRA_CXX_FLAGS                 "${EXTRA_CXX_FLAGS}"                CACHE INTERNAL "Extra CXX flags")
set(EXTRA_FLAGS_DEBUG               "${EXTRA_FLAGS_DEBUG}"              CACHE INTERNAL "Extra debug flags")
set(EXTRA_FLAGS_RELEASE             "${EXTRA_FLAGS_RELEASE}"            CACHE INTERNAL "Extra release flags")
set(EXTRA_EXE_LINKER_FLAGS          "${EXTRA_EXE_LINKER_FLAGS}"         CACHE INTERNAL "Extra linker flags")
set(EXTRA_EXE_LINKER_FLAGS_DEBUG    "${EXTRA_EXE_LINKER_FLAGS_DEBUG}"   CACHE INTERNAL "Extra linker flags debug")
set(EXTRA_EXE_LINKER_FLAGS_RELEASE  "${EXTRA_EXE_LINKER_FLAGS_RELEASE}" CACHE INTERNAL "Extra linker flags release")

set(CMAKE_C_FLAGS                  "${CMAKE_C_FLAGS} ${EXTRA_C_FLAGS} ${EXTRA_FLAGS}")
set(CMAKE_CXX_FLAGS                "${CMAKE_CXX_FLAGS} ${EXTRA_CXX_FLAGS} ${EXTRA_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE} ${EXTRA_FLAGS_RELEASE}")
set(CMAKE_C_FLAGS_RELEASE          "${CMAKE_C_FLAGS_RELEASE} ${EXTRA_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG} ${EXTRA_FLAGS_DEBUG}")
set(CMAKE_C_FLAGS_DEBUG            "${CMAKE_C_FLAGS_DEBUG} ${EXTRA_FLAGS_DEBUG}")
set(CMAKE_EXE_LINKER_FLAGS         "${CMAKE_EXE_LINKER_FLAGS} ${EXTRA_EXE_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} ${EXTRA_EXE_LINKER_FLAGS_RELEASE}")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG   "${CMAKE_EXE_LINKER_FLAGS_DEBUG} ${EXTRA_EXE_LINKER_FLAGS_DEBUG}")

