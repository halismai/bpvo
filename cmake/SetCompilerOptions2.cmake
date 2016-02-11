include(cmake/VcMacros.cmake)
include(cmake/OptimizeForArchitecture.cmake)

vc_determine_compiler()

option(USE_CCACHE "If enabled, ccache will be used (if it exists on the system) to speed up recompiles." OFF)
   if(USE_CCACHE)
      find_program(CCACHE_COMMAND ccache)
      if(CCACHE_COMMAND)
         mark_as_advanced(CCACHE_COMMAND)
         set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_COMMAND}")
      endif()
   endif()

   # TODO: check that 'decltype' compiles
   # TODO: check that 'constexpr' compiles
   if(NOT Vc_COMPILER_IS_MSVC) # MSVC doesn't provide a switch to turn C++11 on/off AFAIK
      AddCompilerFlag("-std=c++14" CXX_RESULT _ok MIC_CXX_RESULT _mic_ok CXX_FLAGS CMAKE_CXX_FLAGS MIC_CXX_FLAGS Vc_MIC_CXX_FLAGS)
      if(MIC_NATIVE_FOUND AND NOT _mic_ok)
         AddCompilerFlag("-std=c++1y" MIC_CXX_RESULT _mic_ok MIC_CXX_FLAGS Vc_MIC_CXX_FLAGS)
         if(NOT _mic_ok)
            AddCompilerFlag("-std=c++11" MIC_CXX_RESULT _mic_ok MIC_CXX_FLAGS Vc_MIC_CXX_FLAGS)
            if(NOT _mic_ok)
               AddCompilerFlag("-std=c++0x" MIC_CXX_RESULT _mic_ok MIC_CXX_FLAGS Vc_MIC_CXX_FLAGS)
               if(NOT _mic_ok)
                  message(FATAL_ERROR "Vc 1.x requires C++11, better even C++14. The MIC native compiler does not support any of the C++11 language flags.")
               endif()
            endif()
         endif()
      endif()
      if(NOT _ok)
         AddCompilerFlag("-std=c++1y" CXX_RESULT _ok CXX_FLAGS CMAKE_CXX_FLAGS)
         if(NOT _ok)
            AddCompilerFlag("-std=c++11" CXX_RESULT _ok CXX_FLAGS CMAKE_CXX_FLAGS)
            if(NOT _ok)
               AddCompilerFlag("-std=c++0x" CXX_RESULT _ok CXX_FLAGS CMAKE_CXX_FLAGS)
               if(NOT _ok)
                  message(FATAL_ERROR "Vc 1.x requires C++11, better even C++14. It seems this is not available. If this was incorrectly determined please notify vc-devel@compeng.uni-frankfurt.de")
               endif()
            endif()
         endif()
      endif()
   elseif(Vc_MSVC_VERSION LESS 180021114)
      message(FATAL_ERROR "Vc 1.x requires C++11 support. This requires at least Visual Studio 2013 with the Nov 2013 CTP.")
   endif()

   if(Vc_COMPILER_IS_GCC)
      if(Vc_GCC_VERSION VERSION_GREATER "5.0.0" OR Vc_GCC_VERSION VERSION_LESS "5.3.0")
         UserWarning("GCC 5 goes into an endless loop comiling example_scaling_scalar. Therefore, this target is disabled.")
         list(APPEND disabled_targets
            example_scaling_scalar
            )
      endif()
   elseif(Vc_COMPILER_IS_MSVC)
      if(MSVC_VERSION LESS 1700)
         # MSVC before 2012 has a broken std::vector::resize implementation. STL + Vc code will probably not compile.
         # UserWarning in VcMacros.cmake
         list(APPEND disabled_targets
            stlcontainer_sse
            stlcontainer_avx
            )
      endif()
      # Disable warning "C++ exception specification ignored except to indicate a function is not __declspec(nothrow)"
      # MSVC emits the warning for the _UnitTest_Compare desctructor which needs the throw declaration so that it doesn't std::terminate
      AddCompilerFlag("/wd4290")
   endif()
   if(MIC_NATIVE_FOUND)
      if("${Vc_MIC_ICC_VERSION}" VERSION_LESS "16.1.0")
         UserWarning("ICC for MIC uses an incompatible STL. Disabling simdize_mic.")
         list(APPEND disabled_targets
            simdize_mic
            example_simdize_mic
            )
      endif()
   endif()
endif()

if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebug RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

vc_set_preferred_compiler_flags(WARNING_FLAGS BUILDTYPE_FLAGS)

add_definitions(${Vc_DEFINITIONS})
add_compile_options(${Vc_COMPILE_FLAGS})

if(Vc_COMPILER_IS_INTEL)
   # per default icc is not IEEE compliant, but we need that for verification
   AddCompilerFlag("-fp-model source")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "" AND NOT CMAKE_CXX_FLAGS MATCHES "-O[123]")
   message(STATUS "WARNING! It seems you are compiling without optimization. Please set CMAKE_BUILD_TYPE.")
endif(CMAKE_BUILD_TYPE STREQUAL "" AND NOT CMAKE_CXX_FLAGS MATCHES "-O[123]")

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)

