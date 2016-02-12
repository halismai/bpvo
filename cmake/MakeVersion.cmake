function(MakeVersion LibName Major Minor Patch)
  set(${LibName}_VERSION_MAJOR "${Major}" PARENT_SCOPE)
  set(${LibName}_VERSION_MINOR "${Minor}" PARENT_SCOPE)

  # set the patch version with hd id string
  #execute_process(COMMAND git id --id OUTPUT_VARIABLE git_id
  #  OUTPUT_STRIP_TRAILING_WHITESPACE)

  execute_process(COMMAND git describe --tags --always OUTPUT_VARIABLE git_id
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if("${git_id}" STREQUAL "")
    set(git_id "dist")
  else()
    string(STRIP ${git_id} git_id)
    string(REPLACE " " "_" git_id ${git_id})
  endif()

  set(${LibName}_VERSION_PATCH ${Patch}-${git_id}-${CMAKE_BUILD_TYPE} PARENT_SCOPE)

  set(${LibName}_VERSION ${Major}.${Minor}.${Patch} PARENT_SCOPE)

  execute_process(COMMAND date OUTPUT_VARIABLE build_date
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  set(${LibName}_BUILD_DATE ${build_date} PARENT_SCOPE)

  message(STATUS ${build_date})
endfunction(MakeVersion)
