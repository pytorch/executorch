
if(NOT "/Users/chenlai/tmp_executorch/executorch/FXdiv-download/fxdiv-prefix/src/fxdiv-stamp/fxdiv-gitinfo.txt" IS_NEWER_THAN "/Users/chenlai/tmp_executorch/executorch/FXdiv-download/fxdiv-prefix/src/fxdiv-stamp/fxdiv-gitclone-lastrun.txt")
  message(STATUS "Avoiding repeated git clone, stamp file is up to date: '/Users/chenlai/tmp_executorch/executorch/FXdiv-download/fxdiv-prefix/src/fxdiv-stamp/fxdiv-gitclone-lastrun.txt'")
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E rm -rf "/Users/chenlai/tmp_executorch/executorch/FXdiv-source"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/Users/chenlai/tmp_executorch/executorch/FXdiv-source'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/usr/local/bin/git"  clone --no-checkout "https://github.com/Maratyszcza/FXdiv.git" "FXdiv-source"
    WORKING_DIRECTORY "/Users/chenlai/tmp_executorch/executorch"
    RESULT_VARIABLE error_code
    )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once:
          ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/Maratyszcza/FXdiv.git'")
endif()

execute_process(
  COMMAND "/usr/local/bin/git"  checkout master --
  WORKING_DIRECTORY "/Users/chenlai/tmp_executorch/executorch/FXdiv-source"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: 'master'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "/usr/local/bin/git"  submodule update --recursive --init 
    WORKING_DIRECTORY "/Users/chenlai/tmp_executorch/executorch/FXdiv-source"
    RESULT_VARIABLE error_code
    )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/Users/chenlai/tmp_executorch/executorch/FXdiv-source'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy
    "/Users/chenlai/tmp_executorch/executorch/FXdiv-download/fxdiv-prefix/src/fxdiv-stamp/fxdiv-gitinfo.txt"
    "/Users/chenlai/tmp_executorch/executorch/FXdiv-download/fxdiv-prefix/src/fxdiv-stamp/fxdiv-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/Users/chenlai/tmp_executorch/executorch/FXdiv-download/fxdiv-prefix/src/fxdiv-stamp/fxdiv-gitclone-lastrun.txt'")
endif()

