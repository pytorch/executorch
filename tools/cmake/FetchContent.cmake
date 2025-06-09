# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# FetchContent gflags and make available at CMake configure time.
# Make the content available at configure time so that BUCK can use it.
include(FetchContent)

# Some third-party libraries need to be materialized in the source tree for
# buck. Unfortunately, FetchContent bakes in generator information in the
# fetched content. Thus, if the generator is changed, it causes build failures.
# So, if the generator has changed, then nuke the content.
function(_invalidate_cache_if_generator_mismatch DIR)
  set(_generator_stamp_file "${DIR}/.executorch_cmake_generator_stamp")
  set(_current_generator "${CMAKE_GENERATOR}")

  if(EXISTS "${_generator_stamp_file}")
    file(READ "${_generator_stamp_file}" _previous_generator)
    string(STRIP "${_previous_generator}" _previous_generator)
    if(NOT _previous_generator STREQUAL _current_generator)
      file(REMOVE_RECURSE ${DIR})
    endif()
  endif()

endfunction()

# Fetch gflags, and make a symlink to third-party/gflags in the source tree.
# Doing this to satisfy BUCK query for gflags. Also try to invalidate the cmake
# cache if the generator has changed. Notice that symlink won't be created again
# if it's already there.
function(FetchContent_gflags)
  # set(_symlink_target ${CMAKE_CURRENT_LIST_DIR}/../../third-party/gflags)
  # if(IS_DIRECTORY ${_symlink_target})
  #   _invalidate_cache_if_generator_mismatch(${_symlink_target})
  # endif()

  FetchContent_Declare(
    gflags
    GIT_REPOSITORY https://github.com/gflags/gflags.git
    GIT_TAG v2.2.2
  )
  set(GFLAGS_INTTYPES_FORMAT C99)
  FetchContent_MakeAvailable(gflags)

  # if(NOT IS_DIRECTORY ${_symlink_target})
  #   message(STATUS "Creating a symlink from ${gflags_SOURCE_DIR} to third-party/gflags")
  #   execute_process(
  #     COMMAND ${CMAKE_COMMAND} -E create_symlink "${gflags_SOURCE_DIR}" "${_symlink_target}"
  #   )
  #   file(WRITE "${_symlink_target}/.executorch_cmake_generator_stamp" "${CMAKE_GENERATOR}")
  # endif()
endfunction()
