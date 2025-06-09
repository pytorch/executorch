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
function(_refresh_cache_if_necessary DIR)
  set(_generator_stamp_file "${DIR}/.executorch_cmake_generator_stamp")
  set(_current_generator "${CMAKE_GENERATOR}")

  if(EXISTS "${_generator_stamp_file}")
    file(READ "${_generator_stamp_file}" _previous_generator)
    string(STRIP "${_previous_generator}" _previous_generator)
    if(NOT _previous_generator STREQUAL _current_generator)
      file(REMOVE_RECURSE ${DIR})
    endif()
  endif()

  file(WRITE "${_generator_stamp_file}" "${_current_generator}")
endfunction()

function(FetchContent_gflags)
  if(NOT ${gflags_SOURCE_DIR} STREQUAL "")
    _refresh_cache_if_necessary(${gflags_SOURCE_DIR})
  endif()

  FetchContent_Declare(
    gflags
    GIT_REPOSITORY https://github.com/gflags/gflags.git
    GIT_TAG v2.2.2
  )
  set(GFLAGS_INTTYPES_FORMAT C99)
  FetchContent_MakeAvailable(gflags)
endfunction()
