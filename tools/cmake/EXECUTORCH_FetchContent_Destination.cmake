# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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

# Declare the content to be fetched in a specific location.
macro(EXECUTORCH_FetchContent_Destination DIR)
  _refresh_cache_if_necessary(${DIR})
  set(FETCHCONTENT_BASE_DIR ${DIR})
endmacro()
