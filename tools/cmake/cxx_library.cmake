# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(${PROJECT_SOURCE_DIR}/tools/cmake/common.cmake)

function(cxx_library)
  cmake_parse_arguments(
    ARG
    ""
    "NAME"
    "SRCS;HEADERS;EXTERNAL_HEADER_DIRS;PREPROCESSOR_DEFS;DEPS"
    ${ARGN}
  )
  if(NOT ARG_NAME)
    message(FATAL_ERROR "NAME is required")
  endif()

  set_target_name(${ARG_NAME} TARGET_NAME)
  set(_visibility_public PUBLIC)
  set(_visibility_private PRIVATE)

  if(ARG_SRCS)
    add_library(${TARGET_NAME} ${ARG_SRCS} ${ARG_HEADERS})
  endif()

  if(ARG_EXTERNAL_HEADER_DIRS)
    if(NOT TARGET ${TARGET_NAME})
      set(_visibility_public INTERFACE)
      set(_visibility_private INTERFACE)
      add_library(${TARGET_NAME} ${_visibility_public})
    endif()

    target_include_directories(${TARGET_NAME} ${_visibility_public} ${ARG_EXTERNAL_HEADER_DIRS})
  elseif(NOT TARGET ${TARGET_NAME})
    message(FATAL_ERROR "SRCS/HEADERS or EXTERNAL_HEADER_DIRS is required")
  endif()

  target_link_libraries(${TARGET_NAME} ${_visibility_public} common_cxx_toolchain)

  if(ARG_PREPROCESSOR_DEFS)
    target_compile_definitions(${TARGET_NAME} ${_visibility_private} ${ARG_PREPROCESSOR_DEFS})
  endif()

  if(ARG_DEPS)
    target_link_libraries(${TARGET_NAME} ${_visibility_private} ${ARG_DEPS})
  endif()
endfunction()
