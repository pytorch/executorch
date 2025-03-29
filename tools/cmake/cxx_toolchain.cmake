# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

function(configure_cxx_toolchain)
  cmake_parse_arguments(
    ARG
    ""
    "NAME;CXX_STD"
    ""
    ${ARGN}
  )
  if(NOT ARG_NAME)
    message(FATAL_ERROR "NAME is required")
  elseif(NOT ARG_CXX_STD)
    message(FATAL_ERROR "CXX_STD is required")
  endif()

  add_library(${ARG_NAME} INTERFACE)
  target_compile_options(${ARG_NAME}
    INTERFACE
      -Wall
      -Wpedantic
      -Wextra
      -Werror
  )
  target_compile_features(${ARG_NAME} INTERFACE "cxx_std_${ARG_CXX_STD}")
  target_include_directories(${ARG_NAME} INTERFACE ${PROJECT_SOURCE_DIR})
endfunction()
