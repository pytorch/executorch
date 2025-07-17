# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# A "pseudo" library that helps propagate common compiler options to libraries.
add_library(cc_library_common_compiler_options INTERFACE)
target_compile_options(cc_library_common_compiler_options
  INTERFACE
    -Wno-deprecated-declarations
    -fPIC
    # TODO: enable some day :')
    # -Wall
    # -Wpedantic
    # -Wextra
    # -Werror
)
target_compile_features(cc_library_common_compiler_options
  INTERFACE
    "cxx_std_${CMAKE_CXX_STANDARD}"
)
target_include_directories(cc_library_common_compiler_options
  INTERFACE
    ${PROJECT_SOURCE_DIR}
)

# Define a C++ library.
function(cc_library)
  cmake_parse_arguments(
    ARG
    ""
    "NAME"
    "SRCS;HEADERS;EXTERNAL_HEADER_DIRS;PREPROCESSOR_FLAGS;DEPS"
    ${ARGN}
  )
  if(NOT ARG_NAME)
    message(FATAL_ERROR "NAME is required")
  endif()

  set(_visibility_public PUBLIC)
  set(_visibility_private PRIVATE)

  if(ARG_SRCS)
    add_library(${ARG_NAME} ${ARG_SRCS} ${ARG_HEADERS})
  endif()

  if(ARG_EXTERNAL_HEADER_DIRS)
    if(NOT TARGET ${ARG_NAME})
      set(_visibility_public INTERFACE)
      set(_visibility_private INTERFACE)
      add_library(${ARG_NAME} ${_visibility_public})
    endif()

    target_include_directories(${ARG_NAME} ${_visibility_public} ${ARG_EXTERNAL_HEADER_DIRS})
  elseif(NOT TARGET ${ARG_NAME})
    message(FATAL_ERROR "SRCS/HEADERS and/or EXTERNAL_HEADER_DIRS is required")
  endif()

  target_link_libraries(${ARG_NAME} ${_visibility_public} cc_library_common_compiler_options)

  if(ARG_PREPROCESSOR_FLAGS)
    target_compile_definitions(${ARG_NAME} ${_visibility_private} ${ARG_PREPROCESSOR_FLAGS})
  endif()

  if(ARG_DEPS)
    target_link_libraries(${ARG_NAME} ${_visibility_private} ${ARG_DEPS})
  endif()
endfunction()
