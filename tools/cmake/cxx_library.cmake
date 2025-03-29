# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(${PROJECT_SOURCE_DIR}/tools/cmake/common.cmake)
include(${PROJECT_SOURCE_DIR}/tools/cmake/platform.cmake)

pragma_once()

function(_determine_toolchain SUPPORTED_PLATFORMS OUTPUT_VAR)
  if(PLATFORM_TARGET_OS STREQUAL PLATFORM_OS_IOS)
    set(${OUTPUT_VAR} ios_toolchain PARENT_SCOPE)
  elseif(PLATFORM_TARGET_OS STREQUAL PLATFORM_OS_MACOSX)
    set(${OUTPUT_VAR} macos_toolchain PARENT_SCOPE)
  else()
    message(FATAL_ERROR "Unsupported target OS: ${PLATFORM_TARGET_OS}")
  endif()

  if(SUPPORTED_PLATFORMS)
    set(_found FALSE)
    foreach(_platform ${SUPPORTED_PLATFORMS})
      if("${_platform}" STREQUAL "${PLATFORM_TARGET_OS}")
        set(_found TRUE)
        break()
      endif()
    endforeach()
    if(NOT _found)
      set(${OUTPUT_VAR} PARENT_SCOPE)
    endif()
  endif()
endfunction()

function(cxx_library)
  cmake_parse_arguments(
    ARG
    ""
    "NAME"
    "SRCS;HEADERS;EXTERNAL_HEADER_DIRS;PREPROCESSOR_FLAGS;DEPS;PLATFORMS"
    ${ARGN}
  )
  if(NOT ARG_NAME)
    message(FATAL_ERROR "NAME is required")
  endif()

  enforce_target_name_standard(${ARG_NAME})

  _determine_toolchain("${ARG_PLATFORMS}" toolchain)
  if(NOT toolchain)
    message(WARNING "Not creating library ${ARG_NAME} due to unsupported platform")
    return()
  endif()

  set(_visibility_public PUBLIC)
  set(_visibility_private PRIVATE)

  if(ARG_SRCS)
    message(STATUS "Adding library ${ARG_NAME}")
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

  target_link_libraries(${ARG_NAME} ${_visibility_public} ${toolchain})

  if(ARG_PREPROCESSOR_FLAGS)
    target_compile_definitions(${ARG_NAME} ${_visibility_private} ${ARG_PREPROCESSOR_FLAGS})
  endif()

  if(ARG_DEPS)
    target_link_libraries(${ARG_NAME} ${_visibility_private} ${ARG_DEPS})
  endif()
endfunction()
