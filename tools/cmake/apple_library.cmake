# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(${PROJECT_SOURCE_DIR}/tools/cmake/common.cmake)
include(${PROJECT_SOURCE_DIR}/tools/cmake/platform.cmake)
include(${PROJECT_SOURCE_DIR}/tools/cmake/cxx_library.cmake)

pragma_once()

function(apple_library)
  cmake_parse_arguments(
    ARG
    "IOS;MACOS"
    "NAME"
    "SRCS;HEADERS;EXTERNAL_HEADER_DIRS;PREPROCESSOR_FLAGS;DEPS;PLATFORMS;FRAMEWORKS"
    ${ARGN}
  )
  if(NOT ARG_NAME)
    message(FATAL_ERROR "NAME is required")
  endif()

  enforce_target_name_standard(${ARG_NAME})

  if(NOT IOS AND NOT MACOS)
    set(ARG_PLATFORMS ${PLATFORM_OS_MACOSX} ${PLATFORM_OS_IOS})
  else()
    set(ARG_PLATFORMS)
    if(IOS)
      list(APPEND ARG_PLATFORMS ${PLATFORM_OS_IOS})
    endif()
    if(MACOS)
      list(APPEND ARG_PLATFORMS ${PLATFORM_OS_MACOSX})
    endif()
  endif()

  cxx_library(
    NAME ${ARG_NAME}
    SRCS ${ARG_SRCS}
    HEADERS ${ARG_HEADERS}
    EXTERNAL_HEADER_DIRS ${ARG_EXTERNAL_HEADER_DIRS}
    PREPROCESSOR_FLAGS ${ARG_PREPROCESSOR_FLAGS}
    DEPS ${ARG_DEPS}
    PLATFORMS "${ARG_PLATFORMS}"
  )

  foreach(framework ${ARG_FRAMEWORKS})
    target_link_libraries(${ARG_NAME} PUBLIC -framework ${framework})
  endforeach()
endfunction()
