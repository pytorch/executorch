#
# Copyright (c) 2023 Apple Inc. All rights reserved. Provided subject to the
# LICENSE file in the top level directory.
#

cmake_minimum_required(VERSION 3.19)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE python3)
endif()

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

set(_common_compile_options -Wno-deprecated-declarations)
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

list(TRANSFORM _mps_backend__srcs PREPEND "${EXECUTORCH_ROOT}/")
add_library(mpsdelegate ${_mps_backend__srcs})
target_link_libraries(mpsdelegate PRIVATE baundled_program
                                          ${_executor_runner_libs})
target_include_directories(mpsdelegate PUBLIC ${_common_include_directories})

install(
  TARGETS mpsdelegate
  DESTINATION lib
  INCLUDES
  DESTINATION ${_common_include_directories})
