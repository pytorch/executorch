#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

#
# mps_executor_runner: Host tool that demonstrates program
#                      execution using MPSBackend.
#

cmake_minimum_required(VERSION 3.19)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE python3)
endif()

if(NOT FLATC_EXECUTABLE)
  set(FLATC_EXECUTABLE flatc)
endif()

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

include(${EXECUTORCH_ROOT}/build/Utils.cmake)

# ios can only build library but not binary
if(NOT CMAKE_TOOLCHAIN_FILE MATCHES ".*iOS\.cmake$")
  #
  # mps_executor_runner: Like executor_runner but with MPS, the binary will
  # be at ${CMAKE_BINARY_DIR}/examples/apple/executor_runner/mps
  #
set(mps_executor_runner_libs "-framework Foundation"
                              "-weak_framework MetalPerformanceShaders"
                              "-weak_framework MetalPerformanceShadersGraph"
                              "-weak_framework Metal")
list(TRANSFORM _mps_executor_runner__srcs PREPEND "${EXECUTORCH_ROOT}/")
add_executable(mps_executor_runner ${_mps_executor_runner__srcs})
target_include_directories(
  mps_executor_runner INTERFACE ${CMAKE_BINARY_DIR}/schema/include/
                          ${EXECUTORCH_ROOT}/third-party/flatbuffers/include)
target_link_libraries(mps_executor_runner bundled_program program_schema
                                          ${_executor_runner_libs}
                                          ${mps_executor_runner_libs})
target_compile_options(mps_executor_runner PUBLIC ${_common_compile_options})
endif()
