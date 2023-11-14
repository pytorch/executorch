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

set(_bundled_program_schema_dir
     ${EXECUTORCH_ROOT}/sdk/bundled_program/schema)

set(_bundled_program_schema__srcs
    ${_bundled_program_schema_dir}/bundled_program_schema.fbs
    ${_bundled_program_schema_dir}/scalar_type.fbs)

# The include directory that will contain the generated schema headers.
set(_program_schema__include_dir "${CMAKE_BINARY_DIR}/schema/include")

set(_bundled_program_schema__outputs)
foreach(fbs_file ${_bundled_program_schema__srcs})
  string(REGEX REPLACE "[.]fbs$" "_generated.h" generated "${fbs_file}")
  list(APPEND _bundled_program_schema__outputs
        "${generated}")
  message("OUTPUTS: " ${_bundled_program_schema__outputs})
endforeach()

add_custom_command(
  OUTPUT ${_bundled_program_schema__outputs}
  COMMAND
    ${FLATC_EXECUTABLE} --cpp --cpp-std c++11 --gen-mutable --scoped-enums -o
    "${_program_schema__include_dir}/executorch/sdk/bundled_program/schema" ${_bundled_program_schema__srcs}
  WORKING_DIRECTORY ${EXECUTORCH_ROOT}
  DEPENDS ${FLATC_EXECUTABLE} ${_bundled_program_schema__srcs}
  COMMENT "Generating bundled_program headers"
  VERBATIM)

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
add_executable(mps_executor_runner ${_mps_executor_runner__srcs} ${_bundled_program_schema__outputs})
target_include_directories(
  mps_executor_runner INTERFACE ${CMAKE_BINARY_DIR}/schema/include/
                          ${EXECUTORCH_ROOT}/third-party/flatbuffers/include)
target_link_libraries(mps_executor_runner program_schema
                                          ${_executor_runner_libs}
                                          ${mps_executor_runner_libs})
target_compile_options(mps_executor_runner PUBLIC ${_common_compile_options})
endif()
