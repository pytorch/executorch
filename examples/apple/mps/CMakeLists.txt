#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

#
# mps_executor_runner: Host tool that demonstrates program execution using
# MPSBackend.
#

cmake_minimum_required(VERSION 3.19)

project(mps_runner_example)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT FLATC_EXECUTABLE)
  set(FLATC_EXECUTABLE flatc)
endif()

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

include(${EXECUTORCH_ROOT}/build/Utils.cmake)

if(NOT PYTHON_EXECUTABLE)
  resolve_python_executable()
endif()

add_compile_options("-Wall" "-Werror")

include(${EXECUTORCH_ROOT}/build/Utils.cmake)

set(_common_compile_options -Wno-deprecated-declarations -fPIC
                            -DET_EVENT_TRACER_ENABLED
)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

# Find prebuilt libraries. executorch package should contain portable_ops_lib,
# etdump, bundled_program.
find_package(executorch CONFIG REQUIRED)
target_include_directories(executorch INTERFACE ${_common_include_directories})
target_compile_options(executorch INTERFACE ${_common_compile_options})

find_package(
  gflags REQUIRED PATHS ${CMAKE_CURRENT_BINARY_DIR}/../../../third-party
)

# ios can only build library but not binary
if(NOT CMAKE_TOOLCHAIN_FILE MATCHES ".*(iOS|ios\.toolchain)\.cmake$")
  #
  # mps_executor_runner: Like executor_runner but with MPS, the binary will be
  # at ${CMAKE_BINARY_DIR}/examples/apple/executor_runner/mps
  #

  # portable_ops_lib
  include(${EXECUTORCH_ROOT}/build/Utils.cmake)
  include(${EXECUTORCH_ROOT}/build/Codegen.cmake)
  gen_selected_ops(LIB_NAME "mps_portable_ops_lib" INCLUDE_ALL_OPS "ON")
  generate_bindings_for_kernels(
    LIB_NAME "mps_portable_ops_lib" FUNCTIONS_YAML
    ${EXECUTORCH_ROOT}/kernels/portable/functions.yaml
  )
  gen_operators_lib(
    LIB_NAME "mps_portable_ops_lib" KERNEL_LIBS portable_kernels DEPS
    executorch
  )

  set(mps_executor_runner_libs
      "-framework Foundation" "-weak_framework MetalPerformanceShaders"
      "-weak_framework MetalPerformanceShadersGraph" "-weak_framework Metal"
  )

  #
  # The `_<target>_srcs` lists are defined by including ${EXECUTORCH_SRCS_FILE}.
  #
  set(EXECUTORCH_SRCS_FILE
      "${CMAKE_CURRENT_BINARY_DIR}/../../../executorch_srcs.cmake"
  )

  extract_sources(${EXECUTORCH_SRCS_FILE})

  set(_mps_schema_headers ${CMAKE_BINARY_DIR}/../../../schema/include/)
  include(${EXECUTORCH_SRCS_FILE})
  target_include_directories(
    bundled_program
    INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/../../../devtools/include
              ${CMAKE_CURRENT_BINARY_DIR}/../../../devtools/bundled_program
              ${EXECUTORCH_ROOT}/third-party/flatbuffers/include
              ${EXECUTORCH_ROOT}/third-party/flatcc/include
              ${_mps_schema_headers}
  )
  list(TRANSFORM _mps_executor_runner__srcs PREPEND "${EXECUTORCH_ROOT}/")
  add_executable(mps_executor_runner ${_mps_executor_runner__srcs})

  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(FLATCC_LIB flatccrt_d)
  else()
    set(FLATCC_LIB flatccrt)
  endif()

  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    target_link_options(mps_executor_runner PUBLIC -fsanitize=undefined)
  endif()

  target_link_libraries(
    mps_executor_runner
    bundled_program
    executorch
    gflags
    etdump
    ${FLATCC_LIB}
    mpsdelegate
    mps_portable_ops_lib
    ${mps_executor_runner_libs}
  )
  target_compile_options(mps_executor_runner PUBLIC ${_common_compile_options})
endif()
