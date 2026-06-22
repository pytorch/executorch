# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Set the minimum required version of CMake for this project.
cmake_minimum_required(VERSION 3.10)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Set the project name.
project(cadence_backend)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories
    ${EXECUTORCH_ROOT}/.. ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
)

add_compile_definitions(C10_USING_CUSTOM_GENERATED_MACROS)

if(EXECUTORCH_CADENCE_CPU_RUNNER)
  include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)

  set(_common_compile_options -Wno-deprecated-declarations -fPIC)

  # Find prebuilt libraries. executorch package should contain portable_ops_lib,
  # etdump, bundled_program.
  find_package(executorch CONFIG REQUIRED)
  executorch_target_link_options_shared_lib(executorch)
  executorch_target_link_options_shared_lib(portable_ops_lib)

  target_include_directories(
    executorch INTERFACE ${_common_include_directories}
  )

  find_package(
    gflags REQUIRED PATHS ${CMAKE_CURRENT_BINARY_DIR}/../../third-party
  )

  add_executable(
    cadence_runner
    ${EXECUTORCH_ROOT}/examples/devtools/example_runner/example_runner.cpp
  )
  target_compile_options(executorch INTERFACE -DET_EVENT_TRACER_ENABLED)

  target_include_directories(
    etdump INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/../../devtools/include
                     ${EXECUTORCH_ROOT}/third-party/flatcc/include
  )

  target_include_directories(
    cadence_runner PUBLIC ${ROOT_DIR}/.. ${CMAKE_BINARY_DIR}
                          ${_common_include_directories}
  )

  target_link_libraries(
    cadence_runner
    executorch
    gflags
    etdump
    extension_data_loader
    bundled_program
    cadence_ops_lib
    flatccrt
  )
endif()

if(EXECUTORCH_NNLIB_OPT)
  set(TARGET_DIR hifi)
  add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_DIR}/third-party/nnlib
    ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
  )
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_DIR}/kernels)
elseif(EXECUTORCH_FUSION_G3_OPT)
  set(TARGET_DIR fusion_g3)
  add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_DIR}/third-party/nnlib
    ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
  )
elseif(EXECUTORCH_VISION_OPT)
  set(TARGET_DIR vision)
  add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_DIR}/third-party
    ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
  )
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_DIR}/kernels)
else()
  set(TARGET_DIR generic)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_DIR}/kernels)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_DIR}/operators)

# Cadence executor_runner: cross-compiled ExecuTorch runner for the Xtensa ISS
# (xt-run / xt-run --turbo). Self-contained, gflags-free argv parser, reads .pte
# via xt-run semi-hosting.
#
# Usage: cmake ... -DEXECUTORCH_BUILD_CADENCE_RUNNER=ON xt-run --turbo
# cmake-out/backends/cadence/cadence_executor_runner \ --model_path=add.pte
if(EXECUTORCH_BUILD_CADENCE_RUNNER)
  add_executable(cadence_executor_runner cadence_executor_runner.cpp)
  target_compile_definitions(
    cadence_executor_runner PRIVATE ET_ENABLE_ENUM_STRINGS=0
  )
  target_include_directories(
    cadence_executor_runner
    PRIVATE ${_common_include_directories} ${CMAKE_BINARY_DIR}
            ${CMAKE_BINARY_DIR}/include
  )
  # Mirror the upstream executor_runner cadence link list (top-level
  # CMakeLists.txt: list(APPEND _executor_runner_libs cadence_ops_lib)). Do NOT
  # add --whole-archive: cadence_ops_lib is also pulled transitively, and
  # forcing a second copy double-runs its static kernel-registration
  # initializers and asserts at runtime.
  target_link_libraries(
    cadence_executor_runner PRIVATE executorch extension_evalue_util
                                    extension_runner_util cadence_ops_lib
  )
  # Vision and Fusion-G3 ops (e.g. op_softmax) reference iDMA scheduling symbols
  # and those cores ship libidma in their LSP. HiFi4 and generic cores do not
  # use iDMA and their LSPs may not provide libidma, so only link it for the
  # cores that need it.
  if(EXECUTORCH_VISION_OPT OR EXECUTORCH_FUSION_G3_OPT)
    target_link_options(cadence_executor_runner PRIVATE -lidma)
  endif()
  target_link_options(cadence_executor_runner PRIVATE -static -lm)
endif()
