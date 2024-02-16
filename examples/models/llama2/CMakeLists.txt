#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

#
# llama_executor_runner
#

cmake_minimum_required(VERSION 3.19)

project(llama_runner_example)

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

# Source root directory for pytorch.
if(NOT TORCH_ROOT)
  set(TORCH_ROOT ${EXECUTORCH_ROOT}/third-party/pytorch)
endif()

add_compile_options("-Wall" "-Werror")

include(${EXECUTORCH_ROOT}/build/Utils.cmake)

set(_common_compile_options -Wno-deprecated-declarations -fPIC)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

# Find prebuilt libraries. executorch package should contain
# portable_ops_lib, etdump, bundled_program.
find_package(executorch CONFIG REQUIRED)
target_include_directories(executorch INTERFACE ${_common_include_directories})
target_compile_options(executorch INTERFACE -DET_EVENT_TRACER_ENABLED)

find_package(
  gflags REQUIRED PATHS ${CMAKE_CURRENT_BINARY_DIR}/../../../third-party
)

# portable_ops_lib
include(${EXECUTORCH_ROOT}/build/Utils.cmake)
include(${EXECUTORCH_ROOT}/build/Codegen.cmake)
gen_selected_ops("" "" "ON")
generate_bindings_for_kernels(
  ${EXECUTORCH_ROOT}/kernels/portable/functions.yaml ""
)
gen_operators_lib("portable_ops_lib" portable_kernels executorch)

set(llama_runner_libs "-framework Foundation")

#
# The `_<target>_srcs` lists are defined by including ${EXECUTORCH_SRCS_FILE}.
#
set(
  EXECUTORCH_SRCS_FILE
  "${CMAKE_CURRENT_BINARY_DIR}/../../../executorch_srcs.cmake"
)

extract_sources(${EXECUTORCH_SRCS_FILE})


add_library(extension_module "${CMAKE_CURRENT_SOURCE_DIR}/../../../extension/module/module.cpp")
target_include_directories(extension_module PUBLIC ${EXECUTORCH_ROOT}/..)

#add_library(extension_data_loader "${CMAKE_CURRENT_SOURCE_DIR}/../../../extension/data_loader/mmap_data_loader.cpp")
#target_link_libraries(extension_data_loader executorch)


# Install libraries
install(
  TARGETS extension_module
  DESTINATION ${CMAKE_BINARY_DIR}/lib
  INCLUDES
  DESTINATION ${_common_include_directories})


set(llama_schema_headers ${CMAKE_BINARY_DIR}/../../../schema/include/)
include(${EXECUTORCH_SRCS_FILE})
target_include_directories(
  bundled_program
  INTERFACE
  ${CMAKE_CURRENT_BINARY_DIR}/../../../sdk/include
  ${CMAKE_CURRENT_BINARY_DIR}/../../../sdk/bundled_program
  ${EXECUTORCH_ROOT}/third-party/flatbuffers/include
  ${EXECUTORCH_ROOT}/third-party/flatcc/include
  ${llama_schema_headers}
)

# Define a CMake target using runtime.cxx_library
add_library(runner
    "${CMAKE_CURRENT_SOURCE_DIR}/runner/runner.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/sampler/sampler.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tokenizer/tokenizer.cpp"
)

# Specify exported headers
target_include_directories(runner PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../..
    ${CMAKE_CURRENT_SOURCE_DIR}/runner
    ${CMAKE_CURRENT_SOURCE_DIR}/sampler
    ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../third-party/pytorch/torch/csrc/api/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../third-party/pytorch
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../libtorch/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../libtorch/include/torch/csrc/api/include/
)

get_target_property(runner_include_dirs runner INCLUDE_DIRECTORIES)

target_sources(runner PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/runner/runner.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/runner/runner.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/runner/util.h"
)

add_library(sampler
    "${CMAKE_CURRENT_SOURCE_DIR}/sampler/sampler.cpp"
)

target_include_directories(sampler PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../..
    ${CMAKE_CURRENT_SOURCE_DIR}/sampler
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../third-party/pytorch/torch/csrc/api/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../third-party/pytorch
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../libtorch/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../libtorch/include/torch/csrc/api/include/
)


target_sources(sampler PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sampler/sampler.cpp"
)


add_library(tokenizer
    "${CMAKE_CURRENT_SOURCE_DIR}/tokenizer/tokenizer.cpp"
)

target_include_directories(tokenizer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../..
    ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../third-party/pytorch/torch/csrc/api/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../third-party/pytorch
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../libtorch/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../libtorch/include/torch/csrc/api/include/
)

target_sources(tokenizer PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/tokenizer/tokenizer.cpp"
)

# Specify visibility
target_compile_options(runner PRIVATE
    -fvisibility=hidden
)

# Specify exported dependencies
target_link_libraries(runner PRIVATE
    //executorch/backends/xnnpack:xnnpack_backend
    //executorch/examples/models/llama2/sampler:sampler
    //executorch/examples/models/llama2/tokenizer:tokenizer
    //executorch/examples/models/llama2/runner:runner
    //executorch/extension/evalue_util:print_evalue
    //executorch/extension/runner_util:managed_tensor
    //executorch/extension/module:module
    //executorch/kernels/quantized:generated_lib
    //executorch/kernels/portable:generated_lib_all_ops
    //executorch/runtime/core/exec_aten:lib
    //executorch/runtime/kernel:kernel_includes
)

set(_llama_executor_runner__srcs
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/runner/runner.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/sampler/sampler.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tokenizer/tokenizer.cpp"
)

#list(TRANSFORM _llama_executor_runner__srcs PREPEND "${EXECUTORCH_ROOT}/")
add_executable(llama_executor_runner ${_llama_executor_runner__srcs})

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(FLATCC_LIB flatcc_d)
else()
    set(FLATCC_LIB flatcc)
endif()

target_link_libraries(llama_executor_runner bundled_program
                                          executorch
                                          gflags
                                          etdump
                                          ${FLATCC_LIB}
                                          portable_ops_lib
                                          extension_module
                                          extension_data_loader
                                          ${llama_runner_libs})
target_compile_options(llama_executor_runner PUBLIC ${_common_compile_options})
