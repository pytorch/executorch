# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories
    ${EXECUTORCH_ROOT}/.. ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
)

# Custom ops that are needed to run the test model.
file(GLOB custom_ops_srcs "*.cpp")
add_library(custom_ops ${custom_ops_srcs})

target_include_directories(
  custom_ops PUBLIC ${ROOT_DIR}/.. ${CMAKE_BINARY_DIR}
                    ${_common_include_directories}
)

target_link_libraries(custom_ops PUBLIC executorch)
target_link_libraries(custom_ops PRIVATE cadence_kernels)

# Generate C++ bindings to register kernels into both PyTorch (for AOT) and
# Executorch (for runtime). Here select all ops in functions.yaml
gen_selected_ops(
  LIB_NAME "cadence_ops_lib" OPS_SCHEMA_YAML
  "${CMAKE_CURRENT_LIST_DIR}/../../aot/functions.yaml" "" ""
)
generate_bindings_for_kernels(
  LIB_NAME "cadence_ops_lib" OPS_SCHEMA_YAML FUNCTIONS_YAML
  ${CMAKE_CURRENT_SOURCE_DIR}/../../aot/functions.yaml
)
message("Generated cadence x86 files ${gen_command_sources}")

# The ATen ops in functions.yaml use the portable kernels. Registration comes
# only from functions.yaml, so linking the whole library adds no other ops.
gen_operators_lib(
  LIB_NAME "cadence_ops_lib" KERNEL_LIBS custom_ops portable_kernels
)
