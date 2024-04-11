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

# Source root directory for pytorch.
if(NOT TORCH_ROOT)
  set(TORCH_ROOT ${EXECUTORCH_ROOT}/third-party/pytorch)
endif()

include(${EXECUTORCH_ROOT}/build/Utils.cmake)
include(${EXECUTORCH_ROOT}/build/Codegen.cmake)

if(NOT PYTHON_EXECUTABLE)
  resolve_python_executable()
endif()

# ATen compliant ops that are needed to run this model.
set(_aten_ops__srcs
    "${CMAKE_CURRENT_SOURCE_DIR}/op_add.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_embedding.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_full.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_view_copy.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/broadcast_util.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/repeat_util.cpp")
add_library(aten_ops_cadence ${_aten_ops__srcs})
target_link_libraries(aten_ops_cadence PUBLIC executorch)
target_link_libraries(aten_ops_cadence PRIVATE cadence_kernels)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

target_include_directories(aten_ops_cadence PUBLIC ${ROOT_DIR}/..
                                                ${CMAKE_BINARY_DIR}
                                                ${_common_include_directories})

# Custom ops that are needed to run the test model.
add_library(
  custom_ops "quantized_linear_out.cpp" "quantized_conv_out.cpp"
  "quantized_relu_out.cpp" "quantized_layer_norm.cpp"
  "quantize_per_tensor.cpp" "dequantize_per_tensor.cpp")
target_include_directories(custom_ops PUBLIC ${ROOT_DIR}/..
                                             ${CMAKE_BINARY_DIR}
                                             ${_common_include_directories})

target_link_libraries(custom_ops PUBLIC executorch)
target_link_libraries(custom_ops PRIVATE cadence_kernels)

# Generate C++ bindings to register kernels into both PyTorch (for AOT) and
# Executorch (for runtime). Here select all ops in functions.yaml
gen_selected_ops("${CMAKE_CURRENT_LIST_DIR}/functions.yaml" "" "")
generate_bindings_for_kernels(
  FUNCTIONS_YAML ${CMAKE_CURRENT_SOURCE_DIR}/functions.yaml)
message("Generated files ${gen_command_sources}")

gen_operators_lib(
  "cadence_ops_lib"
  KERNEL_LIBS custom_ops
  DEPS aten_ops_cadence)
