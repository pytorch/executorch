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

if(NOT PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE python3)
endif()

# Source root directory for pytorch.
if(NOT TORCH_ROOT)
  set(TORCH_ROOT ${EXECUTORCH_ROOT}/third-party/pytorch)
endif()

include(${EXECUTORCH_ROOT}/build/Utils.cmake)
include(${EXECUTORCH_ROOT}/build/Codegen.cmake)

# ATen compliant ops that are needed to run this model.
set(_aten_ops__srcs
    "${CMAKE_CURRENT_SOURCE_DIR}/op_add.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_embedding.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_full.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_view_copy.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/broadcast_util.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/repeat_util.cpp")
add_library(aten_ops_xtensa ${_aten_ops__srcs})
target_link_libraries(aten_ops_xtensa PUBLIC executorch)

# Custom ops that are needed to run the test model.
add_library(
  custom_ops "quantized_linear_out.cpp" "quantized_conv_out.cpp"
  "quantized_relu.cpp" "quantized_layer_norm.cpp"
  "quantize_per_tensor.cpp" "dequantize_per_tensor.cpp")
target_link_libraries(custom_ops PUBLIC executorch)
target_link_libraries(custom_ops PRIVATE xtensa_kernels)

# Generate C++ bindings to register kernels into both PyTorch (for AOT) and
# Executorch (for runtime). Here select all ops in functions.yaml
gen_selected_ops("${CMAKE_CURRENT_LIST_DIR}/functions.yaml" "" "")
generate_bindings_for_kernels(${CMAKE_CURRENT_SOURCE_DIR}/functions.yaml "")
message("Generated files ${gen_command_sources}")

gen_operators_lib("xtensa_ops_lib" custom_ops aten_ops_xtensa)
