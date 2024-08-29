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
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/activation_ops_util.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/copy_ops_util.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/broadcast_util.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/index_util.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/kernel_ops_util.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/matmul_ops_util.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/reduce_util.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/repeat_util.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/pattern/unary_ufunc_realhb_to_floath.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_bmm.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_cat.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_clone.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_div.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_mean.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_mul.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_permute_copy.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_rsqrt.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_sigmoid.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_slice_copy.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_softmax.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_split_with_sizes_copy.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_sub.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_to_copy.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_where.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_expand_copy.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_gelu.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_empty.cpp"
)
add_library(aten_ops_cadence ${_aten_ops__srcs})
target_link_libraries(aten_ops_cadence PUBLIC executorch)
target_link_libraries(aten_ops_cadence PRIVATE cadence_kernels)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

target_include_directories(
  aten_ops_cadence PUBLIC ${ROOT_DIR}/.. ${CMAKE_BINARY_DIR}
                          ${_common_include_directories}
)

# Custom ops that are needed to run the test model.
add_library(
  custom_ops
  "quantized_linear_out.cpp"
  "quantized_conv_out.cpp"
  "quantized_relu_out.cpp"
  "quantized_layer_norm.cpp"
  "quantize_per_tensor.cpp"
  "dequantize_per_tensor.cpp"
  "quantized_matmul_out.cpp"
)
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

gen_operators_lib(
  LIB_NAME "cadence_ops_lib" KERNEL_LIBS custom_ops DEPS aten_ops_cadence
)
