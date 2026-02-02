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

# ATen compliant ops that are needed to run this model.
set(_aten_ops__srcs
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_add.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_atan2.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_bitwise_and.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_bitwise_or.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_bitwise_xor.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_bmm.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_cat.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_clamp.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_div.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_embedding.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_eq.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_fmod.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_full.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_ge.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_gt.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_hardtanh.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_le.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_lt.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_masked_fill.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_maximum.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_mean.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_minimum.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_mm.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_mul.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_ne.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_permute_copy.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_pow.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_remainder.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_rsqrt.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_select_copy.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_slice_copy.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_softmax.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_split_with_sizes_copy.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_sigmoid.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_sub.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_tanh.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_view_copy.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/hifi/operators/op_where.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_clone.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_gelu.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_max_pool2d_with_indices.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_to_copy.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/pattern/unary_ufunc_realhbbf16_to_floathbf16.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/activation_ops_util.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/broadcast_util.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/copy_ops_util.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/dtype_util.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/delinearize_index.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/index_util.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/kernel_ops_util.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/matmul_ops_util.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/reduce_util.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/repeat_util.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/select_copy_util.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/slice_util.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/delinearize_index.cpp"
)
add_library(aten_ops_cadence ${_aten_ops__srcs})
target_link_libraries(aten_ops_cadence PUBLIC executorch)
target_link_libraries(aten_ops_cadence PRIVATE cadence_kernels)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories
    ${EXECUTORCH_ROOT}/.. ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
)

target_include_directories(
  aten_ops_cadence PUBLIC ${ROOT_DIR}/.. ${CMAKE_BINARY_DIR}
                          ${_common_include_directories}
)

# Generic kernels that the generic operators depend on.
# These provide impl::generic::kernels functions like quantize/dequantize.
set(_generic_kernels__srcs
    "${EXECUTORCH_ROOT}/backends/cadence/generic/kernels/kernels.cpp"
)

# Generic custom ops that HiFi operators depend on as fallbacks.
# These provide impl::generic::native implementations called by HiFi operators.
set(_generic_custom_ops__srcs
    "${EXECUTORCH_ROOT}/backends/cadence/generic/operators/op_quantized_linear.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/generic/operators/op_quantized_conv2d.cpp"
    "${EXECUTORCH_ROOT}/backends/cadence/generic/operators/op_quantized_matmul.cpp"
)

# Custom ops that are needed to run the test model.
add_library(
  custom_ops
  ${_generic_kernels__srcs}
  ${_generic_custom_ops__srcs}
  "op_quantized_add_asym8sxasym8s_asym8s_per_tensor_out.cpp"
  "op_quantized_add_asym8uxasym8u_asym8u_per_tensor_out.cpp"
  "op_quantized_linear_out.cpp"
  "op_quantized_layer_norm.cpp"
  "op_quantized_linear_asym8sxasym8s_asym8s_per_tensor_out.cpp"
  "op_quantized_linear_asym8uxasym8u_asym8u_per_tensor_out.cpp"
  "op_quantized_matmul_out.cpp"
  "op_quantized_matmul_asym8sxasym8s_asym8s_out.cpp"
  "op_quantized_matmul_asym8uxasym8u_asym8u_out.cpp"
  "op_quantize_per_tensor.cpp"
  "op_quantize_per_tensor_asym8s.cpp"
  "op_quantized_relu_out.cpp"
  "op_quantized_relu_asym8s_asym8s_per_tensor_out.cpp"
  "op_quantized_relu_asym8u_asym8u_per_tensor_out.cpp"
  "op_dequantize_per_tensor.cpp"
  "op_dequantize_per_tensor_asym8s.cpp"
  "op_quantized_conv1d_ncl_asym8sxsym8s_asym8s_per_tensor_out.cpp"
  "op_quantized_conv1d_ncl_asym8uxsym8u_asym8u_per_tensor_out.cpp"
  "op_quantized_conv1d_nlc_asym8sxsym8s_asym8s_per_tensor_out.cpp"
  "op_quantized_conv1d_nlc_asym8uxsym8u_asym8u_per_tensor_out.cpp"
  "op_quantized_conv2d_nchw_out.cpp"
  "op_quantized_conv2d_nchw_asym8sxsym8s_asym8s_per_tensor_out.cpp"
  "op_quantized_conv2d_nchw_asym8uxsym8u_asym8u_per_tensor_out.cpp"
  "op_quantized_conv2d_nchw_depthwise_asym8sxsym8s_asym8s_per_tensor_out.cpp"
  "op_quantized_conv2d_nchw_depthwise_asym8uxsym8u_asym8u_per_tensor_out.cpp"
  "op_quantized_conv2d_nchw_dilated_asym8sxsym8s_asym8s_per_tensor_out.cpp"
  "op_quantized_conv2d_nchw_dilated_asym8uxsym8u_asym8u_per_tensor_out.cpp"
  "op_quantized_conv2d_nhwc_out.cpp"
  "op_quantized_conv2d_nhwc_asym8sxsym8s_asym8s_per_tensor_out.cpp"
  "op_quantized_conv2d_nhwc_asym8uxsym8u_asym8u_per_tensor_out.cpp"
  "op_quantized_conv2d_nhwc_depthwise_asym8sxsym8s_asym8s_per_tensor_out.cpp"
  "op_quantized_conv2d_nhwc_depthwise_asym8uxsym8u_asym8u_per_tensor_out.cpp"
  "op_quantized_conv2d_nhwc_dilated_asym8sxsym8s_asym8s_per_tensor_out.cpp"
  "op_quantized_conv2d_nhwc_dilated_asym8uxsym8u_asym8u_per_tensor_out.cpp"
  "op_quantized_fully_connected_out.cpp"
  "op_quantized_fully_connected_asym8sxasym8s_asym8s_per_tensor_out.cpp"
  "op_quantized_fully_connected_asym8uxasym8u_asym8u_per_tensor_out.cpp"
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
  "${CMAKE_CURRENT_LIST_DIR}/../../aot/functions_hifi.yaml" "" ""
)
generate_bindings_for_kernels(
  LIB_NAME "cadence_ops_lib" OPS_SCHEMA_YAML FUNCTIONS_YAML
  ${CMAKE_CURRENT_SOURCE_DIR}/../../aot/functions_hifi.yaml
)
message("Generated files ${gen_command_sources}")

gen_operators_lib(
  LIB_NAME "cadence_ops_lib" KERNEL_LIBS custom_ops DEPS aten_ops_cadence
)
