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
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/activation_ops_util.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/copy_ops_util.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/broadcast_util.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/index_util.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/kernel_ops_util.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/matmul_ops_util.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/reduce_util.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/repeat_util.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/slice_util.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_add.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_mul.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_cat.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_softmax.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_native_layer_norm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_quantize.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_dequantize.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_sub.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_div.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_mean.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_slice_copy.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_permute_copy.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_transpose_copy.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_exp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_sigmoid.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_sqrt.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_rsqrt.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_tanh.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_lt.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_where.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_clamp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_hardtanh.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_bmm.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_clone.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_div.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_embedding.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_full.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_permute_copy.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_sigmoid.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_slice_copy.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_split_with_sizes_copy.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_sub.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_to_copy.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_view_copy.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/op_where.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/dtype_util.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/util/normalization_ops_util.cpp"
    "${EXECUTORCH_ROOT}/kernels/portable/cpu/pattern/unary_ufunc_realhbbf16_to_floathbf16.cpp"
)
add_library(aten_ops_cadence ${_aten_ops__srcs})
target_link_libraries(aten_ops_cadence PUBLIC executorch)
target_link_libraries(aten_ops_cadence PRIVATE xa_nnlib)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories
    ${EXECUTORCH_ROOT}/.. ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
)

target_include_directories(
  aten_ops_cadence
  PUBLIC
    ${ROOT_DIR}/..
    ${CMAKE_BINARY_DIR}
    ${_common_include_directories}
    ${EXECUTORCH_ROOT}/backends/cadence/fusion_g3/third-party/nnlib/nnlib-FusionG3/xa_nnlib/algo/common/include/
    ${EXECUTORCH_ROOT}/backends/cadence/fusion_g3/third-party/nnlib/nnlib-FusionG3/xa_nnlib/include/nnlib
    ${EXECUTORCH_ROOT}/backends/cadence/fusion_g3/third-party/nnlib/nnlib-FusionG3/xa_nnlib/include
    ${EXECUTORCH_ROOT}/backends/cadence/fusion_g3/third-party/nnlib/nnlib-FusionG3/xa_nnlib/algo/kernels/tables/include
)

# Generate C++ bindings to register kernels into both PyTorch (for AOT) and
# Executorch (for runtime). Here select all ops in functions.yaml
gen_selected_ops(
  LIB_NAME "cadence_ops_lib" OPS_SCHEMA_YAML
  "${CMAKE_CURRENT_LIST_DIR}/../../aot/functions_fusion_g3.yaml" "" ""
)
generate_bindings_for_kernels(
  LIB_NAME "cadence_ops_lib" OPS_SCHEMA_YAML FUNCTIONS_YAML
  ${CMAKE_CURRENT_SOURCE_DIR}/../../aot/functions_fusion_g3.yaml
)
message("Generated files ${gen_command_sources}")

gen_operators_lib(LIB_NAME "cadence_ops_lib" KERNEL_LIBS DEPS aten_ops_cadence)
