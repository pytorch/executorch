# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# lint_cmake: -linelength
add_library(
  cadence_kernels
  kernels.cpp
  ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/matmul_asym8uxasym8u_asym8u.cpp
  ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/xa_nn_broadcast_32.c
  ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/xa_nn_concat_32.c
  ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/xa_nn_elm_add_f32_broadcast.c
  ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/xa_nn_elm_atan2_f32.c
  ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/xa_nn_elm_clamp_f32_broadcast.c
  ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/xa_nn_elm_div_f32_broadcast.c
  ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/xa_nn_elm_div_mode_f32_broadcast.c
  ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/xa_nn_elm_fmod_broadcast_f32.c
  ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/xa_nn_greater_lesser_equal_f32.c
  ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/xa_nn_elm_logicalxor_bool_bool.c
  ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/xa_nn_elm_minimum_maximum_f32.c
  ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/xa_nn_elm_mul_f32_broadcast.c
  ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/xa_nn_elm_pow_f32.c
  ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/xa_nn_elm_remainder_broadcast_f32.c
  ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/xa_nn_elm_where_f32xf32_f32.c
  ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/xa_nn_reduce_32_32.c
  ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/xa_nn_transpose_32.c
)
# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories
    ${EXECUTORCH_ROOT}/.. ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
)

target_include_directories(
  cadence_kernels
  PUBLIC
    .
    ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/nnlib-hifi4/xa_nnlib/algo/common/include/
    ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/nnlib-hifi4/xa_nnlib/include/nnlib
    ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/nnlib-hifi4/xa_nnlib/include
    ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/nnlib-hifi4/xa_nnlib/algo/ndsp/hifi4/include/
    ${_common_include_directories}
)

target_link_libraries(cadence_kernels PRIVATE xa_nnlib)
