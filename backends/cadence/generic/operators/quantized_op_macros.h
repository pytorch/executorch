/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/cadence/generic/operators/cadence_type_util.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

#define DECLARE_POINTWISE_TENSOR_QUANTIZED_BINARY_OP(BINARY_FUNC_NAME, OP) \
  template <typename T>                                                    \
  void BINARY_FUNC_NAME(                                                   \
      const ::executorch::aten::Tensor& X,                                 \
      float X_scale,                                                       \
      int32_t X_zero_point,                                                \
      const ::executorch::aten::Tensor& Y,                                 \
      float Y_scale,                                                       \
      int32_t Y_zero_point,                                                \
      float out_scale,                                                     \
      int32_t out_zero_point,                                              \
      ::executorch::aten::Tensor& out) {                                   \
    float inv_out_scale = 1.0f / out_scale;                                \
    ::torch::executor::apply_binary_elementwise_fn<T, T, T>(              \
        [X_scale, X_zero_point, Y_scale, Y_zero_point, inv_out_scale,     \
         out_zero_point](const T x_val, const T y_val) {                   \
          float x = ::impl::generic::kernels::dequantize<T>(               \
              x_val, X_scale, X_zero_point);                               \
          float y = ::impl::generic::kernels::dequantize<T>(               \
              y_val, Y_scale, Y_zero_point);                               \
          float z = x OP y;                                                \
          return ::impl::generic::kernels::quantize<T>(                    \
              z, inv_out_scale, out_zero_point);                           \
        },                                                                 \
        X,                                                                 \
        Y,                                                                 \
        out);                                                              \
  }
