/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/cadence/generic/operators/cadence_type_util.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

// Generate kernels that perform elementwise arithmetic on two quantized
// tensors. The tensors are either the same size, or the second tensor is a
// scalar.
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
    const T* __restrict__ X_data = X.const_data_ptr<T>();                  \
    const T* __restrict__ Y_data = Y.const_data_ptr<T>();                  \
    T* __restrict__ out_data = out.mutable_data_ptr<T>();                  \
    float inv_out_scale = 1.0f / out_scale;                                \
    for (size_t i = 0, e = X.numel(); i < e; ++i) {                        \
      float x = ::impl::generic::kernels::dequantize<T>(                   \
          X_data[i], X_scale, X_zero_point);                               \
      float y = ::impl::generic::kernels::dequantize<T>(                   \
          Y_data[i], Y_scale, Y_zero_point);                               \
      float z = x OP y;                                                    \
      out_data[i] = ::impl::generic::kernels::quantize<T>(                 \
          z, inv_out_scale, out_zero_point);                               \
    }                                                                      \
  }
