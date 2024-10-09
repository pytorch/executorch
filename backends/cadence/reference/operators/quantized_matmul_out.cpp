/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/reference/kernels/kernels.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace impl {
namespace reference {
namespace native {

using executorch::aten::Tensor;
using executorch::runtime::getLeadingDims;
using executorch::runtime::KernelRuntimeContext;

// The quantized matmul. The quantized matmul accumulates in a wider register,
// whose type is TA.
template <
    typename TZ,
    typename TA = float,
    bool transposed = false,
    typename TX = TZ,
    typename TY = TZ>
__attribute__((noinline)) void qmatmul(
    TZ* __restrict__ Z,
    int32_t Z_multiplier,
    int32_t Z_shift,
    int32_t Z_zero_point,
    const TX* __restrict__ X,
    int32_t X_zero_point,
    const TY* __restrict__ y,
    int32_t Y_zero_point,
    size_t m,
    size_t n,
    size_t p) {
  // Compute the Z_scale from Z_multiplier and Z_shift
  const float Z_scale = -Z_multiplier * 1.0 / (1 << 31) * pow(2, Z_shift);
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < p; ++j) {
      TA sum = 0;
      for (size_t k = 0; k < n; ++k) {
        if (transposed) {
          sum += (X[i * n + k] - X_zero_point) * (y[j * n + k] - Y_zero_point);
        } else {
          sum += (X[i * n + k] - X_zero_point) * (y[k * p + j] - Y_zero_point);
        }
      }
      Z[i * p + j] = kernels::quantize<uint8_t>(sum, Z_scale, Z_zero_point);
    }
  }
}

template <typename T>
void inline _typed_quantized_matmul(
    const Tensor& X,
    int64_t X_zero_point,
    const Tensor& Y,
    int64_t Y_zero_point,
    const exec_aten::optional<Tensor>& bias,
    int64_t out_multiplier,
    int64_t out_shift,
    int64_t out_zero_point,
    bool transposed,
    Tensor& out) {
  size_t batch_size = getLeadingDims(X, X.dim() - 2);
  size_t leading_dim = X.size(X.dim() - 2);
  size_t out_dim = Y.size(Y.dim() - 1 - transposed);
  size_t in_dim = X.size(X.dim() - 1);

  T* __restrict__ out_data = out.mutable_data_ptr<T>();
  const T* __restrict__ X_data = X.const_data_ptr<T>();
  const T* __restrict__ Y_data = Y.const_data_ptr<T>();
  for (size_t i = 0; i < batch_size; ++i) {
    const T* x = X_data + i * leading_dim * in_dim;
    const T* y = Y_data + i * in_dim * out_dim;
    T* z = out_data + i * leading_dim * out_dim;
    if (transposed) {
      qmatmul<T, int32_t, true>(
          z,
          static_cast<int32_t>(out_multiplier),
          static_cast<int32_t>(out_shift),
          static_cast<int32_t>(out_zero_point),
          x,
          static_cast<int32_t>(X_zero_point),
          y,
          static_cast<int32_t>(Y_zero_point),
          leading_dim,
          in_dim,
          out_dim);
    } else {
      qmatmul<T, int32_t, false>(
          z,
          static_cast<int32_t>(out_multiplier),
          static_cast<int32_t>(out_shift),
          static_cast<int32_t>(out_zero_point),
          x,
          static_cast<int32_t>(X_zero_point),
          y,
          static_cast<int32_t>(Y_zero_point),
          leading_dim,
          in_dim,
          out_dim);
    }
  }
}

void quantized_matmul_out(
    KernelRuntimeContext& ctx,
    const Tensor& X,
    int64_t X_zero_point,
    const Tensor& Y,
    int64_t Y_zero_point,
    const exec_aten::optional<Tensor>& bias,
    int64_t out_multiplier,
    int64_t out_shift,
    int64_t out_zero_point,
    bool transposed,
    Tensor& out) {
  if (out.scalar_type() == exec_aten::ScalarType::Byte) {
    _typed_quantized_matmul<uint8_t>(
        X,
        X_zero_point,
        Y,
        Y_zero_point,
        bias,
        out_multiplier,
        out_shift,
        out_zero_point,
        transposed,
        out);
  } else if (out.scalar_type() == exec_aten::ScalarType::Char) {
    _typed_quantized_matmul<int8_t>(
        X,
        X_zero_point,
        Y,
        Y_zero_point,
        bias,
        out_multiplier,
        out_shift,
        out_zero_point,
        transposed,
        out);
  }
}

}; // namespace native
}; // namespace reference
}; // namespace impl
