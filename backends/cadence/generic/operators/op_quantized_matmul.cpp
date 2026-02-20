/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_quantized_matmul.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>

#include <executorch/backends/cadence/generic/kernels/kernels.h>
#include <executorch/backends/cadence/generic/operators/cadence_type_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

namespace impl {
namespace generic {
namespace native {

using ::executorch::aten::IntArrayRef;
using ::executorch::aten::optional;
using ::executorch::aten::Scalar;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;
using ::impl::generic::kernels::quantize;

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
      Z[i * p + j] = quantize<TZ>(sum, Z_scale, Z_zero_point);
    }
  }
}

Tensor& quantized_matmul_out(
    ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& X,
    int64_t X_zero_point,
    const Tensor& Y,
    int64_t Y_zero_point,
    ET_UNUSED const optional<Tensor>& bias,
    int64_t out_multiplier,
    int64_t out_shift,
    int64_t out_zero_point,
    bool transposed,
    Tensor& out) {
  size_t batch_size = ::executorch::runtime::getLeadingDims(X, X.dim() - 2);
  size_t leading_dim = X.size(X.dim() - 2);
  size_t out_dim = Y.size(Y.dim() - 1 - transposed);
  size_t in_dim = X.size(X.dim() - 1);

  // Handle W8A16 heterogeneous type (int16_t activations, int8_t weights)
  if (out.scalar_type() == ScalarType::Short &&
      X.scalar_type() == ScalarType::Short &&
      Y.scalar_type() == ScalarType::Char) {
    int16_t* __restrict__ out_data = out.mutable_data_ptr<int16_t>();
    const int16_t* __restrict__ X_data = X.const_data_ptr<int16_t>();
    const int8_t* __restrict__ Y_data = Y.const_data_ptr<int8_t>();
    for (size_t i = 0; i < batch_size; ++i) {
      const int16_t* x = X_data + i * leading_dim * in_dim;
      const int8_t* y = Y_data + i * in_dim * out_dim;
      int16_t* z = out_data + i * leading_dim * out_dim;
      if (transposed) {
        qmatmul<int16_t, int32_t, true, int16_t, int8_t>(
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
        qmatmul<int16_t, int32_t, false, int16_t, int8_t>(
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
    return out;
  }

#define typed_quantized_matmul(ctype, dtype)                      \
  case ScalarType::dtype: {                                       \
    ctype* __restrict__ out_data = out.mutable_data_ptr<ctype>(); \
    const ctype* __restrict__ X_data = X.const_data_ptr<ctype>(); \
    const ctype* __restrict__ Y_data = Y.const_data_ptr<ctype>(); \
    for (size_t i = 0; i < batch_size; ++i) {                     \
      const ctype* x = X_data + i * leading_dim * in_dim;         \
      const ctype* y = Y_data + i * in_dim * out_dim;             \
      ctype* z = out_data + i * leading_dim * out_dim;            \
      if (transposed) {                                           \
        qmatmul<ctype, int32_t, true>(                            \
            z,                                                    \
            static_cast<int32_t>(out_multiplier),                 \
            static_cast<int32_t>(out_shift),                      \
            static_cast<int32_t>(out_zero_point),                 \
            x,                                                    \
            static_cast<int32_t>(X_zero_point),                   \
            y,                                                    \
            static_cast<int32_t>(Y_zero_point),                   \
            leading_dim,                                          \
            in_dim,                                               \
            out_dim);                                             \
      } else {                                                    \
        qmatmul<ctype, int32_t, false>(                           \
            z,                                                    \
            static_cast<int32_t>(out_multiplier),                 \
            static_cast<int32_t>(out_shift),                      \
            static_cast<int32_t>(out_zero_point),                 \
            x,                                                    \
            static_cast<int32_t>(X_zero_point),                   \
            y,                                                    \
            static_cast<int32_t>(Y_zero_point),                   \
            leading_dim,                                          \
            in_dim,                                               \
            out_dim);                                             \
      }                                                           \
    }                                                             \
    break;                                                        \
  }

  ScalarType dtype = out.scalar_type();
  switch (dtype) {
    ET_FORALL_CADENCE_QUANTIZED_TYPES_WITH_INT16(typed_quantized_matmul);
    default:
      ET_DCHECK_MSG(
          false, "Unhandled dtype %s", torch::executor::toString(dtype));
  }
#undef typed_quantized_matmul
  return out;
}

template <typename T>
void _typed_quantized_matmul(
    const Tensor& X,
    int64_t X_zero_point,
    const Tensor& Y,
    int64_t Y_zero_point,
    ET_UNUSED const optional<Tensor>& bias,
    int64_t out_multiplier,
    int64_t out_shift,
    int64_t out_zero_point,
    bool transposed,
    Tensor& out) {
  size_t batch_size = ::executorch::runtime::getLeadingDims(X, X.dim() - 2);
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

Tensor& quantized_matmul_asym8sxasym8s_asym8s_out(
    ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& X,
    int64_t X_zero_point,
    const Tensor& Y,
    int64_t Y_zero_point,
    const optional<Tensor>& bias,
    int64_t out_multiplier,
    int64_t out_shift,
    int64_t out_zero_point,
    bool transposed,
    Tensor& out) {
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
  return out;
}

Tensor& quantized_matmul_asym8uxasym8u_asym8u_out(
    ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& X,
    int64_t X_zero_point,
    const Tensor& Y,
    int64_t Y_zero_point,
    const optional<Tensor>& bias,
    int64_t out_multiplier,
    int64_t out_shift,
    int64_t out_zero_point,
    bool transposed,
    Tensor& out) {
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
  return out;
}

} // namespace native
} // namespace generic
} // namespace impl
