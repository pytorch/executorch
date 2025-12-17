/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_softmax.h>

#include <algorithm>
#include <cmath>

#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/kernels/portable/cpu/util/reduce_util.h>

namespace impl {
namespace generic {
namespace native {

namespace {

using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;

void vec_softmax_f32_f32(
    float* __restrict__ y,
    const float* __restrict__ x,
    int n) {
  // compute softmax(x, x+n) and returns in y
  // y = e ^ (x - max(x)) / sum(e^(x - max(x))
  float max_x = *std::max_element(x, x + n);
  float sum = 0;

  for (int i = 0; i < n; ++i) {
    y[i] = expf(x[i] - max_x);
    sum += y[i];
  }

  for (int i = 0; i < n; ++i) {
    y[i] /= sum;
  }
}

// This function is borrowed from the portable kernel implementation, with only
// float type supported.
void _softmax_portable(const Tensor& in, int64_t dim, Tensor& out) {
  const float* const in_data = in.const_data_ptr<float>();
  float* const out_data = out.mutable_data_ptr<float>();

  torch::executor::apply_over_dim(
      [in_data, out_data](
          const size_t size, const size_t stride, const size_t base) {
        // calculate max in log_softmax dim. During log_softmax
        // computation each value is subtracted by the maximum in
        // value before calling exp to preserve numerical stability.
        const float max_in = torch::executor::apply_unary_reduce_fn(
            [](const float val_in, float val_accum) {
              return std::max(val_in, val_accum);
            },
            in_data + base,
            size,
            stride);

        float temp_sum =
            torch::executor::apply_unary_map_reduce_fn<float, float>(
                [max_in](const float val_in) {
                  return std::exp(val_in - max_in);
                },
                [](const float mapped_in, float val_accum) {
                  return val_accum + mapped_in;
                },
                in_data + base,
                size,
                stride);

        torch::executor::apply_unary_map_fn(
            [max_in, temp_sum](const float val_in) {
              return std::exp(val_in - max_in) / temp_sum;
            },
            in_data + base,
            out_data + base,
            size,
            stride);
      },
      in,
      dim);
}

} // namespace

Tensor& _softmax_out(
    ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& X,
    int64_t dim,
    ET_UNUSED bool half_to_float,
    Tensor& Y) {
  if (dim < 0) {
    dim += X.dim();
  }

  // If dim is not the last dimension, we cannot use the kernel below.
  // Falling back on a more generic kernel.
  if (dim < X.dim() - 1) {
    _softmax_portable(X, dim, Y);
    return Y;
  }

  const float* __restrict__ x_data = X.const_data_ptr<float>();
  float* __restrict__ y_data = Y.mutable_data_ptr<float>();

  size_t K = X.size(X.dim() - 1);
  size_t leading_dim = ::executorch::runtime::getLeadingDims(X, X.dim() - 1);

  for (size_t i = 0; i < leading_dim; ++i) {
    const float* x = x_data + i * K;
    float* y = y_data + i * K;
    vec_softmax_f32_f32(y, x, K);
  }

  return Y;
}

Tensor& _softmax_f32_f32_out(
    __ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& X,
    int64_t dim,
    __ET_UNUSED ::executorch::aten::optional<bool> half_to_float,
    Tensor& Y) {
  _softmax_out(ctx, X, dim, false, Y);

  return Y;
}

} // namespace native
} // namespace generic
} // namespace impl
