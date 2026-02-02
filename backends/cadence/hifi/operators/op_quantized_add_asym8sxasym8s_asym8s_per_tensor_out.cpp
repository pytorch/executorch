/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace impl {
namespace HiFi {
namespace native {

using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;
// using ::impl::generic::kernels::dequantize;
// using ::impl::generic::kernels::quantize;

void quantized_add_asym8sxasym8s_asym8s_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& X,
    double X_scale,
    int64_t X_zero_point,
    const Tensor& Y,
    double Y_scale,
    int64_t Y_zero_point,
    double out_scale,
    int64_t out_zero_point,
    Tensor& out) {
  const int8_t* __restrict__ X_data = X.const_data_ptr<int8_t>();
  const int8_t* __restrict__ Y_data = Y.const_data_ptr<int8_t>();
  int8_t* __restrict__ out_data = out.mutable_data_ptr<int8_t>();

  ssize_t Y_numel = Y.numel();
  ssize_t X_numel = X.numel();
  ssize_t out_numel = out.numel();

  float X_scale_f = static_cast<float>(X_scale);
  float Y_scale_f = static_cast<float>(Y_scale);
  float out_scale_f = static_cast<float>(out_scale);
  int32_t X_zero_point_i32 = static_cast<int32_t>(X_zero_point);
  int32_t Y_zero_point_i32 = static_cast<int32_t>(Y_zero_point);
  int32_t out_zero_point_i32 = static_cast<int32_t>(out_zero_point);

  float inv_out_scale = 1.0f / out_scale_f;
  constexpr float min_val =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  constexpr float max_val =
      static_cast<float>(std::numeric_limits<int8_t>::max());

  /* Tensor X exactly matches Y in shape, no broadcasting */
  if (X_numel == Y_numel && Y_numel == out_numel) {
    for (size_t i = 0; i < X_numel; ++i) {
      float x = X_scale_f * (X_data[i] - X_zero_point_i32);
      float y = Y_scale_f * (Y_data[i] - Y_zero_point_i32);
      float z = x + y;
      float tmp = roundf(z * inv_out_scale + out_zero_point_i32);
      out_data[i] =
          static_cast<int8_t>(std::max(std::min(tmp, max_val), min_val));
    }
  } /* if Y is a scalar Tensor */
  else if (Y_numel == 1) {
    float y =
        kernels::dequantize<int8_t>(Y_data[0], Y_scale_f, Y_zero_point_i32);
    for (size_t i = 0; i < X_numel; ++i) {
      float x =
          kernels::dequantize<int8_t>(X_data[i], X_scale_f, X_zero_point_i32);
      float z = x + y;
      out_data[i] =
          kernels::quantize<int8_t>(z, inv_out_scale, out_zero_point_i32);
    }
  } /* if X is a scalar Tensor */
  else if (X_numel == 1) {
    float x =
        kernels::dequantize<int8_t>(X_data[0], X_scale_f, X_zero_point_i32);
    for (size_t i = 0; i < Y_numel; ++i) {
      float y =
          kernels::dequantize<int8_t>(Y_data[i], Y_scale_f, Y_zero_point_i32);
      float z = x + y;
      out_data[i] =
          kernels::quantize<int8_t>(z, inv_out_scale, out_zero_point_i32);
    }
  } /* other broadcasting cases */
  else {
    /* Broadcasting implementation */
    ssize_t X_dim = X.dim();
    ssize_t Y_dim = Y.dim();
    ssize_t out_dim = out.dim();

    /* Precompute strides for X and Y tensors */
    constexpr size_t max_dim = executorch::runtime::kTensorDimensionLimit;
    size_t X_strides[max_dim] = {0};
    size_t Y_strides[max_dim] = {0};
    size_t X_stride_val = 1;
    size_t Y_stride_val = 1;

    /* Calculate strides from last dimension to first */
    for (int d = out_dim - 1; d >= 0 && d >= out_dim - max_dim; --d) {
      int idx = out_dim - 1 - d; /* Index into the fixed-size array */
      if (d >= out_dim - X_dim) {
        size_t X_d = d - (out_dim - X_dim);
        X_strides[idx] = X_stride_val;
        X_stride_val *= X.size(X_d);
      }

      if (d >= out_dim - Y_dim) {
        size_t Y_d = d - (out_dim - Y_dim);
        Y_strides[idx] = Y_stride_val;
        Y_stride_val *= Y.size(Y_d);
      }
    }

    /* Iterate over output tensor */
    for (ssize_t i = 0; i < out_numel; ++i) {
      size_t out_idx = i;
      size_t X_idx = 0;
      size_t Y_idx = 0;

      /* Compute corresponding indices in input tensors */
      for (int d = out_dim - 1; d >= 0; --d) {
        size_t out_dim_idx = out_idx % out.size(d);
        out_idx /= out.size(d);

        /* Compute X index */
        if (d >= out_dim - X_dim) {
          size_t X_d = d - (out_dim - X_dim);
          size_t X_dim_idx = out_dim_idx % X.size(X_d);
          if (d >= out_dim - max_dim) {
            int idx = out_dim - 1 - d;
            X_idx += X_dim_idx * X_strides[idx];
          } else {
            size_t X_stride = 1;
            for (int k = out_dim - 1; k > d; --k) {
              if (k >= out_dim - X_dim) {
                size_t X_k = k - (out_dim - X_dim);
                X_stride *= X.size(X_k);
              }
            }
            X_idx += X_dim_idx * X_stride;
          }
        }

        /* Compute Y index */
        if (d >= out_dim - Y_dim) {
          size_t Y_d = d - (out_dim - Y_dim);
          size_t Y_dim_idx = out_dim_idx % Y.size(Y_d);
          if (d >= out_dim - max_dim) {
            int idx = out_dim - 1 - d;
            Y_idx += Y_dim_idx * Y_strides[idx];
          } else {
            size_t Y_stride = 1;
            for (int k = out_dim - 1; k > d; --k) {
              if (k >= out_dim - Y_dim) {
                size_t Y_k = k - (out_dim - Y_dim);
                Y_stride *= Y.size(Y_k);
              }
            }
            Y_idx += Y_dim_idx * Y_stride;
          }
        }
      }

      /* Apply the operation */
      float x = kernels::dequantize<int8_t>(
          X_data[X_idx], X_scale_f, X_zero_point_i32);
      float y = kernels::dequantize<int8_t>(
          Y_data[Y_idx], Y_scale_f, Y_zero_point_i32);
      float z = x + y;
      out_data[i] =
          kernels::quantize<int8_t>(z, inv_out_scale, out_zero_point_i32);
    }
  }
}

} // namespace native
} // namespace HiFi
} // namespace impl
