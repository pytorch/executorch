/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_quantized_max_pool2d.h>

#include <algorithm>
#include <cstdint>
#include <limits>

#include <executorch/backends/cadence/generic/operators/cadence_type_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

namespace impl {
namespace generic {
namespace native {

using ::executorch::aten::IntArrayRef;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;

namespace {

template <typename T>
void quantized_max_pool2d_nchw_impl(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    ET_UNUSED bool ceil_mode,
    Tensor& output) {
  const T* __restrict__ in_data = input.const_data_ptr<T>();
  T* __restrict__ out_data = output.mutable_data_ptr<T>();

  // Input dimensions: [N, C, H, W]
  const int64_t batch_size = input.size(0);
  const int64_t channels = input.size(1);
  const int64_t in_height = input.size(2);
  const int64_t in_width = input.size(3);

  // Output dimensions
  const int64_t out_height = output.size(2);
  const int64_t out_width = output.size(3);

  // Pooling parameters
  const int64_t kernel_h = kernel_size[0];
  const int64_t kernel_w = kernel_size[1];
  const int64_t stride_h = stride[0];
  const int64_t stride_w = stride[1];
  const int64_t pad_h = padding[0];
  const int64_t pad_w = padding[1];
  const int64_t dilation_h = dilation[0];
  const int64_t dilation_w = dilation[1];

  // Iterate over batch and channels
  for (int64_t n = 0; n < batch_size; ++n) {
    for (int64_t c = 0; c < channels; ++c) {
      // Iterate over output spatial dimensions
      for (int64_t oh = 0; oh < out_height; ++oh) {
        for (int64_t ow = 0; ow < out_width; ++ow) {
          // Compute the input region for this output pixel
          const int64_t ih_start = oh * stride_h - pad_h;
          const int64_t iw_start = ow * stride_w - pad_w;

          // Initialize with minimum value for the type
          T max_val = std::numeric_limits<T>::lowest();

          // Iterate over the kernel
          for (int64_t kh = 0; kh < kernel_h; ++kh) {
            for (int64_t kw = 0; kw < kernel_w; ++kw) {
              const int64_t ih = ih_start + kh * dilation_h;
              const int64_t iw = iw_start + kw * dilation_w;

              // Check bounds
              if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                const int64_t in_idx =
                    ((n * channels + c) * in_height + ih) * in_width + iw;
                max_val = std::max(max_val, in_data[in_idx]);
              }
            }
          }

          // Write output
          const int64_t out_idx =
              ((n * channels + c) * out_height + oh) * out_width + ow;
          out_data[out_idx] = max_val;
        }
      }
    }
  }
}

} // namespace

Tensor& quantized_max_pool2d_nchw_out(
    ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    Tensor& output) {
#define typed_quantized_max_pool2d_nchw(ctype, dtype)                      \
  case ScalarType::dtype: {                                                \
    quantized_max_pool2d_nchw_impl<ctype>(                                 \
        input, kernel_size, stride, padding, dilation, ceil_mode, output); \
    break;                                                                 \
  }

  ScalarType dtype = input.scalar_type();
  // NOLINTBEGIN(clang-diagnostic-switch-enum)
  switch (dtype) {
    ET_FORALL_CADENCE_QUANTIZED_TYPES(typed_quantized_max_pool2d_nchw)
    default:
      ET_DCHECK_MSG(
          false, "Unhandled dtype %s", torch::executor::toString(dtype));
  }
    // NOLINTEND(clang-diagnostic-switch-enum)

#undef typed_quantized_max_pool2d_nchw
  return output;
}

} // namespace native
} // namespace generic
} // namespace impl
