/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_quantized_max_pool2d_nhwc.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
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
void quantized_max_pool2d_nhwc_impl(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    ET_UNUSED bool ceil_mode,
    Tensor& output) {
  const T* __restrict__ in_data = input.const_data_ptr<T>();
  T* __restrict__ out_data = output.mutable_data_ptr<T>();

  // Input dimensions: [N, H, W, C]
  const int64_t batch_size = input.size(0);
  const int64_t in_height = input.size(1);
  const int64_t in_width = input.size(2);
  const int64_t channels = input.size(3);

  // Output dimensions: [N, H_out, W_out, C]
  const int64_t out_height = output.size(1);
  const int64_t out_width = output.size(2);

  // Pooling parameters
  const int64_t kernel_h = kernel_size[0];
  const int64_t kernel_w = kernel_size[1];
  const int64_t stride_h = stride[0];
  const int64_t stride_w = stride[1];
  const int64_t pad_h = padding[0];
  const int64_t pad_w = padding[1];
  const int64_t dilation_h = dilation[0];
  const int64_t dilation_w = dilation[1];

  for (int64_t n = 0; n < batch_size; ++n) {
    for (int64_t oh = 0; oh < out_height; ++oh) {
      for (int64_t ow = 0; ow < out_width; ++ow) {
        const int64_t ih_start = oh * stride_h - pad_h;
        const int64_t iw_start = ow * stride_w - pad_w;

        T* __restrict__ out_ptr =
            out_data + ((n * out_height + oh) * out_width + ow) * channels;

        // Initialize all channels to the minimum value.
        for (int64_t c = 0; c < channels; ++c) {
          out_ptr[c] = std::numeric_limits<T>::lowest();
        }

        // For each kernel position, compute element-wise max across all
        // channels. The inner loop over channels is a stride-1 contiguous
        // access in NHWC layout, enabling SIMD auto-vectorization.
        for (int64_t kh = 0; kh < kernel_h; ++kh) {
          const int64_t ih = ih_start + kh * dilation_h;
          if (ih < 0 || ih >= in_height) {
            continue;
          }
          for (int64_t kw = 0; kw < kernel_w; ++kw) {
            const int64_t iw = iw_start + kw * dilation_w;
            if (iw < 0 || iw >= in_width) {
              continue;
            }

            const T* __restrict__ in_ptr =
                in_data + ((n * in_height + ih) * in_width + iw) * channels;

            for (int64_t c = 0; c < channels; ++c) {
              out_ptr[c] = std::max(out_ptr[c], in_ptr[c]);
            }
          }
        }
      }
    }
  }
}

} // namespace

Tensor& quantized_max_pool2d_nhwc_out(
    ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    Tensor& output) {
#define typed_quantized_max_pool2d_nhwc(ctype, dtype)                      \
  case ScalarType::dtype: {                                                \
    quantized_max_pool2d_nhwc_impl<ctype>(                                 \
        input, kernel_size, stride, padding, dilation, ceil_mode, output); \
    break;                                                                 \
  }

  ScalarType dtype = input.scalar_type();
  // NOLINTBEGIN(clang-diagnostic-switch-enum)
  switch (dtype) {
    ET_FORALL_CADENCE_QUANTIZED_TYPES(typed_quantized_max_pool2d_nhwc)
    default:
      ET_DCHECK_MSG(
          false, "Unhandled dtype %s", torch::executor::toString(dtype));
  }
    // NOLINTEND(clang-diagnostic-switch-enum)

#undef typed_quantized_max_pool2d_nhwc
  return output;
}

} // namespace native
} // namespace generic
} // namespace impl
