/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_avg_pool2d.h>

#include <algorithm>
#include <cmath>

#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

namespace impl {
namespace generic {
namespace native {

using ::executorch::aten::IntArrayRef;
using ::executorch::aten::optional;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::getLeadingDims;
using ::executorch::runtime::KernelRuntimeContext;

// Compute the avg_pool2d for in_data in NCHW layout. IT is the input datatype,
// and AT is the accumulation datatype. 'quantized' is true when the input is
// quantized tensor.
template <typename IT, typename AT = IT, bool quantized = false>
void avg_pool2d_nchw(
    const IT* __restrict__ in_data,
    const int32_t in_zero_point,
    IT* __restrict__ out_data,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool count_include_pad,
    int64_t divisor,
    int leading_dims,
    int ih,
    int iw,
    int oh,
    int ow) {
  int kh = kernel_size[0];
  int kw = kernel_size[1];
  int s0 = stride[0];
  int s1 = stride[1];
  int p0 = padding[0];
  int p1 = padding[1];

  for (int _n = 0; _n < leading_dims; ++_n) {
    for (int _ih = 0, _oh = 0; _oh < oh; ++_oh, _ih += s0) {
      int input_offset = _n * ih * iw;
      int output_offset = _n * oh * ow + _oh * ow;
      for (int _iw = 0, _ow = 0; _ow < ow; ++_ow, _iw += s1) {
        int kh_lo = std::max(0, _ih - p0);
        int kh_hi = std::min(ih, _ih + kh - p0);
        int kw_lo = std::max(0, _iw - p1);
        int kw_hi = std::min(iw, _iw + kw - p1);
        // Count the number of contributions sans padding
        int count = (kh_hi - kh_lo) * (kw_hi - kw_lo);
        // Set the accumulator
        AT acc = count_include_pad ? in_zero_point * (kh * kw - count) : 0;
        // Accumulate values
        for (int _kh = kh_lo; _kh < kh_hi; ++_kh) {
          for (int _kw = kw_lo; _kw < kw_hi; ++_kw) {
            int input_addr = input_offset + _kh * iw + _kw;
            acc += in_data[input_addr];
          }
        }
        // The divisor changes depending on whether the count includes
        // padded cells or not.
        float inv_divisor = 1. / (count_include_pad ? divisor : count);
        float val = acc * inv_divisor;
        if (quantized) {
          int32_t min_val =
              static_cast<int32_t>(std::numeric_limits<IT>::min());
          int32_t max_val =
              static_cast<int32_t>(std::numeric_limits<IT>::max());
          out_data[output_offset + _ow] = std::min(
              std::max(int32_t(std::nearbyint(val)), min_val), max_val);
        } else {
          out_data[output_offset + _ow] = val;
        }
      }
    }
  }
}

Tensor& avg_pool2d_out(
    ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    optional<int64_t> divisor_override,
    const optional<Tensor>& in_zero_point_t,
    bool channel_last,
    Tensor& out) {
  ET_DCHECK_MSG(!channel_last, "NHWC layout for avg_pool2d not yet supported");
  const int32_t in_zero_point = in_zero_point_t.has_value()
      ? in_zero_point_t.value().const_data_ptr<int32_t>()[0]
      : 0;
  const int64_t divisor = divisor_override.has_value()
      ? divisor_override.value()
      : kernel_size[0] * kernel_size[1];

  const int odim = out.dim();
  const int on = getLeadingDims(out, odim - 2);
  const int oh = out.size(odim - 2);
  const int ow = out.size(odim - 1);
  const int ih = input.size(odim - 2);
  const int iw = input.size(odim - 1);

  // We generate the kernel for float and uint8_t types. The operator also
  // works for double, but does not support other dtypes.
#define typed_avg_pool2d(btype, ctype, quantized, dtype) \
  case ScalarType::dtype: {                              \
    avg_pool2d_nchw<btype, ctype, quantized>(            \
        input.const_data_ptr<btype>(),                   \
        in_zero_point,                                   \
        out.mutable_data_ptr<btype>(),                   \
        kernel_size,                                     \
        stride,                                          \
        padding,                                         \
        count_include_pad,                               \
        divisor,                                         \
        on,                                              \
        ih,                                              \
        iw,                                              \
        oh,                                              \
        ow);                                             \
    break;                                               \
  }

  ScalarType dtype = input.scalar_type();
  switch (dtype) {
    typed_avg_pool2d(float, float, false, Float);
    typed_avg_pool2d(uint8_t, int32_t, true, Byte);
    default:
      ET_DCHECK_MSG(
          false,
          "avg_pool2d not implemented for dtype %s",
          torch::executor::toString(dtype));
  }

  return out;
}

} // namespace native
} // namespace generic
} // namespace impl
