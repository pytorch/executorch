/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_quantized_conv1d_ncl.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using Tensor = executorch::aten::Tensor;
using KernelRuntimeContext = torch::executor::KernelRuntimeContext;
using ::executorch::aten::IntArrayRef;

namespace impl {
namespace HiFi {
namespace native {

// Depthwise conv1d NCL for HiFi: falls back to the generic implementation.
// In practice this op is always converted to the NLC variant by
// ReplaceConvWithChannelLastConvPass before reaching C++ kernels,
// so no NNLib optimization is needed here.
void quantized_depthwise_conv1d_ncl_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    int64_t in_zero_point,
    int64_t weight_zero_point,
    double bias_scale,
    double output_scale,
    int64_t output_zero_point,
    int64_t out_multiplier,
    int64_t out_shift,
    Tensor& out) {
  impl::generic::native::quantized_conv1d_ncl_per_tensor_out(
      ctx,
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      groups,
      in_zero_point,
      weight_zero_point,
      bias_scale,
      output_scale,
      output_zero_point,
      out_multiplier,
      out_shift,
      out);
}

} // namespace native
} // namespace HiFi
} // namespace impl
