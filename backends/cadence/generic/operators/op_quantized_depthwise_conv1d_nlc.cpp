/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_quantized_conv1d_nlc.h>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>

namespace impl {
namespace generic {
namespace native {

using ::executorch::aten::IntArrayRef;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;

// Depthwise conv1d NLC: delegates to the regular conv1d NLC implementation
// which already handles grouped (depthwise) convolution correctly via
// ocpg/icpg decomposition. This operator exists as a separate entry point
// so that depthwise and regular conv1d are cleanly separated at the graph
// level, enabling independent optimization.
::executorch::aten::Tensor& quantized_depthwise_conv1d_nlc_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    int64_t input_zero_point,
    int64_t weight_zero_point,
    double bias_scale,
    double output_scale,
    int64_t output_zero_point,
    int64_t out_multiplier,
    int64_t out_shift,
    Tensor& out) {
  return quantized_conv1d_nlc_per_tensor_out(
      ctx,
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      groups,
      input_zero_point,
      weight_zero_point,
      bias_scale,
      output_scale,
      output_zero_point,
      out_multiplier,
      out_shift,
      out);
}

} // namespace native
} // namespace generic
} // namespace impl
