/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/binary/BinaryOp.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/unary/UnaryOp.h>

#include <optional>

namespace vkcompute {

enum class Conv2dMethod : uint8_t {
  Depthwise,
  Pointwise,
  SlidingWindow,
  Transposed,
};

/*
 * Element-wise ops applied to the convolution output:
 *
 *   out = activation(residual_op(conv, residual * scale, residual_alpha))
 *
 * `residual` has the output's sizes and `scale` is a constant tensor with one
 * value per output channel at dim -3. Both are optional; `scale` requires a
 * residual. The activation is one of clamp, relu and leaky_relu.
 */
struct Conv2dEpilogue final {
  ValueRef residual = kDummyValueRef;
  ValueRef scale = kDummyValueRef;
  BinaryOpType residual_op = BinaryOpType::kAdd;
  float residual_alpha = 1.0f;
  std::optional<UnaryOpType> activation;
  float activation_a = 0.0f;
  float activation_b = 0.0f;

  bool has_residual() const {
    return is_valid(residual);
  }

  bool has_scale() const {
    return is_valid(scale);
  }

  bool has_activation() const {
    return activation.has_value();
  }
};

// Uniforms read by the epilogue of conv2d shaders.
struct Conv2dEpilogueParams final {
  float activation_a;
  float activation_b;
  float residual_alpha;
};

/*
 * Tensors bound to the residual and scale slots of conv2d shaders. Absent
 * operands are bound to the convolution input and packed bias, which have the
 * slots' storage types; the shaders never read them.
 */
struct Conv2dEpilogueArgs final {
  ValueRef residual;
  ValueRef scale;

  // `packed_scale` is the result of prepack_conv2d_residual_scale().
  Conv2dEpilogueArgs(
      const Conv2dEpilogue& epilogue,
      const ValueRef in,
      const ValueRef packed_bias,
      const ValueRef packed_scale);
};

// Prepacks the epilogue's residual scale like a bias. Returns kDummyValueRef
// if the epilogue has no scale.
ValueRef prepack_conv2d_residual_scale(
    ComputeGraph& graph,
    const Conv2dEpilogue& epilogue,
    const ValueRef weight_data,
    const bool transposed);

// Called by the resize functions of conv2d nodes, which pass the epilogue's
// residual (or kDummyValueRef) as their last resize arg. Checks that the
// residual still has the resized output's sizes.
void check_conv2d_residual_on_resize(
    ComputeGraph* graph,
    const ValueRef out,
    const std::vector<ValueRef>& resize_args);

// With `f16_epilogue`, for shaders that compute the epilogue in f16, the params
// are rounded to the nearest f16 so that the shaders convert them exactly. Some
// GPUs round f32 to f16 conversions toward zero.
Conv2dEpilogueParams create_conv2d_epilogue_params(
    const Conv2dEpilogue& epilogue,
    const bool f16_epilogue);

// Appends the has_activation, activation_op, has_residual, has_scale and
// residual_op specialization constants.
void append_conv2d_epilogue_spec_vars(
    vkapi::SpecVarList& spec_vars,
    const Conv2dEpilogue& epilogue);

void conv2d_pw_impl(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef out,
    const bool transposed_val,
    const Conv2dEpilogue& epilogue);

// `b1x1_override` forces or disables the 1x1 output tile of 3x3 and 5x5
// kernels instead of choosing it by GPU.
void conv2d_dw_impl(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef out,
    const Conv2dEpilogue& epilogue,
    const std::optional<bool> b1x1_override = std::nullopt);

void resize_conv2d_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args);

// Returns the epilogue described by the arguments of aten.convolution (none),
// conv_with_clamp (a clamp) or conv2d.with_epilogue, and checks that its
// operands are supported.
Conv2dEpilogue conv2d_epilogue_from_args(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args);

// Implements aten.convolution, conv_with_clamp and conv2d.with_epilogue for 4D
// inputs.
void conv2d_with_epilogue_impl(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args);

} // namespace vkcompute
