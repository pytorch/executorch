/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/ops/OperatorRegistry.h>
#include <executorch/backends/webgpu/runtime/ops/unary/UnaryOp.h>
#include <executorch/backends/webgpu/runtime/ops/unary/abs_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/unary/clamp_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/unary/cos_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/unary/exp_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/unary/hardswish_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/unary/neg_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/unary/round_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/unary/rsqrt_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/unary/sin_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/unary/sqrt_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/unary/tanh_wgsl.h>

#include <limits>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

// Scalar bound arg, or +/-inf when None; mirrors Vulkan get_val_or_inf.
float get_val_or_inf(WebGPUGraph& graph, int id, bool is_max) {
  const auto t = graph.get_value_type(id);
  if (t == WebGPUGraph::ValueType::Int) {
    return static_cast<float>(graph.get_int(id));
  }
  if (t == WebGPUGraph::ValueType::Double) {
    return static_cast<float>(graph.get_double(id));
  }
  if (t != WebGPUGraph::ValueType::Null) {
    throw std::runtime_error("unary bound must be a scalar or None");
  }
  return is_max ? std::numeric_limits<float>::infinity()
                : -std::numeric_limits<float>::infinity();
}

void abs_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  add_unary_op(
      graph, args.at(0), args.at(1), kAbsWGSL, kAbsWorkgroupSizeX, "abs");
}

void exp_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  add_unary_op(
      graph, args.at(0), args.at(1), kExpWGSL, kExpWorkgroupSizeX, "exp");
}

void sqrt_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  add_unary_op(
      graph, args.at(0), args.at(1), kSqrtWGSL, kSqrtWorkgroupSizeX, "sqrt");
}

void rsqrt_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  add_unary_op(
      graph, args.at(0), args.at(1), kRsqrtWGSL, kRsqrtWorkgroupSizeX, "rsqrt");
}

void sin_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  add_unary_op(
      graph, args.at(0), args.at(1), kSinWGSL, kSinWorkgroupSizeX, "sin");
}

void cos_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  add_unary_op(
      graph, args.at(0), args.at(1), kCosWGSL, kCosWorkgroupSizeX, "cos");
}

void tanh_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  add_unary_op(
      graph, args.at(0), args.at(1), kTanhWGSL, kTanhWorkgroupSizeX, "tanh");
}

void round_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  add_unary_op(
      graph, args.at(0), args.at(1), kRoundWGSL, kRoundWorkgroupSizeX, "round");
}

void neg_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  add_unary_op(
      graph, args.at(0), args.at(1), kNegWGSL, kNegWorkgroupSizeX, "neg");
}

void hardswish_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  add_unary_op(
      graph,
      args.at(0),
      args.at(1),
      kHardswishWGSL,
      kHardswishWorkgroupSizeX,
      "hardswish");
}

void clamp_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // aten.clamp.default args: [in, min, max, out]; min/max None -> +/-inf.
  const float lo = get_val_or_inf(graph, args.at(1), /*is_max=*/false);
  const float hi = get_val_or_inf(graph, args.at(2), /*is_max=*/true);
  add_unary_op(
      graph,
      args.at(0),
      args.at(3),
      kClampWGSL,
      kClampWorkgroupSizeX,
      "clamp",
      lo,
      hi);
}

void hardtanh_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // aten.hardtanh.default args: [in, min_val, max_val, out].
  const float lo = get_val_or_inf(graph, args.at(1), /*is_max=*/false);
  const float hi = get_val_or_inf(graph, args.at(2), /*is_max=*/true);
  add_unary_op(
      graph,
      args.at(0),
      args.at(3),
      kClampWGSL,
      kClampWorkgroupSizeX,
      "hardtanh",
      lo,
      hi);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.abs.default, abs_impl);
  WEBGPU_REGISTER_OP(aten.exp.default, exp_impl);
  WEBGPU_REGISTER_OP(aten.sqrt.default, sqrt_impl);
  WEBGPU_REGISTER_OP(aten.rsqrt.default, rsqrt_impl);
  WEBGPU_REGISTER_OP(aten.sin.default, sin_impl);
  WEBGPU_REGISTER_OP(aten.cos.default, cos_impl);
  WEBGPU_REGISTER_OP(aten.tanh.default, tanh_impl);
  WEBGPU_REGISTER_OP(aten.round.default, round_impl);
  WEBGPU_REGISTER_OP(aten.neg.default, neg_impl);
  WEBGPU_REGISTER_OP(aten.hardswish.default, hardswish_impl);
  WEBGPU_REGISTER_OP(aten.clamp.default, clamp_impl);
  WEBGPU_REGISTER_OP(aten.hardtanh.default, hardtanh_impl);
}

} // namespace executorch::backends::webgpu
