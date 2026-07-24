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
#include <executorch/backends/webgpu/runtime/ops/unary/cos_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/unary/exp_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/unary/hardswish_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/unary/neg_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/unary/round_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/unary/rsqrt_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/unary/sin_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/unary/sqrt_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/unary/tanh_wgsl.h>

#include <vector>

namespace executorch::backends::webgpu {

namespace {

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
}

} // namespace executorch::backends::webgpu
