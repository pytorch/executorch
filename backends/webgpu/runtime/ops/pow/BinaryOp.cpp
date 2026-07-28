/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/ops/OperatorRegistry.h>
#include <executorch/backends/webgpu/runtime/ops/binary/BinaryOp.h>
#include <executorch/backends/webgpu/runtime/ops/pow/binary_pow_wgsl.h>

#include <vector>

namespace executorch::backends::webgpu {

namespace {

// aten.pow.Tensor_Tensor -> pow(in1, in2), with NumPy broadcasting (mirrors mul
// + Vulkan). WGSL pow(x,y) is undefined for x<0 (exact Vulkan-GLSL parity).
void pow_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  add_binary_broadcast_op(
      graph,
      args.at(0),
      args.at(1),
      args.at(2),
      kBinaryPowWGSL,
      kBinaryPowWorkgroupSizeX,
      "pow");
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.pow.Tensor_Tensor, pow_impl);
}

} // namespace executorch::backends::webgpu
