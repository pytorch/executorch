/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/ops/OperatorRegistry.h>
#include <executorch/backends/webgpu/runtime/ops/sigmoid/sigmoid_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/unary/UnaryOp.h>

#include <vector>

namespace executorch::backends::webgpu {

namespace {

void sigmoid_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // aten.sigmoid.default args: [in, out]
  add_unary_op(
      graph,
      args.at(0),
      args.at(1),
      kSigmoidWGSL,
      kSigmoidWorkgroupSizeX,
      "sigmoid");
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.sigmoid.default, sigmoid_impl);
}

} // namespace executorch::backends::webgpu
