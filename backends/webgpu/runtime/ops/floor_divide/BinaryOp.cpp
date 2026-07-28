/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>
#include <executorch/backends/webgpu/runtime/ops/OperatorRegistry.h>
#include <executorch/backends/webgpu/runtime/ops/binary/BinaryOp.h>
#include <executorch/backends/webgpu/runtime/ops/floor_divide/binary_floor_divide_wgsl.h>

#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

// aten.div.Tensor_mode -> floor(a/b), with NumPy broadcasting (mirrors mul +
// Vulkan). Only rounding_mode='floor' is supported.
void floor_divide_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // args: [in1, in2, rounding_mode(str), out]
  const int mode_id = args.at(2);
  if (graph.get_value_type(mode_id) != WebGPUGraph::ValueType::String ||
      graph.get_string(mode_id) != "floor") {
    throw std::runtime_error("floor_divide: only rounding_mode='floor'");
  }
  add_binary_broadcast_op(
      graph,
      args.at(0),
      args.at(1),
      args.at(args.size() - 1),
      kBinaryFloorDivideWGSL,
      kBinaryFloorDivideWorkgroupSizeX,
      "floor_divide");
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.div.Tensor_mode, floor_divide_impl);
}

} // namespace executorch::backends::webgpu
