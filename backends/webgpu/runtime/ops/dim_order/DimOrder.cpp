/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>
#include <executorch/backends/webgpu/runtime/ops/OperatorRegistry.h>
#include <executorch/backends/webgpu/runtime/ops/view_copy/view_copy.h>

#include <vector>

namespace executorch::backends::webgpu {

namespace {

// _clone_dim_order = numel-preserving flat copy (shared DMA helper).
void clone_dim_order_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  add_flat_copy(graph, args.at(0), args.at(args.size() - 1));
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(
      dim_order_ops._clone_dim_order.default, clone_dim_order_impl);
}

} // namespace executorch::backends::webgpu
