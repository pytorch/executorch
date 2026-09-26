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

// squeeze_copy.dims = numel-preserving flat copy (Vulkan Squeeze.cpp:102-104).
void squeeze_copy_dims_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // args: [self, dims, out]; dims ignored (out shape fixed AOT).
  add_flat_copy(graph, args.at(0), args.at(args.size() - 1));
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.squeeze_copy.dims, squeeze_copy_dims_impl);
}

} // namespace executorch::backends::webgpu
