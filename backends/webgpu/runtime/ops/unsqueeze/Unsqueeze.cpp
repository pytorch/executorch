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

// unsqueeze_copy = numel-preserving flat copy (Vulkan Unsqueeze.cpp:101-103).
void unsqueeze_copy_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // args: [self, dim, out]; dim ignored (out shape fixed AOT, like view_copy).
  add_flat_copy(graph, args.at(0), args.at(args.size() - 1));
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.unsqueeze_copy.default, unsqueeze_copy_impl);
}

} // namespace executorch::backends::webgpu
