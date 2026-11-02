// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Config.h>

namespace vkcompute::shader_profiling {
Json build_conv2d_direct(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args,
    const Json& options);
Json build_conv2d_im2col(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args,
    const Json& options);
Json inspect_conv2d_route(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args,
    size_t first,
    size_t end);
} // namespace vkcompute::shader_profiling
