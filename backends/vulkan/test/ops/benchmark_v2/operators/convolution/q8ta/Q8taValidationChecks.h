// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Config.h>
namespace vkcompute::shader_profiling {
Json check_q8ta_unsigned_dot_selection(const Json& spec, ComputeGraph& graph);
Json check_q8ta_streaming_dynamic_shrink(const Json& spec, ComputeGraph& graph);
} // namespace vkcompute::shader_profiling
