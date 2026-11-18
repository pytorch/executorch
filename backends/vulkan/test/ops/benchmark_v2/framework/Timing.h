// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Report.h>

#include <cstdint>
#include <functional>

namespace vkcompute::shader_profiling {

Report measure(
    const Json& spec,
    ComputeGraph& graph,
    const std::function<void(uint32_t)>& encode_batch,
    const Json& work_spec = Json::object());

} // namespace vkcompute::shader_profiling
