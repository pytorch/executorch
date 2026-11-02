// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Config.h>

#include <map>
#include <string>

namespace vkcompute::shader_profiling {

struct WorkgroupSizes {
  GlobalWorkGrid gwg;
  LocalWorkGroup lwg;
};

WorkgroupSizes pick_workgroup_sizes(
    const Json& spec,
    ComputeGraph& graph,
    const vkapi::ShaderInfo& shader,
    const std::map<std::string, ValueRef>& tensors);

} // namespace vkcompute::shader_profiling
