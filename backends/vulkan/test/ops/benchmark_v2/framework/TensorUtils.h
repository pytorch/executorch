// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Config.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace vkcompute::shader_profiling {

using TensorData = std::map<std::string, std::vector<uint8_t>>;

TensorData prepare_tensor_data(const Json& spec);

vkapi::ScalarType scalar_type(const std::string& name);

vkapi::BufferBindInfo
meta_ubo(ComputeGraph& graph, ValueRef tensor, const std::string& field);

std::map<std::string, ValueRef> create_tensors(
    const Json& spec,
    ComputeGraph& graph,
    const TensorData& host_data,
    bool with_staging = false);

Json actual_tensor_dtypes(
    ComputeGraph& graph,
    const std::map<std::string, ValueRef>& tensors);
Json dispatch_tensor_bindings(ComputeGraph& graph, const ExecuteNode& node);

} // namespace vkcompute::shader_profiling
