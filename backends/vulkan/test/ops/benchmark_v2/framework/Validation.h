// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Reference.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/TensorUtils.h>

namespace vkcompute::shader_profiling {

void copy_validation_inputs(
    const Json& spec,
    ComputeGraph& graph,
    const std::map<std::string, ValueRef>& tensors,
    const std::map<std::string, std::vector<uint8_t>>& host_data);

std::map<std::string, ValueRef> add_validation_outputs(
    const Json& spec,
    ComputeGraph& graph,
    const std::map<std::string, ValueRef>& tensors,
    const Json& accesses);

Report validate_outputs(
    const Json& spec,
    ComputeGraph& graph,
    const std::map<std::string, ValueRef>& tensors,
    const std::map<std::string, ValueRef>& staging,
    const std::map<std::string, std::vector<uint8_t>>& host_data);

} // namespace vkcompute::shader_profiling
