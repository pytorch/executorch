// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Report.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/TensorUtils.h>
#include <map>

namespace vkcompute::shader_profiling {

std::vector<ValueRef> build_operator_args(
    const Json& spec,
    ComputeGraph& graph,
    const std::map<std::string, ValueRef>& tensors);

Report benchmark_operator(
    const Json& spec,
    const TensorData& host_data,
    bool validation_only = false);

// Case check that builds `parameters.operator_case` once, then resizes its
// staged inputs per `validation_steps` entry and validates the output.
Json validate_operator_dynamic_resize(const Json& spec, ComputeGraph& graph);

} // namespace vkcompute::shader_profiling
