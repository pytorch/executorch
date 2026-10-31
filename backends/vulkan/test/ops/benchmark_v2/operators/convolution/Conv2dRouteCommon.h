// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/RouteRegistry.h>

#include <array>

namespace vkcompute::shader_profiling::convolution {

struct ConvShape {
  std::vector<int64_t> input;
  std::vector<int64_t> output;
  std::vector<int64_t> weight;
  std::array<int64_t, 2> kernel;
  int64_t groups;
};

void supported(bool condition, const std::string& message);
void check_options(const Json& options, bool intermediate_storage = false);
std::array<int64_t, 2>
spatial_parameter(ComputeGraph& graph, ValueRef value, bool positive);
void constant_tensor(
    ComputeGraph& graph,
    ValueRef value,
    const std::string& name);
int64_t product(int64_t a, int64_t b);
int64_t padded4(int64_t value);
void grouped_packing(const ConvShape& shape);
Json selected(const std::string& name, const Json& options = Json::object());
ConvShape conv_shape_base(
    ComputeGraph& graph,
    ValueRef input,
    ValueRef weight,
    ValueRef output,
    int64_t groups,
    size_t weight_rank);
void check_conv_output(
    ComputeGraph& graph,
    const ConvShape& shape,
    ValueRef stride,
    ValueRef padding,
    ValueRef dilation);
} // namespace vkcompute::shader_profiling::convolution
