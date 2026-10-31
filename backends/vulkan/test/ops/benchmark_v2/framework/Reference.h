// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Report.h>

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace vkcompute::shader_profiling {

using ReferenceOutputs = std::map<std::string, std::vector<double>>;

namespace reference {

struct Tensor {
  std::vector<int64_t> sizes;
  std::vector<double> values;
};

using TensorMap = std::map<std::string, Tensor>;

void parallel_for(
    size_t count,
    size_t threads,
    const std::function<void(size_t)>& function);
const Tensor&
tensor_arg(const Json& spec, size_t index, const TensorMap& tensors);
void require(bool condition, const std::string& message);

} // namespace reference

std::string reference_name(const Json& spec);
ReferenceOutputs reference_outputs(
    const Json& spec,
    const Json& actual_dtypes,
    const std::map<std::string, std::vector<uint8_t>>& host_data,
    size_t threads = 0);
Report compare_outputs(
    const Json& spec,
    const ReferenceOutputs& expected,
    const ReferenceOutputs& actual,
    const Json& actual_dtypes);

} // namespace vkcompute::shader_profiling
