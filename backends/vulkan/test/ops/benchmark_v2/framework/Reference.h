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
// Whether the validated device supports packed int8 dot products. Operators
// with quantized activations fall back to weight-only quantization without
// them. Defaults to true when the spec has no reference context.
bool int8_dot_product(const Json& spec);

} // namespace reference

std::string reference_name(const Json& spec);
ReferenceOutputs reference_outputs(
    const Json& spec,
    const Json& actual_dtypes,
    const std::map<std::string, std::vector<uint8_t>>& host_data,
    size_t threads = 0);
// `fault_rel` scales every actual value by 1 + fault_rel before the comparison,
// so a correctness gate can be shown to fail on a known error.
Report compare_outputs(
    const Json& spec,
    const ReferenceOutputs& expected,
    const ReferenceOutputs& actual,
    const Json& actual_dtypes,
    double fault_rel = 0);

} // namespace vkcompute::shader_profiling
