// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/DataInitializer.h>

#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace vkcompute::shader_profiling {

using Json = nlohmann::json;

// Expands sweeps and normalizes inline tensor definitions.
std::vector<Json> parse_configs(const Json& document);
std::set<std::string> validation_output_names(const Json& spec);
size_t tensor_numel(const Json& sizes);
uint32_t
choose_dispatch_count(double probe_us, double target_us, uint32_t limit);

} // namespace vkcompute::shader_profiling
