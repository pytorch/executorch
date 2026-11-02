// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace vkcompute::shader_profiling {

void validate_data_initializer(const nlohmann::json& spec, size_t count);
std::vector<uint8_t>
make_data(const nlohmann::json& spec, size_t count, uint32_t seed = 0);
std::map<std::string, std::vector<uint8_t>> make_tensor_data(
    const nlohmann::json& tensors);

} // namespace vkcompute::shader_profiling
