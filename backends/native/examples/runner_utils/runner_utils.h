// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace ptn::runner_utils {

std::vector<float> read_floats(const std::string& path);
void write_floats(const std::string& path, std::span<const float> values);
std::vector<std::string> read_lines(const std::string& path);

int64_t numel_of(const std::vector<int64_t>& sizes);
std::string shape_str(const std::vector<int64_t>& sizes);
std::vector<int> top_k(const std::vector<float>& values, int k);

bool can_compare_floats(
    size_t actual_size,
    size_t expected_size,
    double rtol,
    double atol);
bool compare_floats(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    double rtol,
    double atol);

} // namespace ptn::runner_utils
