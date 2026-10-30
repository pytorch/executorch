// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ptn::runner_utils {

std::vector<float> read_floats(const std::string& path);
std::vector<std::string> read_lines(const std::string& path);

int64_t numel_of(const std::vector<int64_t>& sizes);
std::string shape_str(const std::vector<int64_t>& sizes);
std::vector<int> top_k(const std::vector<float>& values, int k);

bool compare_floats(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    double rtol,
    double atol);

} // namespace ptn::runner_utils
