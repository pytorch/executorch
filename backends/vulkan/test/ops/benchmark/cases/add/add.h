// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/test/ops/benchmark/framework/registry.h>
#include <executorch/backends/vulkan/test/ops/benchmark/framework/utils.h>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace add {

int64_t add_flop_calculator(const TestCase& test_case);

void add_reference_compute(TestCase& test_case);

} // namespace add
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
