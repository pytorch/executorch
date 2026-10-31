// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Config.h>

namespace vkcompute::shader_profiling {

std::vector<Json> expand_operator_sweep(const Json& definition);
std::vector<Json> expand_shader_sweep(const Json& definition);

} // namespace vkcompute::shader_profiling
