// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Reference.h>

namespace vkcompute::shader_profiling::reference {

Tensor
matmul_reference(const Json& spec, const TensorMap& tensors, size_t threads);

} // namespace vkcompute::shader_profiling::reference
