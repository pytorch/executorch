/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include <executorch/backends/vulkan/runtime/graph/containers/Value.h>
#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

namespace vkcompute {

void conv1d_impl(ComputeGraph& graph, const std::vector<ValueRef>& args);

} // namespace vkcompute
