// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <vector>

#include <executorch/backends/native/runtime/graph/Graph.h>

namespace ptn::vulkan {

size_t materialize_view_copies(
    Graph& graph,
    const std::vector<ValueId>& value_ids);

// ET-VK views keep the sizes they were built with, and only aten.view gets a
// resize hook, so permute, select, squeeze, unsqueeze and alias of a
// dynamically shaped source run as their copy kernels instead, with negative
// dims made non-negative. Views in an alias group that something mutates stay
// views. Returns the rewritten nodes.
std::vector<NodeId> materialize_dynamic_view_copies(Graph& graph);

} // namespace ptn::vulkan
