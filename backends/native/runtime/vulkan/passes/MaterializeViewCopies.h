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

} // namespace ptn::vulkan
