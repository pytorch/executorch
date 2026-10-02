// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/native/runtime/graph/Graph.h>

namespace ptn {

// Orders active nodes by their data dependencies. Ties retain schedule order.
void stable_topological_sort(Graph& graph);

// Checks ids, def-use, aliases, active-node uniqueness, and topological order.
// Recurses into subgraphs. Throws std::runtime_error on the first violation.
void validate_graph(const Graph& graph);

} // namespace ptn
