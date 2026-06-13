#pragma once

#include <executorch/backends/xnnpack/runtime/graph/graph.h>
#include <executorch/backends/xnnpack/runtime/graph/handles.h>

#include <vector>

namespace executorch::backends::xnnpack::plan {

/*
 * Flatten a computational graph down to a linear schedule. This is
 * an ordering of nodes that respects dependency orders - i.e. a
 * topological sort.
 */
std::vector<graph::NodeHandle> schedule(const graph::Graph& graph);

} // namespace executorch::backends::xnnpack::plan
