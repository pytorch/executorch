#pragma once

#include <executorch/backends/xnnpack/runtime/graph/graph.h>

namespace executorch::backends::xnnpack::plan {

// Returns true if XNNPACK can run the given operator node.
bool check_xnn_node_support(
    const graph::CallOperatorNode& node,
    const graph::Graph& graph);

// Returns true if we have a preferred in-tree kernel for this node, meaning it
// should not be delegated to XNNPACK even when XNNPACK supports it.
bool prefer_in_tree_kernel(
    const graph::CallOperatorNode& node,
    const graph::Graph& graph);

} // namespace executorch::backends::xnnpack::plan
