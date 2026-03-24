#pragma once

#include <executorch/backends/xnnpack/runtime/graph/graph.h>

namespace executorch::backends::xnnpack::plan {

bool check_xnn_node_support(graph::CallOperatorNode& node, graph::Graph& graph);

}
