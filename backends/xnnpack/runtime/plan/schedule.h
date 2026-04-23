#pragma once

#include <executorch/backends/xnnpack/runtime/graph/graph.h>
#include <executorch/backends/xnnpack/runtime/graph/handles.h>

#include <vector>

namespace executorch::backends::xnnpack::plan {

std::vector<graph::NodeHandle> schedule(const graph::Graph& graph);

}
