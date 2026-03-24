#pragma once

#include <executorch/backends/xnnpack/runtime/graph/graph.h>

namespace executorch::backends::xnnpack::plan {

void rewrite_nhwc(graph::Graph& graph);

}
