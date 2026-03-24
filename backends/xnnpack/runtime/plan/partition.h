#pragma once

#include <executorch/backends/xnnpack/runtime/graph/graph.h>

namespace executorch::backends::xnnpack::plan {

void partition_xnn_subgraphs(graph::Graph& graph);
uint16_t assign_partitions(graph::Graph& graph);
void fuse_partitions(graph::Graph& graph, uint16_t partition_count);

}
