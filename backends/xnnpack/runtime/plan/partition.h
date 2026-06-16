#pragma once

#include <executorch/backends/xnnpack/runtime/graph/graph.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

namespace executorch::backends::xnnpack::plan {

/*
 * Partitions the graph into XNNPACK-delegated subgraphs: tags the nodes
 * XNNPACK can run, groups them into partitions, and fuses each partition into
 * a single subgraph node.
 */
runtime::Error partition_xnn_subgraphs(graph::Graph& graph);

/*
 * Groups the tagged XNNPACK nodes into partitions, recording each node's
 * partition in its tag, and returns the number of partitions. This is
 * primarily an internal step in partition_xnn_subgraphs.
 */
runtime::Result<uint32_t> assign_partitions(graph::Graph& graph);

/*
 * Replaces the nodes of each of the `partition_count` partitions with a single
 * fused subgraph node. This is primarily an internal step in the partitioning
 * process.
 */
runtime::Error fuse_partitions(graph::Graph& graph, uint32_t partition_count);

} // namespace executorch::backends::xnnpack::plan
