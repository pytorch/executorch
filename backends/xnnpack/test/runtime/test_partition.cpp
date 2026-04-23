#include <gtest/gtest.h>

#include <executorch/backends/xnnpack/runtime/graph/graph_builder.h>
#include <executorch/backends/xnnpack/runtime/plan/partition.h>

using namespace executorch::backends::xnnpack::core;
using namespace executorch::backends::xnnpack::graph;
using namespace executorch::backends::xnnpack::plan;

TEST(TestPartition, single_node) {
    auto builder = GraphBuilder();

    auto spec = TensorSpec {
        .dtype = DType::Float32,
        .sizes = { DimSizeSpec::constant(1), DimSizeSpec::constant(10) }
    };

    auto input_a = builder.createInput(spec);
    auto input_b = builder.createInput(spec);
    auto add = builder.createOperator(Operator::Add, spec, input_a, input_b);
    builder.createOutput(add);

    auto graph = builder.build();

    // Pre-tag the add node for XNNPACK.
    graph.nodes[add.node].flags |= NodeFlags::UseXnnpack;

    graph.update_users();
    auto partition_count = assign_partitions(graph);

    // There should be exactly one partition.
    EXPECT_EQ(partition_count, 1);

    // The add node should be assigned to partition 1.
    EXPECT_EQ(graph.nodes[add.node].tag, 1);

    // Input nodes should not be assigned to any partition.
    EXPECT_EQ(graph.nodes[input_a.node].tag, 0);
    EXPECT_EQ(graph.nodes[input_b.node].tag, 0);
}

// Helper to set UseXnnpack on a node.
static void set_xnnpack(Graph& graph, ValueHandle handle) {
    graph.nodes[handle.node].flags |= NodeFlags::UseXnnpack;
}

TEST(TestPartition, sequential_all_delegated) {
    // Build a chain: input_a, input_b -> add1 -> add2 -> add3 -> add4 -> output
    auto builder = GraphBuilder();

    auto spec = TensorSpec {
        .dtype = DType::Float32,
        .sizes = { DimSizeSpec::constant(1), DimSizeSpec::constant(10) }
    };

    auto input_a = builder.createInput(spec);
    auto input_b = builder.createInput(spec);
    auto add1 = builder.createOperator(Operator::Add, spec, input_a, input_b);
    auto add2 = builder.createOperator(Operator::Add, spec, add1, input_b);
    auto add3 = builder.createOperator(Operator::Add, spec, add2, input_b);
    auto add4 = builder.createOperator(Operator::Add, spec, add3, input_b);
    builder.createOutput(add4);

    auto graph = builder.build();

    set_xnnpack(graph, add1);
    set_xnnpack(graph, add2);
    set_xnnpack(graph, add3);
    set_xnnpack(graph, add4);

    graph.update_users();
    auto partition_count = assign_partitions(graph);

    // All delegated nodes should be in a single partition.
    EXPECT_EQ(partition_count, 1);

    EXPECT_EQ(graph.nodes[add1.node].tag, 1);
    EXPECT_EQ(graph.nodes[add2.node].tag, 1);
    EXPECT_EQ(graph.nodes[add3.node].tag, 1);
    EXPECT_EQ(graph.nodes[add4.node].tag, 1);

    // Input nodes should not be assigned to any partition.
    EXPECT_EQ(graph.nodes[input_a.node].tag, 0);
    EXPECT_EQ(graph.nodes[input_b.node].tag, 0);
}

TEST(TestPartition, sequential_alternating) {
    // Chain: input_a, input_b -> add1 (D) -> add2 -> add3 (D) -> add4 -> output
    // Delegated nodes are separated by undelegated nodes, so they can't be in
    // the same partition without creating a cycle.
    auto builder = GraphBuilder();

    auto spec = TensorSpec {
        .dtype = DType::Float32,
        .sizes = { DimSizeSpec::constant(1), DimSizeSpec::constant(10) }
    };

    auto input_a = builder.createInput(spec);
    auto input_b = builder.createInput(spec);
    auto add1 = builder.createOperator(Operator::Add, spec, input_a, input_b);
    auto add2 = builder.createOperator(Operator::Add, spec, add1, input_b);
    auto add3 = builder.createOperator(Operator::Add, spec, add2, input_b);
    auto add4 = builder.createOperator(Operator::Add, spec, add3, input_b);
    builder.createOutput(add4);

    auto graph = builder.build();

    set_xnnpack(graph, add1);
    set_xnnpack(graph, add3);

    graph.update_users();
    auto partition_count = assign_partitions(graph);

    EXPECT_EQ(partition_count, 2);

    // Each delegated node should be in a separate partition.
    EXPECT_EQ(graph.nodes[add1.node].tag, 1);
    EXPECT_EQ(graph.nodes[add3.node].tag, 2);

    // Undelegated nodes should not be assigned.
    EXPECT_EQ(graph.nodes[add2.node].tag, 0);
    EXPECT_EQ(graph.nodes[add4.node].tag, 0);
    EXPECT_EQ(graph.nodes[input_a.node].tag, 0);
    EXPECT_EQ(graph.nodes[input_b.node].tag, 0);
}

TEST(TestPartition, diamond_skip_connection) {
    auto builder = GraphBuilder();

    auto spec = TensorSpec {
        .dtype = DType::Float32,
        .sizes = { DimSizeSpec::constant(1), DimSizeSpec::constant(10) }
    };

    auto input_a = builder.createInput(spec);
    auto input_b = builder.createInput(spec);
    auto add1 = builder.createOperator(Operator::Add, spec, input_a, input_b);
    auto add2 = builder.createOperator(Operator::Add, spec, add1, input_b);
    auto add3 = builder.createOperator(Operator::Add, spec, add1, input_b);
    auto add4 = builder.createOperator(Operator::Add, spec, add2, input_b);
    auto add5 = builder.createOperator(Operator::Add, spec, add3, input_b);
    auto add6 = builder.createOperator(Operator::Add, spec, add4, add5);
    builder.createOutput(add6);

    auto graph = builder.build();

    set_xnnpack(graph, add1);
    // add2 is NOT delegated.
    set_xnnpack(graph, add3);
    set_xnnpack(graph, add4);
    set_xnnpack(graph, add5);
    set_xnnpack(graph, add6);

    graph.update_users();
    auto partition_count = assign_partitions(graph);

    EXPECT_EQ(partition_count, 2);

    // Partition 1: add1, add3, add5 (connected through delegated nodes only).
    EXPECT_EQ(graph.nodes[add1.node].tag, 1);
    EXPECT_EQ(graph.nodes[add3.node].tag, 1);
    EXPECT_EQ(graph.nodes[add5.node].tag, 1);

    // Partition 2: add4, add6 (add4 blocked from partition 1; add6 depends on add4).
    EXPECT_EQ(graph.nodes[add4.node].tag, 2);
    EXPECT_EQ(graph.nodes[add6.node].tag, 2);

    // Non-delegated / inputs unassigned.
    EXPECT_EQ(graph.nodes[add2.node].tag, 0);
    EXPECT_EQ(graph.nodes[input_a.node].tag, 0);
    EXPECT_EQ(graph.nodes[input_b.node].tag, 0);
}

TEST(TestPartition, converging_delegated) {
    auto builder = GraphBuilder();

    auto spec = TensorSpec {
        .dtype = DType::Float32,
        .sizes = { DimSizeSpec::constant(1), DimSizeSpec::constant(10) }
    };

    auto input_a = builder.createInput(spec);
    auto input_b = builder.createInput(spec);
    auto add1 = builder.createOperator(Operator::Add, spec, input_a, input_a);
    auto add2 = builder.createOperator(Operator::Add, spec, input_b, input_b);
    auto add3 = builder.createOperator(Operator::Add, spec, add1, input_a);
    auto add_sink = builder.createOperator(Operator::Add, spec, add2, input_b);
    auto add5 = builder.createOperator(Operator::Add, spec, add3, input_a);
    auto add6 = builder.createOperator(Operator::Add, spec, add5, add2);
    builder.createOutput(add6);
    builder.createOutput(add_sink);

    auto graph = builder.build();

    set_xnnpack(graph, add1);
    set_xnnpack(graph, add2);
    set_xnnpack(graph, add3);
    // add_sink is NOT delegated — makes add2 escape.
    set_xnnpack(graph, add5);
    set_xnnpack(graph, add6);

    graph.update_users();
    auto partition_count = assign_partitions(graph);

    // All delegated nodes should be in one partition.
    EXPECT_EQ(partition_count, 1);
    EXPECT_EQ(graph.nodes[add1.node].tag, 1);
    EXPECT_EQ(graph.nodes[add2.node].tag, 1);
    EXPECT_EQ(graph.nodes[add3.node].tag, 1);
    EXPECT_EQ(graph.nodes[add5.node].tag, 1);
    EXPECT_EQ(graph.nodes[add6.node].tag, 1);
    EXPECT_EQ(graph.nodes[add_sink.node].tag, 0);
}

// --- fuse_partitions tests ---

TEST(TestFusePartitions, single_node) {
    auto builder = GraphBuilder();

    auto spec = TensorSpec {
        .dtype = DType::Float32,
        .sizes = { DimSizeSpec::constant(1), DimSizeSpec::constant(10) }
    };

    auto input_a = builder.createInput(spec);
    auto input_b = builder.createInput(spec);
    auto add = builder.createOperator(Operator::Add, spec, input_a, input_b);
    builder.createOutput(add);

    auto graph = builder.build();

    graph.nodes[add.node].flags |= NodeFlags::UseXnnpack;

    graph.update_users();
    auto partition_count = assign_partitions(graph);

    EXPECT_EQ(partition_count, 1);

    fuse_partitions(graph, partition_count);

    // The add node should have been replaced with a CallSubgraphNode.
    EXPECT_TRUE(std::holds_alternative<CallSubgraphNode>(graph.nodes[add.node].value));

    // The subgraph should contain the original operator.
    auto& subgraph_node = std::get<CallSubgraphNode>(graph.nodes[add.node].value);
    EXPECT_NE(subgraph_node.subgraph, nullptr);

    // The fused node's args should be the original inputs.
    EXPECT_EQ(subgraph_node.args.size(), 2);

    // Graph outputs should still reference the fused node.
    EXPECT_EQ(graph.outputs.size(), 1);
    EXPECT_EQ(graph.outputs[0].node, add.node);
}

TEST(TestFusePartitions, parallel_multi_output) {
    auto builder = GraphBuilder();

    auto spec = TensorSpec {
        .dtype = DType::Float32,
        .sizes = { DimSizeSpec::constant(1), DimSizeSpec::constant(10) }
    };

    auto input_a = builder.createInput(spec);
    auto input_b = builder.createInput(spec);
    auto add1 = builder.createOperator(Operator::Add, spec, input_a, input_b);
    auto add2 = builder.createOperator(Operator::Add, spec, input_a, input_b);
    builder.createOutput(add1);
    builder.createOutput(add2);

    auto graph = builder.build();

    set_xnnpack(graph, add1);
    set_xnnpack(graph, add2);

    graph.update_users();
    auto partition_count = assign_partitions(graph);

    EXPECT_EQ(partition_count, 1);
    EXPECT_EQ(graph.nodes[add1.node].tag, 1);
    EXPECT_EQ(graph.nodes[add2.node].tag, 1);

    fuse_partitions(graph, partition_count);

    // The anchor (add2) should be a CallSubgraphNode.
    EXPECT_TRUE(std::holds_alternative<CallSubgraphNode>(graph.nodes[add2.node].value));
    auto& fused = std::get<CallSubgraphNode>(graph.nodes[add2.node].value);
    EXPECT_NE(fused.subgraph, nullptr);

    // The subgraph should have two outputs.
    EXPECT_EQ(fused.subgraph->outputs.size(), 2);

    // The fused node should have a multi-output spec.
    EXPECT_TRUE(std::holds_alternative<std::vector<TensorSpec>>(fused.output_specs));
    EXPECT_EQ(std::get<std::vector<TensorSpec>>(fused.output_specs).size(), 2);

    // graph.outputs[0] (was add1) should now reference the anchor with output 0.
    EXPECT_EQ(graph.outputs[0].node, add2.node);
    EXPECT_EQ(graph.outputs[0].output, 0);

    // graph.outputs[1] (was add2/anchor) keeps its original reference.
    EXPECT_EQ(graph.outputs[1].node, add2.node);

    // --- Compaction ---
    // Before compaction, add1 is a Dead tombstone.
    EXPECT_EQ((graph.nodes[add1.node].flags & NodeFlags::Dead), NodeFlags::Dead);

    size_t pre_compact_size = graph.nodes.size();
    graph.compact_nodes();

    // Dead nodes should be removed.
    EXPECT_LT(graph.nodes.size(), pre_compact_size);

    // Graph outputs should still be valid and reference the fused node.
    EXPECT_EQ(graph.outputs.size(), 2);

    auto& fused_after = std::get<CallSubgraphNode>(graph.nodes[graph.outputs[0].node].value);
    EXPECT_NE(fused_after.subgraph, nullptr);
    EXPECT_EQ(graph.outputs[0].node, graph.outputs[1].node);
}

TEST(TestCompactNodes, sequential_chain) {
    auto builder = GraphBuilder();

    auto spec = TensorSpec {
        .dtype = DType::Float32,
        .sizes = { DimSizeSpec::constant(1), DimSizeSpec::constant(10) }
    };

    auto input_a = builder.createInput(spec);
    auto input_b = builder.createInput(spec);
    auto add1 = builder.createOperator(Operator::Add, spec, input_a, input_b);
    auto add2 = builder.createOperator(Operator::Add, spec, add1, input_b);
    auto add3 = builder.createOperator(Operator::Add, spec, add2, input_b);
    builder.createOutput(add3);

    auto graph = builder.build();
    size_t original_size = graph.nodes.size();

    set_xnnpack(graph, add1);
    set_xnnpack(graph, add2);
    set_xnnpack(graph, add3);

    graph.update_users();
    auto partition_count = assign_partitions(graph);
    EXPECT_EQ(partition_count, 1);

    fuse_partitions(graph, partition_count);

    // add1, add2 are tombstoned; add3 (anchor) is the CallSubgraphNode.
    EXPECT_EQ((graph.nodes[add1.node].flags & NodeFlags::Dead), NodeFlags::Dead);
    EXPECT_EQ((graph.nodes[add2.node].flags & NodeFlags::Dead), NodeFlags::Dead);

    graph.compact_nodes();

    // Two dead nodes removed: inputs (2) + anchor (1) = 3 live nodes.
    EXPECT_EQ(graph.nodes.size(), original_size - 2);

    // No Dead nodes remain.
    for (size_t i = 0; i < graph.nodes.size(); i++) {
        EXPECT_EQ((graph.nodes[i].flags & NodeFlags::Dead), NodeFlags::None)
            << "Node " << i << " is still dead after compaction";
    }

    // Graph output should point to a valid CallSubgraphNode.
    EXPECT_EQ(graph.outputs.size(), 1);
    auto& out_node = graph.nodes[graph.outputs[0].node];
    EXPECT_TRUE(std::holds_alternative<CallSubgraphNode>(out_node.value));

    // The fused node's args should reference valid input nodes.
    auto& fused = std::get<CallSubgraphNode>(out_node.value);
    for (auto& arg : fused.args) {
        EXPECT_LT(arg.node, graph.nodes.size());
        EXPECT_TRUE(std::holds_alternative<InputNode>(graph.nodes[arg.node].value));
    }
}
