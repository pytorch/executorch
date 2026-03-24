#include <gtest/gtest.h>

#include <executorch/backends/xnnpack/runtime/graph/graph_builder.h>
#include <executorch/backends/xnnpack/runtime/plan/partition.h>
#include <executorch/backends/xnnpack/runtime/plan/schedule.h>

#include <algorithm>
#include <unordered_set>

using namespace executorch::backends::xnnpack::core;
using namespace executorch::backends::xnnpack::graph;
using namespace executorch::backends::xnnpack::plan;

// Helper: check that the order is a valid topological sort of the graph.
static void assert_topological(const Graph& graph, const std::vector<NodeHandle>& order) {
    ASSERT_EQ(order.size(), graph.nodes.size());

    // Build position map: node -> index in order.
    std::vector<uint32_t> pos(graph.nodes.size());
    for (uint32_t i = 0; i < order.size(); i++) {
        pos[order[i]] = i;
    }

    // Every arg source must appear before the node that uses it.
    for (NodeHandle n = 0; n < graph.nodes.size(); n++) {
        for (auto& arg : graph.nodes[n].get_args()) {
            EXPECT_LT(pos[arg.node], pos[n])
                << "Node " << arg.node << " (arg) should appear before node " << n;
        }
    }
}

// Helper: check that all nodes appear exactly once.
static void assert_all_nodes_present(const Graph& graph, const std::vector<NodeHandle>& order) {
    ASSERT_EQ(order.size(), graph.nodes.size());

    std::unordered_set<NodeHandle> seen(order.begin(), order.end());
    EXPECT_EQ(seen.size(), graph.nodes.size());
}

static TensorSpec make_spec() {
    return TensorSpec {
        .dtype = DType::Float32,
        .sizes = { DimSizeSpec::constant(1), DimSizeSpec::constant(10) }
    };
}

TEST(TestSchedule, linear_chain) {
    // input -> op1 -> op2 -> output
    auto builder = GraphBuilder();
    auto spec = make_spec();

    auto input = builder.createInput(spec);
    auto op1 = builder.createOperator(Operator::Add, spec, input, input);
    auto op2 = builder.createOperator(Operator::Add, spec, op1, input);
    builder.createOutput(op2);

    auto graph = builder.build();
    graph.update_users();

    auto order = schedule(graph);

    assert_all_nodes_present(graph, order);
    assert_topological(graph, order);

    // Input must be first, then op1, then op2.
    EXPECT_EQ(order[0], input.node);
    EXPECT_EQ(order[1], op1.node);
    EXPECT_EQ(order[2], op2.node);
}

TEST(TestSchedule, diamond) {
    // input_a, input_b -> add1; input_b -> add2; add1, add2 -> add3
    auto builder = GraphBuilder();
    auto spec = make_spec();

    auto input_a = builder.createInput(spec);
    auto input_b = builder.createInput(spec);
    auto add1 = builder.createOperator(Operator::Add, spec, input_a, input_b);
    auto add2 = builder.createOperator(Operator::Add, spec, add1, input_b);
    builder.createOutput(add2);

    auto graph = builder.build();
    graph.update_users();

    auto order = schedule(graph);

    assert_all_nodes_present(graph, order);
    assert_topological(graph, order);

    // Both inputs before add1, add1 before add2.
    std::vector<uint32_t> pos(graph.nodes.size());
    for (uint32_t i = 0; i < order.size(); i++) {
        pos[order[i]] = i;
    }
    EXPECT_LT(pos[input_a.node], pos[add1.node]);
    EXPECT_LT(pos[input_b.node], pos[add1.node]);
    EXPECT_LT(pos[add1.node], pos[add2.node]);
}

TEST(TestSchedule, post_fusion) {
    auto builder = GraphBuilder();
    auto spec = make_spec();

    auto input_a = builder.createInput(spec);
    auto input_b = builder.createInput(spec);
    auto add1 = builder.createOperator(Operator::Add, spec, input_a, input_b);
    auto add2 = builder.createOperator(Operator::Add, spec, add1, input_b);
    auto add3 = builder.createOperator(Operator::Add, spec, add2, input_b);
    builder.createOutput(add3);

    auto graph = builder.build();

    // Mark all ops for XNNPACK so they get fused.
    graph.nodes[add1.node].flags |= NodeFlags::UseXnnpack;
    graph.nodes[add2.node].flags |= NodeFlags::UseXnnpack;
    graph.nodes[add3.node].flags |= NodeFlags::UseXnnpack;

    partition_xnn_subgraphs(graph);

    auto order = schedule(graph);

    assert_all_nodes_present(graph, order);
    assert_topological(graph, order);

    // Find the CallSubgraphNode and verify it comes after inputs.
    std::vector<uint32_t> pos(graph.nodes.size());
    for (uint32_t i = 0; i < order.size(); i++) {
        pos[order[i]] = i;
    }

    for (NodeHandle n = 0; n < graph.nodes.size(); n++) {
        if (std::holds_alternative<CallSubgraphNode>(graph.nodes[n].value)) {
            auto& fused = std::get<CallSubgraphNode>(graph.nodes[n].value);
            for (auto& arg : fused.args) {
                EXPECT_LT(pos[arg.node], pos[n]);
            }
        }
    }
}

TEST(TestSchedule, multiple_inputs_no_ops) {
    // Graph with only input nodes (degenerate case).
    auto builder = GraphBuilder();
    auto spec = make_spec();

    auto input_a = builder.createInput(spec);
    auto input_b = builder.createInput(spec);
    builder.createOutput(input_a);
    builder.createOutput(input_b);

    auto graph = builder.build();
    graph.update_users();

    auto order = schedule(graph);

    assert_all_nodes_present(graph, order);
    EXPECT_EQ(order.size(), 2u);
}
