#include <gtest/gtest.h>

#include <executorch/backends/xnnpack/runtime/graph/graph_builder.h>

using namespace executorch::backends::xnnpack::core;
using namespace executorch::backends::xnnpack::graph;

TEST(TestGraphBuilder, add) {
    // A simple tests which constructs a graph with one input tensor,
    // one add operator, and one output tensor.
    auto builder = GraphBuilder();

    auto tensor_spec = TensorSpec {
        .dtype = DType::Float32,
        .sizes = { DimSizeSpec::constant(1), DimSizeSpec::constant(10) }
    };

    auto input_a = builder.createInput(tensor_spec);
    auto input_b = builder.createInput(tensor_spec);
    auto add = builder.createOperator(Operator::Add, tensor_spec, input_a, input_b);
    auto output = builder.createOutput(add);

    auto graph = builder.build();

    // Verify the graph.
    EXPECT_EQ(graph.input_specs.size(), 2);
    EXPECT_EQ(graph.input_specs[0], tensor_spec);
    EXPECT_EQ(graph.input_specs[1], tensor_spec);

    EXPECT_EQ(graph.nodes.size(), 3);
    EXPECT_EQ(std::get<InputNode>(graph.nodes[0].value), InputNode { 0 });
    EXPECT_EQ(std::get<InputNode>(graph.nodes[1].value), InputNode { 1 });
    EXPECT_EQ(std::get<CallOperatorNode>(graph.nodes[2].value), (CallOperatorNode {
        .args = { ValueHandle{0}, ValueHandle{1} },
        .op = Operator::Add,
        .output_specs = tensor_spec
    }));

    EXPECT_EQ(graph.outputs.size(), 1);
    EXPECT_EQ(graph.outputs[0], (ValueHandle{2}));
}
