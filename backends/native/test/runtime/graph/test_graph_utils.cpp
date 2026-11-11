// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/graph/utils/GraphUtils.h>

#include <array>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace ptn {
namespace {

Value tensor(const char* name) {
  return Value(name, ScalarType::Float, {1});
}

Node placeholder(const char* name, ValueId output) {
  return Node{
      .name = name,
      .op_kind = OpKind::Placeholder,
      .outputs = {{.value_id = output}},
  };
}

Node call(const char* name, ValueId input, ValueId output) {
  return Node{
      .name = name,
      .target = "test.op",
      .inputs = {{.name = "input", .arg = TensorArg{input}}},
      .outputs = {{.value_id = output}},
  };
}

Node output(const char* name, std::vector<ValueId> inputs) {
  Node node{.name = name, .op_kind = OpKind::Output};
  for (ValueId input : inputs) {
    node.inputs.push_back({.arg = TensorArg{input}});
  }
  return node;
}

Graph make_linear_graph() {
  Graph graph;
  graph.values = {tensor("input"), tensor("first"), tensor("output")};
  graph.values[0].role = ValueRole::UserInput;
  graph.nodes = {
      placeholder("input", 0),
      call("first", 0, 1),
      call("second", 1, 2),
      output("output", {2}),
  };
  graph.input_ids = {0};
  graph.output_ids = {2};
  graph.initialize_schedule();
  graph.rebuild_def_use();
  return graph;
}

// cppcheck-suppress-begin syntaxError
TEST(GraphUtilsTest, StableTopologicalSortPreservesIndependentOrder) {
  Graph graph = make_linear_graph();
  const ValueId independent_value = graph.append_value(tensor("independent"));
  const NodeId independent =
      graph.insert_node_before(3, call("independent", 0, independent_value));
  graph.schedule = {0, 2, independent, 1, 3};

  stable_topological_sort(graph);

  EXPECT_EQ(graph.schedule, std::vector<NodeId>({0, independent, 1, 2, 3}));
  EXPECT_NO_THROW(validate_graph(graph));
}

TEST(GraphUtilsTest, StableTopologicalSortRejectsCycle) {
  Graph graph = make_linear_graph();
  graph.node(1).inputs = {{.arg = TensorArg{2}}};

  EXPECT_THROW(stable_topological_sort(graph), std::runtime_error);
}

TEST(GraphUtilsTest, StableTopologicalSortRejectsMissingProducer) {
  Graph graph = make_linear_graph();
  graph.schedule.erase(graph.schedule.begin() + 1);

  EXPECT_THROW(stable_topological_sort(graph), std::runtime_error);
}

TEST(GraphUtilsTest, StableTopologicalSortRejectsDuplicateScheduleEntries) {
  Graph graph = make_linear_graph();
  graph.schedule.push_back(1);

  EXPECT_THROW(stable_topological_sort(graph), std::runtime_error);
}

TEST(GraphUtilsTest, StableTopologicalSortRejectsInvalidScheduledNode) {
  Graph graph = make_linear_graph();
  graph.schedule.push_back(99);

  EXPECT_THROW(stable_topological_sort(graph), std::runtime_error);
}

TEST(GraphUtilsTest, StableTopologicalSortRejectsMultipleProducers) {
  Graph graph = make_linear_graph();
  graph.node(1).outputs.push_back({.value_id = 2});

  EXPECT_THROW(stable_topological_sort(graph), std::runtime_error);
}

TEST(GraphUtilsTest, StableTopologicalSortRejectsInvalidInputId) {
  Graph graph = make_linear_graph();
  graph.node(2).inputs = {{.arg = TensorArg{99}}};

  EXPECT_THROW(stable_topological_sort(graph), std::runtime_error);
}

TEST(GraphUtilsTest, StableTopologicalSortRejectsInvalidOutputId) {
  Graph graph = make_linear_graph();
  graph.node(2).outputs = {{.value_id = 99}};

  EXPECT_THROW(stable_topological_sort(graph), std::runtime_error);
}

TEST(GraphUtilsTest, StableTopologicalSortDeduplicatesRepeatedDependencies) {
  Graph graph = make_linear_graph();
  graph.node(2).inputs = {
      {.arg = TensorArg{1}},
      {.arg = TensorListArg{{1, 1}}},
  };
  graph.schedule = {0, 2, 1, 3};

  stable_topological_sort(graph);

  EXPECT_EQ(graph.schedule, std::vector<NodeId>({0, 1, 2, 3}));
  EXPECT_EQ(graph.value(1).consumer_ids, std::vector<NodeId>({2}));
}

TEST(GraphUtilsTest, ValidateRejectsDuplicateScheduleEntries) {
  Graph graph = make_linear_graph();
  graph.schedule.push_back(1);

  EXPECT_THROW(validate_graph(graph), std::runtime_error);
}

TEST(GraphUtilsTest, ValidateRejectsInvalidScheduledNode) {
  Graph graph = make_linear_graph();
  graph.schedule.push_back(99);

  EXPECT_THROW(validate_graph(graph), std::runtime_error);
}

TEST(GraphUtilsTest, ValidateRejectsMultipleProducers) {
  Graph graph = make_linear_graph();
  graph.node(1).outputs.push_back({.value_id = 2});
  graph.rebuild_def_use();

  EXPECT_THROW(validate_graph(graph), std::runtime_error);
}

TEST(GraphUtilsTest, ValidateRejectsNonTopologicalSchedule) {
  Graph graph = make_linear_graph();
  graph.schedule = {0, 2, 1, 3};
  graph.rebuild_def_use();

  EXPECT_THROW(validate_graph(graph), std::runtime_error);
}

TEST(GraphUtilsTest, ValidateRejectsInvalidNodeValueIds) {
  Graph invalid_input = make_linear_graph();
  invalid_input.node(2).inputs = {{.arg = TensorArg{99}}};
  EXPECT_THROW(validate_graph(invalid_input), std::runtime_error);

  Graph invalid_output = make_linear_graph();
  invalid_output.node(2).outputs = {{.value_id = 99}};
  EXPECT_THROW(validate_graph(invalid_output), std::runtime_error);
}

TEST(GraphUtilsTest, ValidateRejectsNonTensorAliases) {
  Graph graph = make_linear_graph();
  const ValueId scalar = graph.append_value(Value("scalar", Scalar(1)));
  graph.value(1).alias_id = scalar;

  EXPECT_THROW(validate_graph(graph), std::runtime_error);

  graph.value(1).alias_id = kInvalid;
  graph.value(scalar).alias_id = 1;
  EXPECT_THROW(validate_graph(graph), std::runtime_error);
}

TEST(GraphUtilsTest, ValidateRejectsInvalidAndCyclicAliases) {
  Graph graph = make_linear_graph();
  graph.value(1).alias_id = 99;
  EXPECT_THROW(validate_graph(graph), std::runtime_error);

  graph.value(1).alias_id = 2;
  graph.value(2).alias_id = 1;
  EXPECT_THROW(validate_graph(graph), std::runtime_error);
}

TEST(GraphUtilsTest, ValidateRequiresPlaceholderForGraphInput) {
  Graph graph = make_linear_graph();
  graph.input_ids = {1};

  EXPECT_THROW(validate_graph(graph), std::runtime_error);
}

TEST(GraphUtilsTest, ValidateRejectsDuplicateGraphInputs) {
  Graph graph = make_linear_graph();
  graph.input_ids = {0, 0};

  EXPECT_THROW(validate_graph(graph), std::runtime_error);
}

TEST(GraphUtilsTest, ValidateRequiresGraphInputOrderToMatchPlaceholders) {
  Graph graph;
  graph.values = {tensor("first"), tensor("second"), tensor("output")};
  graph.nodes = {
      placeholder("first", 0),
      placeholder("second", 1),
      Node{
          .name = "combine",
          .target = "test.combine",
          .inputs = {{.arg = TensorListArg{{0, 1}}}},
          .outputs = {{.value_id = 2}},
      },
      output("output", {2}),
  };
  graph.input_ids = {1, 0};
  graph.output_ids = {2};
  graph.initialize_schedule();
  graph.rebuild_def_use();

  EXPECT_THROW(validate_graph(graph), std::runtime_error);
}

TEST(GraphUtilsTest, ValidateRequiresGraphOutputOrderToMatchOutputNode) {
  Graph graph = make_linear_graph();
  graph.node(3) = output("output", {1, 2});
  graph.output_ids = {2, 1};
  graph.rebuild_def_use();

  EXPECT_THROW(validate_graph(graph), std::runtime_error);
}

TEST(GraphUtilsTest, ValidateRejectsStaleDefUseInformation) {
  Graph graph = make_linear_graph();
  graph.value(1).consumer_ids.clear();

  EXPECT_THROW(validate_graph(graph), std::runtime_error);
}

TEST(GraphUtilsTest, ValidateRejectsInvalidSubgraphReference) {
  Graph graph = make_linear_graph();
  graph.node(2).inputs.push_back({.arg = GraphArg{"body", 0}});
  graph.rebuild_def_use();

  EXPECT_THROW(validate_graph(graph), std::runtime_error);
}

TEST(GraphUtilsTest, ValidateAcceptsValidSubgraphReference) {
  Graph graph = make_linear_graph();
  graph.subgraphs.push_back(make_linear_graph());
  graph.node(2).inputs.push_back({.arg = GraphArg{"body", 0}});
  graph.rebuild_def_use();

  EXPECT_NO_THROW(validate_graph(graph));
}

TEST(GraphUtilsTest, ValidateRecursesIntoSubgraphs) {
  Graph graph = make_linear_graph();
  graph.subgraphs.push_back(make_linear_graph());
  graph.subgraphs[0].schedule.push_back(1);

  EXPECT_THROW(validate_graph(graph), std::runtime_error);
}

TEST(GraphUtilsTest, ValidateAllowsSparseTensorListOutputs) {
  Graph graph;
  graph.values = {tensor("input"), tensor("output")};
  graph.values[0].role = ValueRole::UserInput;
  graph.nodes = {
      placeholder("input", 0),
      Node{
          .name = "tensor_list",
          .target = "test.tensor_list",
          .inputs = {{.name = "input", .arg = TensorArg{0}}},
          .outputs =
              {{.kind = OutputValueKind::TensorList,
                .elem_ids = {1, kInvalid}}},
      },
      output("output", {1}),
  };
  graph.input_ids = {0};
  graph.output_ids = {1};
  graph.initialize_schedule();
  graph.rebuild_def_use();

  EXPECT_NO_THROW(validate_graph(graph));
}

TEST(GraphUtilsTest, FindArgMatchesNameThenUnnamedPosition) {
  const Node node{
      .name = "op",
      .target = "test.op",
      .inputs =
          {{.arg = TensorArg{0}},
           {.name = "scale", .arg = FloatArg{0.5}},
           {.name = "flag", .arg = BoolArg{true}}},
  };
  EXPECT_EQ(tensor_id(find_arg(node, "input", 0)), 0);
  EXPECT_EQ(find_arg(node, "flag", 0), &node.inputs[2].arg);
  // A named input belongs to its own parameter, even at another's position.
  EXPECT_EQ(find_arg(node, "mask", 1), nullptr);
  EXPECT_EQ(find_arg(node, "dropout", 3), nullptr);
  EXPECT_EQ(tensor_id(find_arg(node, "scale", 1)), std::nullopt);
}

TEST(GraphUtilsTest, IsBoolOrAbsentTakesDefaultWhenAbsent) {
  const Argument literal = BoolArg{true};
  const Argument symbolic = BoolArg{.value = true, .id = 0};
  EXPECT_TRUE(is_bool_or_absent(nullptr, false));
  EXPECT_TRUE(is_bool_or_absent(&literal, true));
  EXPECT_FALSE(is_bool_or_absent(&literal, false));
  EXPECT_FALSE(is_bool_or_absent(&symbolic, true));
}

TEST(GraphUtilsTest, FloatOrAbsentReadsOnlyLiterals) {
  const Argument literal = FloatArg{0.5};
  const Argument symbolic = FloatArg{.value = 0.5, .id = 0};
  const Argument none = NoneArg{};
  EXPECT_EQ(float_or_absent(nullptr, 1.0), 1.0);
  EXPECT_EQ(float_or_absent(&literal, 1.0), 0.5);
  EXPECT_EQ(float_or_absent(&symbolic, 1.0), std::nullopt);
  EXPECT_EQ(float_or_absent(&none, 1.0), std::nullopt);
}

TEST(GraphUtilsTest, InputsFitSchemaRejectsUnknownNamesAndExtraInputs) {
  constexpr std::array<std::string_view, 2> params{"self", "dim"};
  const auto node = [](std::vector<std::string> names) {
    Node n;
    for (std::string& name : names) {
      n.inputs.push_back({.name = std::move(name), .arg = IntArg{0}});
    }
    return n;
  };
  EXPECT_TRUE(inputs_fit_schema(node({"dim"}), params));
  EXPECT_TRUE(inputs_fit_schema(node({"", ""}), params));
  EXPECT_FALSE(inputs_fit_schema(node({"self", "keepdim"}), params));
  EXPECT_FALSE(inputs_fit_schema(node({"", "", ""}), params));
}
// cppcheck-suppress-end syntaxError

} // namespace
} // namespace ptn
