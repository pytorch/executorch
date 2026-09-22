// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/graph/Graph.h>

#include <stdexcept>
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

Node output(const char* name, ValueId input) {
  return Node{
      .name = name,
      .op_kind = OpKind::Output,
      .inputs = {{.arg = TensorArg{input}}},
  };
}

Graph make_linear_graph() {
  Graph graph;
  graph.values = {tensor("input"), tensor("first"), tensor("output")};
  graph.values[0].role = ValueRole::UserInput;
  graph.nodes = {
      placeholder("input", 0),
      call("first", 0, 1),
      call("second", 1, 2),
      output("output", 2),
  };
  graph.input_ids = {0};
  graph.output_ids = {2};
  graph.initialize_schedule();
  graph.rebuild_def_use();
  return graph;
}

// cppcheck-suppress-begin syntaxError
TEST(GraphTest, RebuildDefUseIgnoresInactiveNodes) {
  Graph graph = make_linear_graph();
  graph.schedule.erase(graph.schedule.begin() + 2);

  graph.rebuild_def_use();

  EXPECT_EQ(graph.value(1).consumer_ids, std::vector<NodeId>{});
  EXPECT_EQ(graph.value(2).producer_id, kInvalid);
  EXPECT_EQ(graph.value(2).consumer_ids, std::vector<NodeId>({3}));
}

TEST(GraphTest, InsertAndReplacePreserveStableNodeIds) {
  Graph graph = make_linear_graph();
  const ValueId transition_value = graph.append_value(tensor("transition"));
  const NodeId transition =
      graph.insert_node_before(2, call("transition", 1, transition_value));

  EXPECT_EQ(transition, 4);
  EXPECT_EQ(graph.schedule, std::vector<NodeId>({0, 1, 4, 2, 3}));
  EXPECT_EQ(graph.replace_all_uses(1, transition_value, transition), 1);
  EXPECT_EQ(
      graph.node(transition).input_value_ids(), std::vector<ValueId>({1}));
  EXPECT_EQ(graph.node(2).input_value_ids(), std::vector<ValueId>({3}));
  EXPECT_EQ(graph.value(1).consumer_ids, std::vector<NodeId>({transition}));
  EXPECT_EQ(
      graph.value(transition_value).consumer_ids, std::vector<NodeId>({2}));
}

TEST(GraphTest, InsertNodeAfterUpdatesScheduleAndDefUse) {
  Graph graph = make_linear_graph();
  const ValueId side_value = graph.append_value(tensor("side"));

  const NodeId side = graph.insert_node_after(1, call("side", 1, side_value));

  EXPECT_EQ(side, 4);
  EXPECT_EQ(graph.schedule, std::vector<NodeId>({0, 1, 4, 2, 3}));
  EXPECT_EQ(graph.value(side_value).producer_id, side);
  EXPECT_EQ(graph.value(1).consumer_ids, std::vector<NodeId>({side, 2}));
}

TEST(GraphTest, ReplaceInputChangesOnlyRequestedNode) {
  Graph graph = make_linear_graph();

  EXPECT_EQ(graph.replace_input(2, 1, 0), 1);

  EXPECT_EQ(graph.node(1).input_value_ids(), std::vector<ValueId>({0}));
  EXPECT_EQ(graph.node(2).input_value_ids(), std::vector<ValueId>({0}));
  EXPECT_EQ(graph.value(1).consumer_ids, std::vector<NodeId>{});
  EXPECT_EQ(graph.output_ids, std::vector<ValueId>({2}));
}

TEST(GraphTest, ReplaceInputUpdatesEveryReferenceBearingArgument) {
  Graph graph = make_linear_graph();
  graph.node(2).inputs = {
      {.arg = TensorArg{1}},
      {.arg = IntArg{7, 1}},
      {.arg = FloatArg{2.5, 1}},
      {.arg = BoolArg{true, 1}},
      {.arg = IntListArg{{3, 4}, {1, kInvalid}}},
      {.arg = TensorListArg{{1, 1}}},
      {.arg = OptionalTensorListArg{{kInvalid, 1}}},
  };
  graph.rebuild_def_use();

  EXPECT_EQ(graph.replace_input(2, 1, 0), 8);

  EXPECT_EQ(
      graph.node(2).input_value_ids(),
      std::vector<ValueId>({0, 0, 0, 0, 0, 0, 0, 0}));
  EXPECT_EQ(graph.node(2).inputs[1].arg.as_int().value, 7);
  EXPECT_DOUBLE_EQ(graph.node(2).inputs[2].arg.as_float().value, 2.5);
  EXPECT_TRUE(graph.node(2).inputs[3].arg.as_bool().value);
}

TEST(GraphTest, ReplaceAllUsesRejectsGraphInput) {
  Graph graph = make_linear_graph();

  EXPECT_THROW(graph.replace_all_uses(0, 1), std::runtime_error);
  EXPECT_EQ(graph.input_ids, std::vector<ValueId>({0}));
}

TEST(GraphTest, ReplaceAllUsesUpdatesOutputMetadata) {
  Graph graph = make_linear_graph();

  EXPECT_EQ(graph.replace_all_uses(2, 1), 1);

  EXPECT_EQ(graph.output_ids, std::vector<ValueId>({1}));
  EXPECT_EQ(graph.node(3).input_value_ids(), std::vector<ValueId>({1}));
  EXPECT_EQ(graph.value(2).consumer_ids, std::vector<NodeId>{});
}

TEST(GraphTest, ReplaceAllUsesUpdatesAliasMetadata) {
  Graph graph = make_linear_graph();
  const ValueId alias = graph.append_value(tensor("alias"));
  graph.value(alias).alias_id = 2;

  EXPECT_EQ(graph.replace_all_uses(2, 1), 2);

  EXPECT_EQ(graph.value(alias).alias_id, 1);
  EXPECT_EQ(graph.node(3).input_value_ids(), std::vector<ValueId>({1}));
}

TEST(GraphTest, ReplaceAllUsesPreservesReplacementAliasSource) {
  Graph graph = make_linear_graph();
  const ValueId alias = graph.append_value(tensor("alias"));
  graph.value(alias).alias_id = 2;

  EXPECT_EQ(graph.replace_all_uses(2, alias), 1);

  EXPECT_EQ(graph.value(alias).alias_id, 2);
  EXPECT_EQ(graph.node(3).input_value_ids(), std::vector<ValueId>({alias}));
}

TEST(GraphTest, RebuildDefUseDeduplicatesRepeatedOperands) {
  Graph graph = make_linear_graph();
  graph.node(2).inputs = {
      {.arg = TensorArg{1}},
      {.arg = TensorListArg{{1, 1}}},
  };

  graph.rebuild_def_use();

  EXPECT_EQ(graph.value(1).consumer_ids, std::vector<NodeId>({2}));
}

TEST(GraphTest, RebuildDefUseRejectsInvalidInputId) {
  Graph graph = make_linear_graph();
  graph.node(2).inputs = {{.arg = TensorArg{99}}};

  EXPECT_THROW(graph.rebuild_def_use(), std::runtime_error);
}

TEST(GraphTest, RebuildDefUseRejectsInvalidOutputId) {
  Graph graph = make_linear_graph();
  graph.node(2).outputs = {{.value_id = 99}};

  EXPECT_THROW(graph.rebuild_def_use(), std::runtime_error);
}

TEST(GraphTest, MutationsRejectInactiveNodes) {
  Graph graph = make_linear_graph();
  graph.schedule.erase(graph.schedule.begin() + 1);
  const std::vector<NodeId> schedule = graph.schedule;
  const size_t node_count = graph.nodes.size();

  EXPECT_THROW(
      graph.insert_node_before(1, call("insert", 0, 1)), std::runtime_error);
  EXPECT_THROW(graph.replace_input(1, 0, 1), std::runtime_error);
  EXPECT_THROW(graph.erase_node(1), std::runtime_error);

  EXPECT_EQ(graph.schedule, schedule);
  EXPECT_EQ(graph.nodes.size(), node_count);
}

TEST(GraphTest, EraseNodeRejectsBoundariesAndLiveOutputs) {
  Graph graph = make_linear_graph();

  EXPECT_THROW(graph.erase_node(0), std::runtime_error);
  EXPECT_THROW(graph.erase_node(3), std::runtime_error);
  EXPECT_THROW(graph.erase_node(1), std::runtime_error);

  const ValueId unused_value = graph.append_value(tensor("unused"));
  const NodeId unused =
      graph.insert_node_before(3, call("unused", 2, unused_value));
  EXPECT_NO_THROW(graph.erase_node(unused));
  EXPECT_EQ(graph.schedule, std::vector<NodeId>({0, 1, 2, 3}));
  EXPECT_EQ(graph.value(unused_value).producer_id, kInvalid);
}

TEST(GraphTest, EraseNodeRejectsGraphOutputWithoutConsumers) {
  Graph graph = make_linear_graph();
  const ValueId retained_value = graph.append_value(tensor("retained"));
  const NodeId retained =
      graph.insert_node_before(3, call("retained", 2, retained_value));
  graph.output_ids.push_back(retained_value);
  const std::vector<NodeId> schedule = graph.schedule;

  EXPECT_THROW(graph.erase_node(retained), std::runtime_error);
  EXPECT_EQ(graph.value(retained_value).producer_id, retained);
  EXPECT_EQ(graph.schedule, schedule);
}

TEST(GraphTest, InitializeScheduleRejectsExistingSchedule) {
  Graph graph = make_linear_graph();
  const std::vector<NodeId> schedule = graph.schedule;

  EXPECT_THROW(graph.initialize_schedule(), std::runtime_error);
  EXPECT_EQ(graph.schedule, schedule);
}

TEST(NodeTest, OutputValueIdsFlattenSparseTensorLists) {
  const Node node{
      .outputs =
          {
              {.value_id = 0},
              {.kind = OutputValueKind::TensorList,
               .elem_ids = {1, kInvalid, 2}},
          },
  };

  EXPECT_EQ(node.output_value_ids(), std::vector<ValueId>({0, 1, 2}));
}
// cppcheck-suppress-end syntaxError

} // namespace
} // namespace ptn
