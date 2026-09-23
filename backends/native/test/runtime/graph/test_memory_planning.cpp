// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/graph/MemoryPlanning.h>

#include <stdexcept>
#include <vector>

#include <executorch/backends/native/runtime/graph/utils/GraphUtils.h>
#include <gtest/gtest.h>

namespace ptn {
namespace {

Value tensor(const char* name) {
  return Value(name, ScalarType::Float, {1});
}

Node placeholder(ValueId output) {
  return Node{
      .name = "input",
      .op_kind = OpKind::Placeholder,
      .outputs = {{.value_id = output}},
  };
}

Node call(const char* name, ValueId input, ValueId output) {
  return Node{
      .name = name,
      .target = "test.op",
      .inputs = {{.arg = TensorArg{input}}},
      .outputs = {{.value_id = output}},
  };
}

Node output(ValueId input) {
  return Node{
      .name = "output",
      .op_kind = OpKind::Output,
      .inputs = {{.arg = TensorArg{input}}},
  };
}

Graph make_graph() {
  Graph graph;
  graph.values = {
      tensor("input"),
      tensor("first"),
      tensor("second"),
      tensor("output"),
  };
  graph.values[0].role = ValueRole::UserInput;
  graph.nodes = {
      placeholder(0),
      call("first", 0, 1),
      call("second", 1, 2),
      call("third", 2, 3),
      output(3),
  };
  graph.input_ids = {0};
  graph.output_ids = {3};
  graph.initialize_schedule();
  graph.rebuild_def_use();
  return graph;
}

// cppcheck-suppress-begin syntaxError
TEST(MemoryPlanningTest, ReusesOnlyStrictlyDisjointLifetimes) {
  const Graph graph = make_graph();
  const MemoryPlan plan = plan_memory(
      graph,
      {
          {
              .value_id = 1,
              .size_bytes = 64,
              .alignment = 16,
              .memory_kind = 0,
          },
          {
              .value_id = 2,
              .size_bytes = 32,
              .alignment = 8,
              .memory_kind = 0,
          },
          {
              .value_id = 3,
              .size_bytes = 48,
              .alignment = 64,
              .memory_kind = 0,
          },
      });

  EXPECT_EQ(plan.allocations().size(), 2);
  EXPECT_EQ(plan.allocation_id(1), 0);
  EXPECT_EQ(plan.allocation_id(1), plan.allocation_id(3));
  EXPECT_NE(plan.allocation_id(1), plan.allocation_id(2));
  const PlannedAllocation& reused = plan.allocation(plan.allocation_id(1));
  EXPECT_EQ(reused.size_bytes, 64);
  EXPECT_EQ(reused.alignment, 64);
  EXPECT_EQ(reused.memory_kind, 0);
  EXPECT_EQ(reused.value_ids, std::vector<ValueId>({1, 3}));
}

TEST(MemoryPlanningTest, KeepsMemoryKindsSeparate) {
  const Graph graph = make_graph();
  const MemoryPlan plan = plan_memory(
      graph,
      {
          {.value_id = 1, .size_bytes = 64, .memory_kind = 1},
          {.value_id = 3, .size_bytes = 48, .memory_kind = 2},
      });

  EXPECT_NE(plan.allocation_id(1), plan.allocation_id(3));
}

TEST(MemoryPlanningTest, BreaksEqualSizeTiesByValueId) {
  const Graph graph = make_graph();
  const MemoryPlan plan = plan_memory(
      graph,
      {
          {.value_id = 2, .size_bytes = 64, .memory_kind = 0},
          {.value_id = 1, .size_bytes = 64, .memory_kind = 0},
      });

  EXPECT_EQ(plan.allocation_id(1), 0);
  EXPECT_EQ(plan.allocation_id(2), 1);
}

TEST(MemoryPlanningTest, AliasGroupUsesOneAllocationAndUnionLifetime) {
  Graph graph = make_graph();
  const ValueId alias = graph.append_value(tensor("alias"));
  graph.value(alias).alias_id = 1;
  const NodeId alias_node = graph.insert_node_after(1, call("view", 1, alias));
  graph.replace_all_uses(1, alias, alias_node);
  stable_topological_sort(graph);

  const MemoryPlan plan = plan_memory(
      graph,
      {
          {.value_id = 1, .size_bytes = 64, .memory_kind = 0},
          {.value_id = alias, .size_bytes = 64, .memory_kind = 0},
          {.value_id = 2, .size_bytes = 48, .memory_kind = 0},
      });

  EXPECT_EQ(plan.allocation_id(1), plan.allocation_id(alias));
  EXPECT_NE(plan.allocation_id(alias), plan.allocation_id(2));
}

TEST(MemoryPlanningTest, BoundaryLifetimesOverlapAdjacentValues) {
  const Graph graph = make_graph();
  const MemoryPlan plan = plan_memory(
      graph,
      {
          {.value_id = 0, .size_bytes = 64, .memory_kind = 0},
          {.value_id = 1, .size_bytes = 64, .memory_kind = 0},
          {.value_id = 2, .size_bytes = 64, .memory_kind = 0},
          {.value_id = 3, .size_bytes = 64, .memory_kind = 0},
      });

  EXPECT_NE(plan.allocation_id(0), plan.allocation_id(1));
  EXPECT_NE(plan.allocation_id(2), plan.allocation_id(3));
}

TEST(MemoryPlanningTest, KeepsRequestedPersistentValuesGraphLive) {
  for (const ValueRole role : {
           ValueRole::Parameter,
           ValueRole::Buffer,
           ValueRole::ConstantTensor,
       }) {
    SCOPED_TRACE(static_cast<int>(role));
    Graph graph = make_graph();
    graph.value(1).role = role;

    const MemoryPlan plan = plan_memory(
        graph,
        {
            {.value_id = 1, .size_bytes = 64, .memory_kind = 0},
            {.value_id = 3, .size_bytes = 48, .memory_kind = 0},
        });

    EXPECT_NE(plan.allocation_id(1), kInvalidAllocationId);
    EXPECT_NE(plan.allocation_id(1), plan.allocation_id(3));
  }
}

TEST(MemoryPlanningTest, RejectsIncompatibleAliasRequests) {
  Graph graph = make_graph();
  graph.value(2).alias_id = 1;

  EXPECT_THROW(
      plan_memory(
          graph,
          {
              {.value_id = 1, .size_bytes = 64, .memory_kind = 1},
              {.value_id = 2, .size_bytes = 64, .memory_kind = 2},
          }),
      std::runtime_error);
}

TEST(MemoryPlanningTest, AliasChainAggregatesSizeAndAlignment) {
  Graph graph = make_graph();
  const ValueId first_alias = graph.append_value(tensor("first_alias"));
  graph.value(first_alias).alias_id = 1;
  const NodeId first_alias_node =
      graph.insert_node_after(1, call("first_view", 1, first_alias));
  graph.replace_all_uses(1, first_alias, first_alias_node);
  const ValueId second_alias = graph.append_value(tensor("second_alias"));
  graph.value(second_alias).alias_id = first_alias;
  const NodeId second_alias_node = graph.insert_node_after(
      first_alias_node, call("second_view", first_alias, second_alias));
  graph.replace_all_uses(first_alias, second_alias, second_alias_node);

  const MemoryPlan plan = plan_memory(
      graph,
      {
          {
              .value_id = 1,
              .size_bytes = 64,
              .alignment = 16,
              .memory_kind = 0,
          },
          {
              .value_id = second_alias,
              .size_bytes = 96,
              .alignment = 64,
              .memory_kind = 0,
          },
      });

  EXPECT_EQ(plan.allocation_id(1), plan.allocation_id(second_alias));
  const PlannedAllocation& allocation = plan.allocation(plan.allocation_id(1));
  EXPECT_EQ(allocation.size_bytes, 96);
  EXPECT_EQ(allocation.alignment, 64);
}

TEST(MemoryPlanningTest, UnrequestedAliasExtendsGroupLifetime) {
  Graph graph = make_graph();
  const ValueId alias = graph.append_value(tensor("alias"));
  graph.value(alias).alias_id = 1;
  graph.insert_node_after(1, call("view", 1, alias));
  graph.node(3).inputs.push_back({.arg = TensorArg{alias}});
  graph.rebuild_def_use();

  const MemoryPlan plan = plan_memory(
      graph,
      {
          {.value_id = 1, .size_bytes = 64, .memory_kind = 0},
          {.value_id = 3, .size_bytes = 48, .memory_kind = 0},
      });

  EXPECT_NE(plan.allocation_id(1), plan.allocation_id(3));
  EXPECT_EQ(plan.allocation_id(alias), kInvalidAllocationId);
}

TEST(MemoryPlanningTest, EmptyRequestsProduceEmptyPlan) {
  const Graph graph = make_graph();

  const MemoryPlan plan = plan_memory(graph, {});

  EXPECT_TRUE(plan.allocations().empty());
  EXPECT_EQ(plan.allocation_id(1), kInvalidAllocationId);
}

TEST(MemoryPlanningTest, AcceptsZeroSizedAllocation) {
  const Graph graph = make_graph();

  const MemoryPlan plan =
      plan_memory(graph, {{.value_id = 1, .size_bytes = 0, .memory_kind = 0}});

  ASSERT_EQ(plan.allocations().size(), 1);
  EXPECT_EQ(plan.allocations().front().size_bytes, 0);
}

TEST(MemoryPlanningTest, RejectsInvalidRequestValueId) {
  const Graph graph = make_graph();

  EXPECT_THROW(
      plan_memory(
          graph, {{.value_id = kInvalid, .size_bytes = 64, .memory_kind = 0}}),
      std::runtime_error);
  EXPECT_THROW(
      plan_memory(
          graph, {{.value_id = 99, .size_bytes = 64, .memory_kind = 0}}),
      std::runtime_error);
}

TEST(MemoryPlanningTest, RejectsNonTensorRequest) {
  Graph graph = make_graph();
  const ValueId scalar = graph.append_value(Value("scalar", Scalar(1)));

  EXPECT_THROW(
      plan_memory(
          graph, {{.value_id = scalar, .size_bytes = 8, .memory_kind = 0}}),
      std::runtime_error);
}

TEST(MemoryPlanningTest, RejectsInactiveRequest) {
  Graph graph = make_graph();
  const ValueId inactive = graph.append_value(tensor("inactive"));

  EXPECT_THROW(
      plan_memory(
          graph, {{.value_id = inactive, .size_bytes = 64, .memory_kind = 0}}),
      std::runtime_error);
}

TEST(MemoryPlanningTest, RejectsDuplicateRequest) {
  const Graph graph = make_graph();

  EXPECT_THROW(
      plan_memory(
          graph,
          {
              {.value_id = 1, .size_bytes = 64, .memory_kind = 0},
              {.value_id = 1, .size_bytes = 64, .memory_kind = 0},
          }),
      std::runtime_error);
}

TEST(MemoryPlanningTest, RejectsInvalidMemoryKind) {
  const Graph graph = make_graph();

  EXPECT_THROW(
      plan_memory(graph, {{.value_id = 1, .size_bytes = 64}}),
      std::runtime_error);
}

TEST(MemoryPlanningTest, RejectsInvalidAlignment) {
  const Graph graph = make_graph();

  EXPECT_THROW(
      plan_memory(
          graph,
          {{
              .value_id = 1,
              .size_bytes = 64,
              .alignment = 0,
              .memory_kind = 0,
          }}),
      std::runtime_error);
  EXPECT_THROW(
      plan_memory(
          graph,
          {{
              .value_id = 1,
              .size_bytes = 64,
              .alignment = 3,
              .memory_kind = 0,
          }}),
      std::runtime_error);
}

TEST(MemoryPlanningTest, RejectsMalformedGraph) {
  Graph graph = make_graph();
  graph.value(1).consumer_ids.clear();

  EXPECT_THROW(
      plan_memory(graph, {{.value_id = 1, .size_bytes = 64, .memory_kind = 0}}),
      std::runtime_error);
}

TEST(MemoryPlanningTest, AccessorsRejectOutOfRangeIds) {
  const Graph graph = make_graph();
  const MemoryPlan plan =
      plan_memory(graph, {{.value_id = 1, .size_bytes = 64, .memory_kind = 0}});

  EXPECT_THROW(plan.allocation_id(kInvalid), std::runtime_error);
  EXPECT_THROW(plan.allocation_id(99), std::runtime_error);
  EXPECT_THROW(plan.allocation(kInvalidAllocationId), std::runtime_error);
  EXPECT_THROW(plan.allocation(99), std::runtime_error);
}
// cppcheck-suppress-end syntaxError

} // namespace
} // namespace ptn
