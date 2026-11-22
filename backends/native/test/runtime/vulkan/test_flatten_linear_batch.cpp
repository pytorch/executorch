// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/vulkan/passes/FlattenLinearBatch.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace ptn::vulkan {
namespace {

// linear(input, weight) for an input of `sizes` and 8 output features.
Graph make_linear_graph(std::vector<int64_t> sizes) {
  std::vector<int64_t> out_sizes = sizes;
  out_sizes.back() = 8;
  Graph graph;
  graph.values = {
      Value("input", ScalarType::Float, sizes),
      Value("weight", ScalarType::Float, {8, sizes.back()}),
      Value("output", ScalarType::Float, out_sizes),
  };
  graph.values[1].role = ValueRole::Parameter;
  graph.nodes = {Node{
      .name = "linear",
      .target = "torch.ops.aten.linear.default",
      .inputs =
          {
              {.name = "input", .arg = TensorArg{0}},
              {.name = "weight", .arg = TensorArg{1}},
              {.name = "bias", .arg = NoneArg{}},
          },
      .outputs = {{.value_id = 2}},
  }};
  graph.schedule = {0};
  graph.rebuild_def_use();
  return graph;
}

std::vector<std::string> scheduled_targets(const Graph& graph) {
  std::vector<std::string> targets;
  for (const NodeId id : graph.schedule) {
    targets.push_back(graph.node(id).target);
  }
  return targets;
}

// cppcheck-suppress-begin syntaxError
TEST(FlattenLinearBatchTest, ReshapesARank4LinearToRank3AndBack) {
  Graph graph = make_linear_graph({2, 3, 5, 4});

  EXPECT_EQ(flatten_linear_batches(graph), 1);

  const std::vector<std::string> expected = {
      "torch.ops.aten.view_copy.default",
      "torch.ops.aten.linear.default",
      "torch.ops.aten.view_copy.default"};
  EXPECT_EQ(scheduled_targets(graph), expected);
  const Node& linear = graph.node(0);
  const std::vector<int64_t> flat_input = {6, 5, 4};
  const std::vector<int64_t> flat_output = {6, 5, 8};
  EXPECT_EQ(
      graph.value(linear.inputs[0].arg.as_tensor().id).tensor_meta().sizes,
      flat_input);
  EXPECT_EQ(
      graph.value(linear.outputs[0].value_id).tensor_meta().sizes, flat_output);
  // The original output is now the reshape's.
  EXPECT_EQ(graph.node(graph.schedule.back()).outputs[0].value_id, 2);
}

TEST(FlattenLinearBatchTest, LeavesRank3LinearAlone) {
  Graph graph = make_linear_graph({3, 5, 4});

  EXPECT_EQ(flatten_linear_batches(graph), 0);
  EXPECT_EQ(graph.schedule.size(), 1);
}

TEST(FlattenLinearBatchTest, RejectsADynamicRank4Input) {
  Graph graph = make_linear_graph({2, 3, 5, 4});
  graph.values[0].tensor_meta().lower_bounds = {1, 3, 5, 4};

  EXPECT_THROW(flatten_linear_batches(graph), std::runtime_error);
}

TEST(FlattenLinearBatchTest, RejectsBatchDimsWhoseProductOverflows) {
  Graph graph = make_linear_graph({int64_t{1} << 62, 4, 5, 4});

  try {
    flatten_linear_batches(graph);
    FAIL() << "expected the overflowing batch dims to be rejected";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("overflows"), std::string::npos)
        << e.what();
  }
}

TEST(FlattenLinearBatchTest, RejectsARank4InputWithoutARank4Output) {
  Graph rank2 = make_linear_graph({2, 3, 5, 4});
  rank2.values[2].tensor_meta().sizes = {30, 8};
  EXPECT_THROW(flatten_linear_batches(rank2), std::runtime_error);

  Graph rank0 = make_linear_graph({2, 3, 5, 4});
  rank0.values[2].tensor_meta().sizes = {};
  EXPECT_THROW(flatten_linear_batches(rank0), std::runtime_error);
}
// cppcheck-suppress-end syntaxError

} // namespace
} // namespace ptn::vulkan
