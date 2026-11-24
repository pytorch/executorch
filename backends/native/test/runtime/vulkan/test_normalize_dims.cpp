// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/vulkan/passes/NormalizeDims.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace ptn::vulkan {
namespace {

// One call of `target` on a rank-4 input (value 0) with `dim_args`.
Graph make_graph(std::string target, std::vector<NamedArgument> dim_args) {
  Graph graph;
  graph.values = {
      Value("input", ScalarType::Float, {2, 3, 4, 8}),
      Value("output", ScalarType::Float, {2, 3, 4, 8}),
  };
  std::vector<NamedArgument> inputs = {{.name = "self", .arg = TensorArg{0}}};
  for (NamedArgument& arg : dim_args) {
    inputs.push_back(std::move(arg));
  }
  graph.nodes = {Node{
      .name = "op",
      .target = std::move(target),
      .inputs = std::move(inputs),
      .outputs = {{.value_id = 1}},
  }};
  graph.schedule = {0};
  graph.rebuild_def_use();
  return graph;
}

int64_t int_value(const Graph& graph, size_t input) {
  return graph.node(0).inputs.at(input).arg.as_int().value;
}

// cppcheck-suppress-begin syntaxError
TEST(NormalizeDimsTest, TreatsAScalarAsRankOne) {
  Graph graph;
  graph.values = {
      Value("input", ScalarType::Float, {}),
      Value("output", ScalarType::Float, {}),
  };
  graph.nodes = {Node{
      .name = "op",
      .target = "torch.ops.aten.squeeze_copy.dims",
      .inputs =
          {{.name = "self", .arg = TensorArg{0}},
           {.name = "dim", .arg = IntListArg{.values = {-1}}}},
      .outputs = {{.value_id = 1}},
  }};
  graph.schedule = {0};
  graph.rebuild_def_use();

  EXPECT_EQ(make_dims_non_negative(graph), 1);
  EXPECT_EQ(
      graph.node(0).inputs.at(1).arg.as_int_list().values,
      std::vector<int64_t>{0});
}

TEST(NormalizeDimsTest, AddsTheRankToNegativeDims) {
  Graph graph = make_graph(
      "torch.ops.aten.transpose_copy.int",
      {{.name = "dim0", .arg = IntArg{-1}},
       {.name = "dim1", .arg = IntArg{1}}});

  EXPECT_EQ(make_dims_non_negative(graph), 1);
  EXPECT_EQ(int_value(graph, 1), 3);
  EXPECT_EQ(int_value(graph, 2), 1);
}

TEST(NormalizeDimsTest, RewritesEveryEntryOfADimList) {
  Graph graph = make_graph(
      "torch.ops.aten.permute_copy.default",
      {{.name = "dims", .arg = IntListArg{.values = {0, -1, -3, -2}}}});

  EXPECT_EQ(make_dims_non_negative(graph), 3);
  const std::vector<int64_t> expected = {0, 3, 1, 2};
  EXPECT_EQ(graph.node(0).inputs.at(1).arg.as_int_list().values, expected);
}

TEST(NormalizeDimsTest, RewritesDim2StartDimAndEndDim) {
  Graph flatten = make_graph(
      "torch.ops.aten.flatten.using_ints",
      {{.name = "start_dim", .arg = IntArg{-3}},
       {.name = "end_dim", .arg = IntArg{-1}}});
  EXPECT_EQ(make_dims_non_negative(flatten), 2);
  EXPECT_EQ(int_value(flatten, 1), 1);
  EXPECT_EQ(int_value(flatten, 2), 3);

  Graph diagonal = make_graph(
      "torch.ops.aten.diagonal_copy.default",
      {{.name = "dim1", .arg = IntArg{0}},
       {.name = "dim2", .arg = IntArg{-1}}});
  EXPECT_EQ(make_dims_non_negative(diagonal), 1);
  EXPECT_EQ(int_value(diagonal, 2), 3);
}

TEST(NormalizeDimsTest, IndexesUnsqueezeDimsByTheOutputRank) {
  Graph graph = make_graph(
      "torch.ops.aten.unsqueeze_copy.default",
      {{.name = "dim", .arg = IntArg{-1}}});

  make_dims_non_negative(graph);

  EXPECT_EQ(int_value(graph, 1), 4);
}

TEST(NormalizeDimsTest, TakesTheRankOfTheFirstListedTensor) {
  Graph graph = make_graph("torch.ops.aten.cat.default", {});
  graph.node(0).inputs = {
      {.name = "tensors", .arg = TensorListArg{.ids = {0, 0}}},
      {.name = "dim", .arg = IntArg{-2}},
  };

  make_dims_non_negative(graph);

  EXPECT_EQ(int_value(graph, 1), 2);
}

TEST(NormalizeDimsTest, LeavesNonAtenOpsAlone) {
  Graph graph = make_graph(
      "torch.ops.test.custom.default", {{.name = "dim", .arg = IntArg{-1}}});

  EXPECT_EQ(make_dims_non_negative(graph), 0);
  EXPECT_EQ(int_value(graph, 1), -1);
}

TEST(NormalizeDimsTest, RejectsOutOfRangeAndSymbolicDims) {
  Graph out_of_range = make_graph(
      "torch.ops.aten._softmax.default", {{.name = "dim", .arg = IntArg{-5}}});
  EXPECT_THROW(make_dims_non_negative(out_of_range), std::runtime_error);

  Graph symbolic = make_graph(
      "torch.ops.aten._softmax.default",
      {{.name = "dim", .arg = IntArg{.value = -1, .id = 1}}});
  EXPECT_THROW(make_dims_non_negative(symbolic), std::runtime_error);

  Graph symbolic_list = make_graph(
      "torch.ops.aten.sum.dim_IntList",
      {{.name = "dim", .arg = IntListArg{.values = {-1}, .ids = {1}}}});
  EXPECT_THROW(make_dims_non_negative(symbolic_list), std::runtime_error);
}

TEST(NormalizeDimsTest, SkipsNodesErasedFromTheSchedule) {
  // The output's id would keep the erased node's output live.
  Graph graph = make_graph(
      "torch.ops.aten._softmax.default",
      {{.name = "dim", .arg = IntArg{.value = -1, .id = 0}}});
  graph.erase_node(0);

  EXPECT_EQ(make_dims_non_negative(graph), 0);
}
// cppcheck-suppress-end syntaxError

} // namespace
} // namespace ptn::vulkan
