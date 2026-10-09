// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/vulkan/passes/MaterializeViewCopies.h>

#include <gtest/gtest.h>

namespace ptn::vulkan {
namespace {

Graph make_view_graph() {
  Graph graph;
  graph.values = {
      Value("input", ScalarType::Half, {1, 1, 128}),
      Value("view", ScalarType::Half, {1, 1, 4, 32}),
  };
  graph.values[1].alias_id = 0;
  graph.nodes = {
      Node{
          .name = "view",
          .target = "torch.ops.aten.view.default",
          .inputs =
              {
                  {.name = "self", .arg = TensorArg{0}},
                  {.name = "size",
                   .arg =
                       IntListArg{
                           .values = {1, 1, 4, 32},
                           .ids = {},
                       }},
              },
          .outputs = {{.value_id = 1}},
      },
  };
  graph.schedule = {0};
  graph.rebuild_def_use();
  return graph;
}

// cppcheck-suppress-begin syntaxError
TEST(MaterializeViewCopiesTest, ReplacesSelectedAliasWithCopy) {
  Graph graph = make_view_graph();

  EXPECT_EQ(materialize_view_copies(graph, {1}), 1);
  EXPECT_EQ(graph.node(0).target, "torch.ops.aten.view_copy.default");
  EXPECT_EQ(graph.value(1).alias_id, kInvalid);
}

TEST(MaterializeViewCopiesTest, LeavesUnselectedViewAsAlias) {
  Graph graph = make_view_graph();

  EXPECT_EQ(materialize_view_copies(graph, {}), 0);
  EXPECT_EQ(graph.node(0).target, "torch.ops.aten.view.default");
  EXPECT_EQ(graph.value(1).alias_id, 0);
}

TEST(MaterializeViewCopiesTest, LeavesMutatedViewAsAlias) {
  Graph graph = make_view_graph();
  graph.values.emplace_back(
      "updated", ScalarType::Half, std::vector<int64_t>{});
  graph.nodes.push_back(Node{
      .name = "update",
      .target = "test.update",
      .inputs = {{.name = "self", .arg = TensorArg{1}, .mutated = true}},
      .outputs = {{.value_id = 2}},
  });
  graph.schedule.push_back(1);
  graph.rebuild_def_use();

  EXPECT_EQ(materialize_view_copies(graph, {1}), 0);
  EXPECT_EQ(graph.node(0).target, "torch.ops.aten.view.default");
  EXPECT_EQ(graph.value(1).alias_id, 0);
}

TEST(MaterializeViewCopiesTest, LeavesViewOfMutatedSourceAsAlias) {
  Graph graph = make_view_graph();
  graph.values.emplace_back(
      "updated", ScalarType::Half, std::vector<int64_t>{});
  graph.nodes.push_back(Node{
      .name = "update",
      .target = "test.update",
      .inputs = {{.name = "self", .arg = TensorArg{0}, .mutated = true}},
      .outputs = {{.value_id = 2}},
  });
  graph.schedule.push_back(1);
  graph.rebuild_def_use();

  EXPECT_EQ(materialize_view_copies(graph, {1}), 0);
  EXPECT_EQ(graph.node(0).target, "torch.ops.aten.view.default");
  EXPECT_EQ(graph.value(1).alias_id, 0);
}

TEST(MaterializeViewCopiesTest, LeavesViewWithMutatedSiblingAsAlias) {
  Graph graph = make_view_graph();
  graph.values.emplace_back(
      "sibling", ScalarType::Half, std::vector<int64_t>{});
  graph.values[2].alias_id = 0;
  graph.values.emplace_back(
      "updated", ScalarType::Half, std::vector<int64_t>{});
  graph.nodes.push_back(Node{
      .name = "update",
      .target = "test.update",
      .inputs = {{.name = "self", .arg = TensorArg{2}, .mutated = true}},
      .outputs = {{.value_id = 3}},
  });
  graph.schedule.push_back(1);
  graph.rebuild_def_use();

  EXPECT_EQ(materialize_view_copies(graph, {1}), 0);
  EXPECT_EQ(graph.node(0).target, "torch.ops.aten.view.default");
  EXPECT_EQ(graph.value(1).alias_id, 0);
}

TEST(MaterializeViewCopiesTest, IgnoresInvalidOutput) {
  Graph graph = make_view_graph();
  graph.node(0).outputs[0].value_id = kInvalid;

  EXPECT_EQ(materialize_view_copies(graph, {1}), 0);
  EXPECT_EQ(graph.node(0).target, "torch.ops.aten.view.default");
}
// cppcheck-suppress-end syntaxError

} // namespace
} // namespace ptn::vulkan
