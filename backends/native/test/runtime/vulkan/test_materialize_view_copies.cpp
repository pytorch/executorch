// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/vulkan/passes/MaterializeViewCopies.h>

#include <string>
#include <utility>
#include <vector>

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

// permute(input, [1, 0]) whose input is static or dynamically sized.
Graph make_permute_graph(bool dynamic_source) {
  Graph graph;
  graph.values = {
      Value("input", ScalarType::Float, {8, 4}),
      Value("permute", ScalarType::Float, {4, 8}),
  };
  if (dynamic_source) {
    graph.values[0].tensor_meta().lower_bounds = {2, 4};
    graph.values[1].tensor_meta().lower_bounds = {4, 2};
  }
  graph.values[1].alias_id = 0;
  graph.nodes = {
      Node{
          .name = "permute",
          .target = "torch.ops.aten.permute.default",
          .inputs =
              {
                  {.name = "self", .arg = TensorArg{0}},
                  {.name = "dims", .arg = IntListArg{.values = {1, 0}}},
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

TEST(MaterializeViewCopiesTest, CopiesPermuteOfDynamicSource) {
  Graph graph = make_permute_graph(/*dynamic_source=*/true);

  EXPECT_EQ(materialize_dynamic_view_copies(graph).size(), 1);
  EXPECT_EQ(graph.node(0).target, "torch.ops.aten.permute_copy.default");
  EXPECT_EQ(graph.value(1).alias_id, kInvalid);
}

TEST(MaterializeViewCopiesTest, CopiesEachMetadataViewOfDynamicSource) {
  const std::vector<std::pair<std::string, std::string>> views = {
      {"torch.ops.aten.alias.default", "torch.ops.aten.clone.default"},
      {"torch.ops.aten.select.int", "torch.ops.aten.select_copy.int"},
      {"torch.ops.aten.squeeze.dims", "torch.ops.aten.squeeze_copy.dims"},
      {"torch.ops.aten.unsqueeze.default",
       "torch.ops.aten.unsqueeze_copy.default"},
  };
  for (const auto& [view, copy] : views) {
    Graph graph = make_permute_graph(/*dynamic_source=*/true);
    graph.node(0).target = view;

    EXPECT_EQ(materialize_dynamic_view_copies(graph).size(), 1) << view;
    EXPECT_EQ(graph.node(0).target, copy);
    EXPECT_EQ(graph.value(1).alias_id, kInvalid) << view;
  }
}

TEST(MaterializeViewCopiesTest, MakesNegativeDimsNonNegative) {
  Graph permute = make_permute_graph(/*dynamic_source=*/true);
  permute.node(0).inputs[1].arg = IntListArg{.values = {-1, 0}};
  Graph squeeze = make_permute_graph(/*dynamic_source=*/true);
  squeeze.node(0).target = "torch.ops.aten.squeeze.dims";
  squeeze.node(0).inputs[1] = {
      .name = "dim", .arg = IntListArg{.values = {-1, -2}}};
  Graph unsqueeze = make_permute_graph(/*dynamic_source=*/true);
  unsqueeze.node(0).target = "torch.ops.aten.unsqueeze.default";
  unsqueeze.node(0).inputs[1] = {.name = "dim", .arg = IntArg{.value = -1}};

  EXPECT_EQ(materialize_dynamic_view_copies(permute).size(), 1);
  EXPECT_EQ(materialize_dynamic_view_copies(squeeze).size(), 1);
  EXPECT_EQ(materialize_dynamic_view_copies(unsqueeze).size(), 1);
  EXPECT_EQ(
      permute.node(0).inputs[1].arg.as_int_list().values,
      (std::vector<int64_t>{1, 0}));
  EXPECT_EQ(
      squeeze.node(0).inputs[1].arg.as_int_list().values,
      (std::vector<int64_t>{1, 0}));
  // unsqueeze's dim counts the dim it inserts.
  EXPECT_EQ(unsqueeze.node(0).inputs[1].arg.as_int().value, 2);
}

TEST(MaterializeViewCopiesTest, GivesCloneTheMemoryFormatAliasLacks) {
  Graph graph = make_permute_graph(/*dynamic_source=*/true);
  graph.node(0).target = "torch.ops.aten.alias.default";
  graph.node(0).inputs.resize(1);

  EXPECT_EQ(materialize_dynamic_view_copies(graph).size(), 1);
  ASSERT_EQ(graph.node(0).inputs.size(), 2);
  EXPECT_EQ(graph.node(0).inputs[1].name, "memory_format");
  EXPECT_EQ(graph.node(0).inputs[1].arg.kind(), ArgKind::None);
}

TEST(MaterializeViewCopiesTest, LeavesPermuteOfStaticSourceAsAlias) {
  Graph graph = make_permute_graph(/*dynamic_source=*/false);

  EXPECT_EQ(materialize_dynamic_view_copies(graph).size(), 0);
  EXPECT_EQ(graph.node(0).target, "torch.ops.aten.permute.default");
  EXPECT_EQ(graph.value(1).alias_id, 0);
}

TEST(MaterializeViewCopiesTest, LeavesDynamicViewToItsResizeHook) {
  Graph graph = make_view_graph();
  graph.values[0].tensor_meta().lower_bounds = {1, 1, 32};
  graph.values[1].tensor_meta().lower_bounds = {1, 1, 1, 32};

  EXPECT_EQ(materialize_dynamic_view_copies(graph).size(), 0);
  EXPECT_EQ(graph.node(0).target, "torch.ops.aten.view.default");
}

TEST(MaterializeViewCopiesTest, LeavesMutatedPermuteOfDynamicSourceAsAlias) {
  Graph graph = make_permute_graph(/*dynamic_source=*/true);
  graph.values.emplace_back(
      "updated", ScalarType::Float, std::vector<int64_t>{});
  graph.nodes.push_back(Node{
      .name = "update",
      .target = "test.update",
      .inputs = {{.name = "self", .arg = TensorArg{1}, .mutated = true}},
      .outputs = {{.value_id = 2}},
  });
  graph.schedule.push_back(1);
  graph.rebuild_def_use();

  EXPECT_EQ(materialize_dynamic_view_copies(graph).size(), 0);
  EXPECT_EQ(graph.node(0).target, "torch.ops.aten.permute.default");
}
// cppcheck-suppress-end syntaxError

} // namespace
} // namespace ptn::vulkan
