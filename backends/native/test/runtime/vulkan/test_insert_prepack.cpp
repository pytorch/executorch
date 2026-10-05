// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/vulkan/passes/InsertPrepack.h>

#include <gtest/gtest.h>

namespace ptn::vulkan {
namespace {

Node placeholder(const char* name, ValueId output) {
  return Node{
      .name = name,
      .op_kind = OpKind::Placeholder,
      .outputs = {{.value_id = output}},
  };
}

Node binary(
    const char* name,
    const char* target,
    ValueId left,
    ValueId right,
    ValueId output) {
  return Node{
      .name = name,
      .target = target,
      .inputs =
          {
              {.name = "self", .arg = TensorArg{left}},
              {.name = "other", .arg = TensorArg{right}},
          },
      .outputs = {{.value_id = output}},
  };
}

Method make_method() {
  Method method;
  method.name = "forward";
  Graph& graph = method.graph;
  graph.values = {
      Value("input", ScalarType::Half, {4}),
      Value("constant", ScalarType::Half, {4}),
      Value("mul", ScalarType::Half, {4}),
      Value("add", ScalarType::Half, {4}),
  };
  graph.values[0].role = ValueRole::UserInput;
  graph.values[1].role = ValueRole::Parameter;
  graph.nodes = {
      placeholder("input", 0),
      placeholder("constant", 1),
      binary("mul", "torch.ops.aten.mul.Tensor", 0, 1, 2),
      binary("add", "torch.ops.aten.add.Tensor", 2, 1, 3),
      Node{
          .name = "output",
          .op_kind = OpKind::Output,
          .inputs = {{.arg = TensorArg{3}}},
      },
  };
  graph.input_ids = {0};
  graph.output_ids = {3};
  graph.initialize_schedule();
  graph.rebuild_def_use();
  method.data_bindings = {{
      .value_id = 1,
      .role = ValueRole::Parameter,
      .key = "constant",
  }};
  return method;
}

// cppcheck-suppress-begin syntaxError
TEST(InsertPrepackTest, SharesOnePrepackAcrossConsumers) {
  Method method = make_method();

  EXPECT_EQ(insert_prepack_nodes(method), 1);

  const Graph& graph = method.graph;
  ASSERT_EQ(graph.values.size(), 5);
  ASSERT_EQ(graph.nodes.size(), 6);
  const Node& prepack = graph.node(5);
  EXPECT_EQ(prepack.target, "torch.ops.et_vk.prepack.default");
  EXPECT_EQ(prepack.input_value_ids(), std::vector<ValueId>({1}));
  EXPECT_EQ(graph.node(2).input_value_ids(), std::vector<ValueId>({0, 4}));
  EXPECT_EQ(graph.node(3).input_value_ids(), std::vector<ValueId>({2, 4}));
  EXPECT_EQ(graph.value(4).role, ValueRole::ConstantTensor);
}

TEST(InsertPrepackTest, LeavesSelfPrepackingOpUntouched) {
  Method method = make_method();
  method.graph.node(2).target = "torch.ops.aten.linear.default";
  method.graph.node(3).target = "torch.ops.et_vk.embedding_q4gsw.default";

  EXPECT_EQ(insert_prepack_nodes(method), 0);
  EXPECT_EQ(method.graph.values.size(), 4);
  EXPECT_EQ(method.graph.nodes.size(), 5);
}

TEST(InsertPrepackTest, LeavesMutableBindingUntouched) {
  Method method = make_method();
  method.data_bindings[0].mutated = true;

  EXPECT_EQ(insert_prepack_nodes(method), 0);
  EXPECT_EQ(method.graph.values.size(), 4);
  EXPECT_EQ(method.graph.nodes.size(), 5);
  EXPECT_EQ(
      method.graph.node(2).input_value_ids(), std::vector<ValueId>({0, 1}));
  EXPECT_EQ(
      method.graph.node(3).input_value_ids(), std::vector<ValueId>({2, 1}));
}
// cppcheck-suppress-end syntaxError

} // namespace
} // namespace ptn::vulkan
