// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/vulkan/passes/FuseQuantizedEmbedding.h>

#include <any>
#include <limits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <executorch/backends/native/runtime/vulkan/passes/FuseQuantizedLinear.h>

namespace ptn::vulkan {
namespace {

NamedArgument tensor_arg(const char* name, ValueId id) {
  return {.name = name, .arg = TensorArg{id}};
}

Method make_method(int64_t quant_min = -8, int64_t quant_max = 7) {
  Method method;
  Graph& graph = method.graph;
  graph.values = {
      Value("indices", ScalarType::Long, {1, 4}),
      Value("weight", ScalarType::Byte, {128, 32}),
      Value("scales", ScalarType::Half, {128, 2}),
      Value("output", ScalarType::Half, {1, 4, 64}),
  };
  graph.nodes = {
      Node{
          .name = "indices",
          .op_kind = OpKind::Placeholder,
          .outputs = {{.value_id = 0}},
      },
      Node{
          .name = "weight",
          .op_kind = OpKind::Placeholder,
          .outputs = {{.value_id = 1}},
      },
      Node{
          .name = "scales",
          .op_kind = OpKind::Placeholder,
          .outputs = {{.value_id = 2}},
      },
      Node{
          .name = "embedding",
          .target = "torch.ops.quantized_decomposed.embedding_4bit.dtype",
          .inputs =
              {
                  tensor_arg("weight", 1),
                  tensor_arg("weight_scales", 2),
                  {.name = "weight_zero_points", .arg = NoneArg{}},
                  {.name = "weight_quant_min", .arg = IntArg{quant_min}},
                  {.name = "weight_quant_max", .arg = IntArg{quant_max}},
                  tensor_arg("indices", 0),
                  {.name = "dtype", .arg = ScalarTypeArg{ScalarType::Half}},
              },
          .outputs = {{.value_id = 3}},
      },
      Node{
          .name = "output",
          .op_kind = OpKind::Output,
          .inputs = {tensor_arg("", 3)},
      },
  };
  graph.input_ids = {0};
  graph.output_ids = {3};
  graph.initialize_schedule();
  graph.rebuild_def_use();
  return method;
}

// Values: indices 0, weight 1, scales 2, zero points 3, output 4. Nodes:
// placeholders 0-3, embedding 4, output 5.
Method make_direct_method(int64_t embed_dim = 64, int64_t group_size = 32) {
  Method method;
  Graph& graph = method.graph;
  graph.values = {
      Value("indices", ScalarType::Long, {1, 4}),
      Value("weight", ScalarType::Byte, {128, embed_dim}),
      Value("scales", ScalarType::Float, {128, embed_dim / group_size}),
      Value("zeros", ScalarType::Char, {128, embed_dim / group_size}),
      Value("output", ScalarType::Float, {1, 4, embed_dim}),
  };
  graph.value(1).tensor_meta().quant = AffineGroupQuant{
      .scale_data_key = "scales",
      .scale_dtype = ScalarType::Float,
      .quant_min = -8,
      .quant_max = 7,
      .group_size = static_cast<int32_t>(group_size),
      .zero_point_data_key = "zeros",
      .zero_point_dtype = ScalarType::Char,
  };
  for (const ValueId id : {1, 2, 3}) {
    graph.value(id).role = ValueRole::Parameter;
    method.data_bindings.push_back(
        DataBinding{.value_id = id, .key = graph.value(id).name});
  }
  for (const ValueId id : {0, 1, 2, 3}) {
    graph.nodes.push_back(Node{
        .name = graph.value(id).name,
        .op_kind = OpKind::Placeholder,
        .outputs = {{.value_id = id}},
    });
  }
  graph.nodes.push_back(Node{
      .name = "embedding",
      .target = "torch.ops.aten.embedding.default",
      .inputs = {tensor_arg("weight", 1), tensor_arg("indices", 0)},
      .outputs = {{.value_id = 4}},
  });
  graph.nodes.push_back(Node{
      .name = "output",
      .op_kind = OpKind::Output,
      .inputs = {tensor_arg("", 4)},
  });
  graph.input_ids = {0};
  graph.output_ids = {4};
  graph.initialize_schedule();
  graph.rebuild_def_use();
  return method;
}

// cppcheck-suppress-begin syntaxError
TEST(FuseQuantizedEmbeddingTest, RewritesSupportedEmbedding) {
  Method method = make_method();

  EXPECT_EQ(fuse_quantized_embeddings(method), 1);

  const Node& node = method.graph.node(3);
  EXPECT_EQ(node.target, "torch.ops.et_vk.embedding_q4gsw.default");
  ASSERT_EQ(node.inputs.size(), 5);
  EXPECT_EQ(node.inputs[0].arg.as_tensor().id, 1);
  EXPECT_EQ(node.inputs[1].arg.as_tensor().id, 2);
  EXPECT_EQ(node.inputs[2].arg.as_int().value, 32);
  EXPECT_EQ(node.inputs[3].arg.as_tensor().id, 0);
  EXPECT_FALSE(node.inputs[4].arg.as_bool().value);
}

TEST(FuseQuantizedEmbeddingTest, RejectsUnsupportedQuantization) {
  Method method = make_method(-7, 7);

  EXPECT_EQ(fuse_quantized_embeddings(method), 0);
  EXPECT_EQ(
      method.graph.node(3).target,
      "torch.ops.quantized_decomposed.embedding_4bit.dtype");
}

TEST(FuseQuantizedEmbeddingTest, RejectsIncompatibleShapes) {
  Method method = make_method();
  Graph& graph = method.graph;
  graph.value(2) = Value("scales", ScalarType::Half, {128, 3});
  graph.rebuild_def_use();

  EXPECT_EQ(fuse_quantized_embeddings(method), 0);
}

TEST(FuseQuantizedEmbeddingTest, RejectsGroupSmallerThanShaderVector) {
  Method method = make_method();
  Graph& graph = method.graph;
  graph.value(2) = Value("scales", ScalarType::Half, {128, 32});
  graph.rebuild_def_use();

  EXPECT_EQ(fuse_quantized_embeddings(method), 0);
}

TEST(FuseQuantizedEmbeddingTest, RejectsNonIntegerIndices) {
  Method method = make_method();
  Graph& graph = method.graph;
  graph.value(0) = Value("indices", ScalarType::Float, {1, 4});
  graph.rebuild_def_use();

  EXPECT_EQ(fuse_quantized_embeddings(method), 0);
}

TEST(FuseQuantizedEmbeddingTest, RejectsEmbeddingDimensionOverflow) {
  Method method = make_method();
  Graph& graph = method.graph;
  graph.value(1) = Value(
      "weight",
      ScalarType::Byte,
      {128, std::numeric_limits<int64_t>::max() / 2 + 1});
  graph.rebuild_def_use();

  EXPECT_EQ(fuse_quantized_embeddings(method), 0);
  EXPECT_EQ(
      graph.node(3).target,
      "torch.ops.quantized_decomposed.embedding_4bit.dtype");
}

TEST(FuseQuantizedEmbeddingTest, RewritesDirectlyQuantizedWeight) {
  Method method = make_direct_method();

  EXPECT_EQ(fuse_quantized_embeddings(method), 1);

  const Graph& graph = method.graph;
  const Node& node = graph.node(4);
  EXPECT_EQ(node.target, "torch.ops.et_vk.embedding_q4gsw.default");
  ASSERT_EQ(node.inputs.size(), 5);
  EXPECT_EQ(node.inputs[0].arg.as_tensor().id, 1);
  EXPECT_EQ(node.inputs[1].arg.as_tensor().id, 2);
  EXPECT_EQ(node.inputs[2].arg.as_int().value, 32);
  EXPECT_EQ(node.inputs[3].arg.as_tensor().id, 0);
  EXPECT_TRUE(node.inputs[4].arg.as_bool().value);

  const Value& weight = graph.value(1);
  EXPECT_FALSE(weight.tensor_meta().quant.has_value());
  EXPECT_EQ(weight.tensor_meta().sizes, (std::vector<int64_t>{128, 32}));
  const auto& transform = std::any_cast<const Q4ConstantTransform&>(
      weight.attrs.at(kQ4ConstantTransformAttr));
  EXPECT_EQ(transform.kind, Q4ConstantTransformKind::PackWeight);
  EXPECT_EQ(transform.zero_points_id, 3);
  EXPECT_EQ(transform.group_size, 32);
  EXPECT_FALSE(graph.value(2).attrs.contains(kQ4ConstantTransformAttr));
}

TEST(FuseQuantizedEmbeddingTest, RejectsDirectWeightWithUnboundScales) {
  Method method = make_direct_method();
  method.data_bindings.erase(method.data_bindings.begin() + 1);

  EXPECT_EQ(fuse_quantized_embeddings(method), 0);
  EXPECT_TRUE(method.graph.value(1).tensor_meta().quant.has_value());
}

TEST(FuseQuantizedEmbeddingTest, RejectsDirectWeightShapesTheKernelCannotRead) {
  for (const auto [embed_dim, group_size] :
       {std::pair<int64_t, int64_t>{48, 16}, {64, 2}}) {
    Method method = make_direct_method(embed_dim, group_size);

    EXPECT_EQ(fuse_quantized_embeddings(method), 0);
  }
}

TEST(FuseQuantizedEmbeddingTest, KeepsDirectWeightReadByAnotherOp) {
  Method method = make_direct_method();
  Graph& graph = method.graph;
  const ValueId copy =
      graph.append_value(Value("copy", ScalarType::Byte, {128, 64}));
  graph.insert_node_before(
      graph.schedule.back(),
      Node{
          .name = "copy",
          .target = "torch.ops.aten.clone.default",
          .inputs = {tensor_arg("self", 1)},
          .outputs = {{.value_id = copy}},
      });
  graph.rebuild_def_use();

  EXPECT_EQ(fuse_quantized_embeddings(method), 0);
  EXPECT_FALSE(graph.value(1).attrs.contains(kQ4ConstantTransformAttr));
}
// cppcheck-suppress-end syntaxError

} // namespace
} // namespace ptn::vulkan
