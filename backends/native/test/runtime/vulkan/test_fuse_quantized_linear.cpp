// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// cppcheck-suppress-file syntaxError

#include <executorch/backends/native/runtime/vulkan/passes/FuseQuantizedLinear.h>

#include <algorithm>
#include <any>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <executorch/backends/native/runtime/graph/Argument.h>
#include <executorch/backends/native/runtime/graph/Node.h>
#include <executorch/backends/native/runtime/graph/ScalarType.h>
#include <executorch/backends/native/runtime/graph/Value.h>

namespace ptn::vulkan {
namespace {

ValueId add_placeholder(
    Graph& graph,
    const std::string& name,
    ScalarType dtype,
    std::vector<int64_t> sizes,
    ValueRole role = ValueRole::Intermediate) {
  const ValueId value_id =
      graph.append_value(Value(name, dtype, std::move(sizes)));
  graph.value(value_id).role = role;
  graph.nodes.push_back(Node{
      .name = name,
      .op_kind = OpKind::Placeholder,
      .outputs = {{.value_id = value_id}},
  });
  return value_id;
}

ValueId add_call(
    Graph& graph,
    const std::string& name,
    const std::string& target,
    std::vector<NamedArgument> inputs,
    ScalarType dtype,
    std::vector<int64_t> sizes) {
  const ValueId value_id =
      graph.append_value(Value(name, dtype, std::move(sizes)));
  graph.nodes.push_back(Node{
      .name = name,
      .target = target,
      .inputs = std::move(inputs),
      .outputs = {{.value_id = value_id}},
  });
  return value_id;
}

Method make_dynamic_q4_linear(ScalarType dtype = ScalarType::Float) {
  Method method;
  method.name = "forward";
  Graph& graph = method.graph;
  const ValueId activation =
      add_placeholder(graph, "activation", dtype, {1, 1, 8});
  const ValueId weight = add_placeholder(
      graph, "weight", ScalarType::Char, {4, 8}, ValueRole::Parameter);
  const ValueId weight_scales = add_placeholder(
      graph, "weight_scales", dtype, {4, 2}, ValueRole::Parameter);
  const ValueId weight_zeros = add_placeholder(
      graph, "weight_zeros", ScalarType::Char, {4, 2}, ValueRole::Parameter);

  const ValueId input_scale =
      graph.append_value(Value("input_scale", ScalarType::Float, {1, 1, 1}));
  const ValueId input_zero =
      graph.append_value(Value("input_zero", ScalarType::Char, {1, 1, 1}));
  graph.nodes.push_back(Node{
      .name = "choose",
      .target = "torch.ops.torchao.choose_qparams_affine.default",
      .inputs =
          {
              {.name = "input", .arg = TensorArg{activation}},
              {.name = "mapping_type", .arg = StringArg{"ASYMMETRIC"}},
              {.name = "block_size", .arg = IntListArg{{1, 1, 8}}},
              {.name = "target_dtype", .arg = ScalarTypeArg{ScalarType::Char}},
              {.name = "quant_min", .arg = IntArg{-128}},
              {.name = "quant_max", .arg = IntArg{127}},
          },
      .outputs = {{.value_id = input_scale}, {.value_id = input_zero}},
  });

  const ValueId quantized = add_call(
      graph,
      "quantized",
      "torch.ops.torchao.quantize_affine.default",
      {
          {.name = "input", .arg = TensorArg{activation}},
          {.name = "block_size", .arg = IntListArg{{1, 1, 8}}},
          {.name = "scale", .arg = TensorArg{input_scale}},
          {.name = "zero_point", .arg = TensorArg{input_zero}},
          {.name = "output_dtype", .arg = ScalarTypeArg{ScalarType::Char}},
          {.name = "quant_min", .arg = IntArg{-128}},
          {.name = "quant_max", .arg = IntArg{127}},
      },
      ScalarType::Char,
      {1, 1, 8});
  const ValueId dequantized = add_call(
      graph,
      "dequantized",
      "torch.ops.torchao.dequantize_affine.default",
      {
          {.name = "input", .arg = TensorArg{quantized}},
          {.name = "block_size", .arg = IntListArg{{1, 1, 8}}},
          {.name = "scale", .arg = TensorArg{input_scale}},
          {.name = "zero_point", .arg = TensorArg{input_zero}},
          {.name = "input_dtype", .arg = ScalarTypeArg{ScalarType::Char}},
          {.name = "quant_min", .arg = IntArg{-128}},
          {.name = "quant_max", .arg = IntArg{127}},
      },
      dtype,
      {1, 1, 8});
  const ValueId dequantized_weight = add_call(
      graph,
      "dequantized_weight",
      "torch.ops.torchao.dequantize_affine.default",
      {
          {.name = "input", .arg = TensorArg{weight}},
          {.name = "block_size", .arg = IntListArg{{1, 4}}},
          {.name = "scale", .arg = TensorArg{weight_scales}},
          {.name = "zero_point", .arg = TensorArg{weight_zeros}},
          {.name = "input_dtype", .arg = ScalarTypeArg{ScalarType::Char}},
          {.name = "quant_min", .arg = IntArg{-8}},
          {.name = "quant_max", .arg = IntArg{7}},
      },
      dtype,
      {4, 8});
  const ValueId output = add_call(
      graph,
      "linear",
      "torch.ops.aten.linear.default",
      {
          {.name = "input", .arg = TensorArg{dequantized}},
          {.name = "weight", .arg = TensorArg{dequantized_weight}},
          {.name = "bias", .arg = NoneArg{}},
      },
      dtype,
      {1, 1, 4});
  graph.nodes.push_back(Node{
      .name = "output",
      .op_kind = OpKind::Output,
      .inputs = {{.arg = TensorArg{output}}},
  });
  graph.input_ids = {activation};
  graph.output_ids = {output};
  graph.initialize_schedule();
  graph.rebuild_def_use();
  return method;
}

const Q4ConstantTransform& transform_of(const Value& value) {
  return std::any_cast<const Q4ConstantTransform&>(
      value.attrs.at(kQ4ConstantTransformAttr));
}

TEST(FuseQuantizedLinearTest, RewritesPortablePatternAtRuntime) {
  Method method = make_dynamic_q4_linear();
  EXPECT_EQ(fuse_quantized_linears(method), 1);

  const Graph& graph = method.graph;
  const auto linear = std::ranges::find_if(graph.schedule, [&](NodeId id) {
    return graph.node(id).target ==
        "torch.ops.et_vk.linear_dq8ca_q4gsw.default";
  });
  ASSERT_NE(linear, graph.schedule.end());
  EXPECT_EQ(graph.node(*linear).inputs[0].arg.as_tensor().id, 0);

  for (const NodeId id : graph.schedule) {
    EXPECT_NE(
        graph.node(id).target, "torch.ops.torchao.quantize_affine.default");
    EXPECT_NE(
        graph.node(id).target, "torch.ops.torchao.dequantize_affine.default");
  }

  const Value& weight = graph.value(1);
  EXPECT_EQ(weight.tensor_meta().dtype, ScalarType::Byte);
  EXPECT_EQ(weight.tensor_meta().sizes, (std::vector<int64_t>{4, 4}));
  EXPECT_EQ(transform_of(weight).kind, Q4ConstantTransformKind::PackWeight);

  const Value& scales = graph.value(2);
  EXPECT_EQ(scales.tensor_meta().sizes, (std::vector<int64_t>{2, 4}));
  EXPECT_EQ(
      transform_of(scales).kind, Q4ConstantTransformKind::TransposeScales);

  const ValueId sums_id = graph.node(*linear).inputs[4].arg.as_tensor().id;
  const Value& sums = graph.value(sums_id);
  EXPECT_EQ(sums.tensor_meta().dtype, ScalarType::Int);
  EXPECT_EQ(sums.tensor_meta().sizes, (std::vector<int64_t>{2, 8}));
  EXPECT_EQ(transform_of(sums).kind, Q4ConstantTransformKind::WeightSums);
}

TEST(FuseQuantizedLinearTest, RejectsNonQ4WeightRange) {
  Method method = make_dynamic_q4_linear();
  Node& weight_dequant = method.graph.node(7);
  weight_dequant.inputs[5].arg = IntArg{-128};

  EXPECT_EQ(fuse_quantized_linears(method), 0);
  EXPECT_NE(
      method.graph.node(8).target,
      "torch.ops.et_vk.linear_dq8ca_q4gsw.default");
}

TEST(FuseQuantizedLinearTest, RejectsGroupSizeNotMultipleOfFour) {
  Method method = make_dynamic_q4_linear();
  method.graph.node(7).inputs[1].arg = IntListArg{{1, 2}};
  method.graph.values[2].tensor_meta().sizes = {4, 4};

  EXPECT_EQ(fuse_quantized_linears(method), 0);
}

TEST(FuseQuantizedLinearTest, PreservesHalfPrecisionScales) {
  Method method = make_dynamic_q4_linear(ScalarType::Half);

  EXPECT_EQ(fuse_quantized_linears(method), 1);
  EXPECT_EQ(method.graph.value(2).tensor_meta().dtype, ScalarType::Half);
}

} // namespace
} // namespace ptn::vulkan
