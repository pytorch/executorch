// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// cppcheck-suppress-file syntaxError

#include <executorch/backends/native/runtime/vulkan/passes/FuseQuantizedLinear.h>

#include <algorithm>
#include <any>
#include <cstdint>
#include <string>
#include <string_view>
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

struct Q4LinearSpec {
  ScalarType dtype = ScalarType::Float;
  bool dynamic_activation = true;
  // Read a constant carrying AffineGroupQuant instead of a dequantize output.
  bool direct_weight = false;
  bool symmetric = false;
};

// Node ids with the defaults: placeholders 0-3, choose 4, quantize 5,
// activation dequantize 6, weight dequantize 7, linear 8, output 9.
Method make_q4_linear(const Q4LinearSpec& spec = {}) {
  const ScalarType dtype = spec.dtype;
  Method method;
  method.name = "forward";
  Graph& graph = method.graph;
  const ValueId activation =
      add_placeholder(graph, "activation", dtype, {1, 1, 8});
  const ValueId weight = add_placeholder(
      graph,
      "weight",
      spec.direct_weight ? ScalarType::Byte : ScalarType::Char,
      {4, 8},
      ValueRole::Parameter);
  const ValueId weight_scales = add_placeholder(
      graph, "weight_scales", dtype, {4, 2}, ValueRole::Parameter);
  const ValueId weight_zeros = add_placeholder(
      graph, "weight_zeros", ScalarType::Char, {4, 2}, ValueRole::Parameter);
  for (const ValueId id : {weight, weight_scales, weight_zeros}) {
    method.data_bindings.push_back(DataBinding{
        .value_id = id,
        .role = ValueRole::Parameter,
        .key = graph.value(id).name,
    });
  }
  if (spec.direct_weight) {
    graph.value(weight).tensor_meta().quant = AffineGroupQuant{
        .scale_data_key = "weight_scales",
        .scale_dtype = dtype,
        .quant_min = -8,
        .quant_max = 7,
        .group_size = 4,
        .zero_point_data_key = spec.symmetric ? "" : "weight_zeros",
        .zero_point_dtype = ScalarType::Char,
    };
  }

  ValueId linear_input = activation;
  if (spec.dynamic_activation) {
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
                {.name = "target_dtype",
                 .arg = ScalarTypeArg{ScalarType::Char}},
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
    linear_input = add_call(
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
  }
  ValueId linear_weight = weight;
  if (!spec.direct_weight) {
    linear_weight = add_call(
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
  }
  const ValueId output = add_call(
      graph,
      "linear",
      "torch.ops.aten.linear.default",
      {
          {.name = "input", .arg = TensorArg{linear_input}},
          {.name = "weight", .arg = TensorArg{linear_weight}},
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

const Node* find_target(const Graph& graph, std::string_view target) {
  const auto found = std::ranges::find_if(graph.schedule, [&](NodeId id) {
    return graph.node(id).target == target;
  });
  return found == graph.schedule.end() ? nullptr : &graph.node(*found);
}

bool has_dequantize(const Graph& graph) {
  return find_target(graph, "torch.ops.torchao.dequantize_affine.default") !=
      nullptr;
}

const Q4ConstantTransform& transform_of(const Value& value) {
  return std::any_cast<const Q4ConstantTransform&>(
      value.attrs.at(kQ4ConstantTransformAttr));
}

std::vector<uint8_t> pack_q4(const std::vector<int8_t>& weight) {
  std::vector<uint8_t> packed;
  for (size_t i = 0; i < weight.size(); i += 2) {
    packed.push_back((weight[i] + 8) | ((weight[i + 1] + 8) << 4));
  }
  return packed;
}

TEST(FuseQuantizedLinearTest, RewritesPortablePatternAtRuntime) {
  Method method = make_q4_linear();
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
  Method method = make_q4_linear();
  Node& weight_dequant = method.graph.node(7);
  weight_dequant.inputs[5].arg = IntArg{-128};

  EXPECT_EQ(fuse_quantized_linears(method), 0);
  EXPECT_NE(
      method.graph.node(8).target,
      "torch.ops.et_vk.linear_dq8ca_q4gsw.default");
}

TEST(FuseQuantizedLinearTest, RejectsGroupSizeNotMultipleOfFour) {
  Method method = make_q4_linear();
  method.graph.node(7).inputs[1].arg = IntListArg{{1, 2}};
  method.graph.values[2].tensor_meta().sizes = {4, 4};

  EXPECT_EQ(fuse_quantized_linears(method), 0);
}

TEST(FuseQuantizedLinearTest, PreservesHalfPrecisionScales) {
  Method method = make_q4_linear({.dtype = ScalarType::Half});

  EXPECT_EQ(fuse_quantized_linears(method), 1);
  EXPECT_EQ(method.graph.value(2).tensor_meta().dtype, ScalarType::Half);
}

TEST(FuseQuantizedLinearTest, RewritesDirectWeightWithDynamicActivation) {
  Method method = make_q4_linear({.direct_weight = true});
  EXPECT_EQ(fuse_quantized_linears(method), 1);

  const Graph& graph = method.graph;
  const Node* linear =
      find_target(graph, "torch.ops.et_vk.linear_dq8ca_q4gsw.default");
  ASSERT_NE(linear, nullptr);
  EXPECT_EQ(linear->inputs[3].arg.as_tensor().id, 1);
  EXPECT_FALSE(has_dequantize(graph));

  const Value& weight = graph.value(1);
  EXPECT_FALSE(weight.tensor_meta().quant.has_value());
  EXPECT_EQ(weight.tensor_meta().sizes, (std::vector<int64_t>{4, 4}));
  EXPECT_EQ(transform_of(weight).kind, Q4ConstantTransformKind::PackWeight);
  EXPECT_EQ(transform_of(weight).zero_points_id, 3);
  EXPECT_EQ(transform_of(weight).group_size, 4);
  EXPECT_EQ(
      transform_of(graph.value(2)).kind,
      Q4ConstantTransformKind::TransposeScales);
  const ValueId sums_id = linear->inputs[4].arg.as_tensor().id;
  EXPECT_EQ(
      transform_of(graph.value(sums_id)).kind,
      Q4ConstantTransformKind::WeightSums);
}

TEST(FuseQuantizedLinearTest, RewritesFloatActivationToWeightOnlyKernel) {
  for (const bool direct_weight : {false, true}) {
    for (const ScalarType dtype : {ScalarType::Float, ScalarType::Half}) {
      Method method = make_q4_linear(
          {.dtype = dtype,
           .dynamic_activation = false,
           .direct_weight = direct_weight});
      const size_t values_before = method.graph.values.size();
      EXPECT_EQ(fuse_quantized_linears(method), 1);

      const Graph& graph = method.graph;
      const Node* linear =
          find_target(graph, "torch.ops.et_vk.linear_q4gsw.default");
      ASSERT_NE(linear, nullptr);
      ASSERT_EQ(linear->inputs.size(), 5);
      EXPECT_EQ(linear->inputs[0].arg.as_tensor().id, 0);
      EXPECT_EQ(linear->inputs[1].arg.as_tensor().id, 1);
      EXPECT_EQ(linear->inputs[2].arg.as_tensor().id, 2);
      EXPECT_EQ(linear->inputs[3].arg.as_int().value, 4);
      EXPECT_EQ(linear->inputs[4].arg.kind(), ArgKind::None);
      EXPECT_FALSE(has_dequantize(graph));
      // No weight sums: the kernel does not quantize its activation.
      EXPECT_EQ(graph.values.size(), values_before);

      EXPECT_EQ(
          transform_of(graph.value(1)).kind,
          Q4ConstantTransformKind::PackWeight);
      const Value& scales = graph.value(2);
      EXPECT_EQ(scales.tensor_meta().dtype, dtype);
      EXPECT_EQ(scales.tensor_meta().sizes, (std::vector<int64_t>{2, 4}));
    }
  }
}

TEST(FuseQuantizedLinearTest, SymmetricDirectWeightHasNoZeroPoints) {
  Method method = make_q4_linear({.direct_weight = true, .symmetric = true});

  EXPECT_EQ(fuse_quantized_linears(method), 1);
  EXPECT_EQ(transform_of(method.graph.value(1)).zero_points_id, kInvalid);
}

TEST(FuseQuantizedLinearTest, RejectsDirectWeightWithUnboundScales) {
  Method method = make_q4_linear({.direct_weight = true});
  method.data_bindings.erase(method.data_bindings.begin() + 1);

  EXPECT_EQ(fuse_quantized_linears(method), 0);
  EXPECT_TRUE(method.graph.value(1).tensor_meta().quant.has_value());
}

TEST(FuseQuantizedLinearTest, RejectsDirectWeightOutsideQ4Range) {
  Method method = make_q4_linear({.direct_weight = true});
  std::get<AffineGroupQuant>(*method.graph.value(1).tensor_meta().quant)
      .quant_max = 15;

  EXPECT_EQ(fuse_quantized_linears(method), 0);
}

TEST(FuseQuantizedLinearTest, RejectsFloatActivationOfAnotherDtype) {
  Method method =
      make_q4_linear({.dynamic_activation = false, .direct_weight = true});
  method.graph.value(0).tensor_meta().dtype = ScalarType::Half;

  EXPECT_EQ(fuse_quantized_linears(method), 0);
}

TEST(FuseQuantizedLinearTest, KeepsWeightReadByAnUnfusedOp) {
  Method method =
      make_q4_linear({.dynamic_activation = false, .direct_weight = true});
  Graph& graph = method.graph;
  const ValueId copy =
      graph.append_value(Value("copy", ScalarType::Byte, {4, 8}));
  graph.insert_node_before(
      graph.schedule.back(),
      Node{
          .name = "copy",
          .target = "torch.ops.aten.clone.default",
          .inputs = {{.name = "self", .arg = TensorArg{1}}},
          .outputs = {{.value_id = copy}},
      });
  graph.rebuild_def_use();

  EXPECT_EQ(fuse_quantized_linears(method), 0);
  EXPECT_TRUE(graph.value(1).tensor_meta().quant.has_value());
  EXPECT_FALSE(graph.value(1).attrs.contains(kQ4ConstantTransformAttr));
}

TEST(FuseQuantizedLinearTest, SumsQ4WeightGroups) {
  // Two rows of six q4 weights, laid out as [rows, cols].
  const std::vector<int8_t> weight = {-8, 7, 1, 2, -3, 0, 3, 3, 3, -1, -1, -1};
  const std::vector<uint8_t> packed = pack_q4(weight);
  const std::vector<uint8_t> unpacked(weight.begin(), weight.end());
  // [groups, output_cols], with a zero column past the last row.
  const std::vector<int32_t> odd_group_sums = {0, 9, 0, -1, -3, 0};
  const std::vector<int32_t> even_group_sums = {-1, 6, 3, 2, -3, -2};

  EXPECT_EQ(
      q4_group_sums(
          packed,
          {.rows = 2,
           .cols = 6,
           .group_size = 3,
           .output_cols = 3,
           .packed = true}),
      odd_group_sums);
  EXPECT_EQ(
      q4_group_sums(
          unpacked,
          {.rows = 2,
           .cols = 6,
           .group_size = 3,
           .output_cols = 3,
           .packed = false}),
      odd_group_sums);
  EXPECT_EQ(
      q4_group_sums(
          packed,
          {.rows = 2,
           .cols = 6,
           .group_size = 2,
           .output_cols = 2,
           .packed = true}),
      even_group_sums);
}

} // namespace
} // namespace ptn::vulkan
