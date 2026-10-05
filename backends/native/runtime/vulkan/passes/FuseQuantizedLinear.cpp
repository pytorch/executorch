// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/vulkan/passes/FuseQuantizedLinear.h>

#include <algorithm>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include <executorch/backends/native/runtime/graph/Argument.h>
#include <executorch/backends/native/runtime/graph/Node.h>
#include <executorch/backends/native/runtime/graph/ScalarType.h>
#include <executorch/backends/native/runtime/graph/utils/GraphUtils.h>

namespace ptn::vulkan {
namespace {

constexpr std::string_view kChooseQParams =
    "torch.ops.torchao.choose_qparams_affine.default";
constexpr std::string_view kQuantize =
    "torch.ops.torchao.quantize_affine.default";
constexpr std::string_view kDequantize =
    "torch.ops.torchao.dequantize_affine.default";
constexpr std::string_view kLinear = "torch.ops.aten.linear.default";
constexpr const char* kDynamicQ4Linear =
    "torch.ops.et_vk.linear_dq8ca_q4gsw.default";

const Argument*
get_arg(const Node& node, std::string_view name, size_t position) {
  const auto found = std::ranges::find_if(
      node.inputs,
      [name](const NamedArgument& input) { return input.name == name; });
  if (found != node.inputs.end()) {
    return &found->arg;
  }
  return position < node.inputs.size() ? &node.inputs[position].arg : nullptr;
}

std::optional<ValueId> tensor_id(const Argument* arg) {
  if (arg == nullptr || arg->kind() != ArgKind::Tensor ||
      !valid(arg->as_tensor().id)) {
    return std::nullopt;
  }
  return arg->as_tensor().id;
}

const Node* producer(const Graph& graph, ValueId id) {
  const NodeId producer_id = graph.value(id).producer_id;
  return valid(producer_id) ? &graph.node(producer_id) : nullptr;
}

bool is_literal_int(const Argument* arg, int64_t expected) {
  return arg != nullptr && arg->kind() == ArgKind::Int &&
      !valid(arg->as_int().id) && arg->as_int().value == expected;
}

bool is_dtype(const Argument* arg, ScalarType expected) {
  return arg != nullptr && arg->kind() == ArgKind::ScalarType &&
      arg->as_scalar_type().value == expected;
}

const IntListArg* literal_int_list(const Argument* arg) {
  if (arg == nullptr || arg->kind() != ArgKind::IntList ||
      !arg->as_int_list().ids.empty()) {
    return nullptr;
  }
  return &arg->as_int_list();
}

bool same_tensor(const Argument* lhs, const Argument* rhs) {
  const std::optional<ValueId> left = tensor_id(lhs);
  const std::optional<ValueId> right = tensor_id(rhs);
  return left.has_value() && left == right;
}

bool solely_consumed_by(const Graph& graph, ValueId value, NodeId consumer) {
  return graph.value(value).consumer_ids == std::vector<NodeId>{consumer};
}

bool is_constant(const Value& value) {
  return value.role == ValueRole::Parameter ||
      value.role == ValueRole::ConstantTensor;
}

int64_t align_up(int64_t value, int64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

// Returns false when the value already carries a different transform.
bool mark_transform(
    Value& value,
    const Q4ConstantTransform& transform,
    TensorMeta transformed_meta) {
  const auto existing = value.attrs.find(kQ4ConstantTransformAttr);
  if (existing != value.attrs.end()) {
    const auto* prior = std::any_cast<Q4ConstantTransform>(&existing->second);
    return prior != nullptr && prior->kind == transform.kind &&
        prior->source_id == transform.source_id &&
        prior->zero_points_id == transform.zero_points_id &&
        prior->group_size == transform.group_size;
  }
  value.attrs.emplace(kQ4ConstantTransformAttr, transform);
  value.tensor_meta() = std::move(transformed_meta);
  return true;
}

struct Match {
  NodeId linear_id;
  NodeId activation_quantize_id;
  NodeId activation_dequantize_id;
  NodeId weight_dequantize_id;
  ValueId activation_id;
  ValueId input_scale_id;
  ValueId input_zero_id;
  ValueId weight_id;
  ValueId weight_scale_id;
  ValueId weight_zero_id;
  int64_t out_channels;
  int64_t in_channels;
  int64_t groups;
  Argument bias;
};

std::optional<Match> match_dynamic_q4_linear(
    const Graph& graph,
    NodeId linear_id) {
  const Node& linear = graph.node(linear_id);
  if (!linear.is_call() || linear.target != kLinear) {
    return std::nullopt;
  }

  const std::optional<ValueId> activation_dq_id =
      tensor_id(get_arg(linear, "input", 0));
  const std::optional<ValueId> weight_dq_id =
      tensor_id(get_arg(linear, "weight", 1));
  if (!activation_dq_id.has_value() || !weight_dq_id.has_value()) {
    return std::nullopt;
  }
  const Node* activation_dq = producer(graph, *activation_dq_id);
  const Node* weight_dq = producer(graph, *weight_dq_id);
  if (activation_dq == nullptr || weight_dq == nullptr ||
      activation_dq->target != kDequantize ||
      weight_dq->target != kDequantize ||
      !solely_consumed_by(graph, *activation_dq_id, linear_id) ||
      !solely_consumed_by(graph, *weight_dq_id, linear_id)) {
    return std::nullopt;
  }

  const std::optional<ValueId> quantized_activation_id =
      tensor_id(get_arg(*activation_dq, "input", 0));
  const std::optional<ValueId> weight_id =
      tensor_id(get_arg(*weight_dq, "input", 0));
  const std::optional<ValueId> weight_scale_id =
      tensor_id(get_arg(*weight_dq, "scale", 2));
  const std::optional<ValueId> weight_zero_id =
      tensor_id(get_arg(*weight_dq, "zero_point", 3));
  if (!quantized_activation_id.has_value() || !weight_id.has_value() ||
      !weight_scale_id.has_value() || !weight_zero_id.has_value()) {
    return std::nullopt;
  }
  const Node* quantize = producer(graph, *quantized_activation_id);
  if (quantize == nullptr || quantize->target != kQuantize ||
      !solely_consumed_by(
          graph,
          *quantized_activation_id,
          graph.value(*activation_dq_id).producer_id)) {
    return std::nullopt;
  }

  const std::optional<ValueId> activation_id =
      tensor_id(get_arg(*quantize, "input", 0));
  const std::optional<ValueId> input_scale_id =
      tensor_id(get_arg(*quantize, "scale", 2));
  const std::optional<ValueId> input_zero_id =
      tensor_id(get_arg(*quantize, "zero_point", 3));
  if (!activation_id.has_value() || !input_scale_id.has_value() ||
      !input_zero_id.has_value() ||
      !same_tensor(
          get_arg(*activation_dq, "scale", 2),
          get_arg(*quantize, "scale", 2)) ||
      !same_tensor(
          get_arg(*activation_dq, "zero_point", 3),
          get_arg(*quantize, "zero_point", 3))) {
    return std::nullopt;
  }

  const Node* choose = producer(graph, *input_scale_id);
  if (choose == nullptr || choose != producer(graph, *input_zero_id) ||
      choose->target != kChooseQParams ||
      !same_tensor(
          get_arg(*choose, "input", 0), get_arg(*quantize, "input", 0))) {
    return std::nullopt;
  }

  const IntListArg* activation_block =
      literal_int_list(get_arg(*quantize, "block_size", 1));
  const IntListArg* activation_dq_block =
      literal_int_list(get_arg(*activation_dq, "block_size", 1));
  const IntListArg* choose_block =
      literal_int_list(get_arg(*choose, "block_size", 2));
  const IntListArg* weight_block =
      literal_int_list(get_arg(*weight_dq, "block_size", 1));
  if (activation_block == nullptr || activation_dq_block == nullptr ||
      choose_block == nullptr || weight_block == nullptr ||
      activation_block->values != activation_dq_block->values ||
      activation_block->values != choose_block->values ||
      weight_block->values.size() != 2 || weight_block->values[0] != 1 ||
      !is_dtype(get_arg(*quantize, "output_dtype", 4), ScalarType::Char) ||
      !is_dtype(get_arg(*activation_dq, "input_dtype", 4), ScalarType::Char) ||
      !is_dtype(get_arg(*weight_dq, "input_dtype", 4), ScalarType::Char) ||
      !is_literal_int(get_arg(*quantize, "quant_min", 5), -128) ||
      !is_literal_int(get_arg(*quantize, "quant_max", 6), 127) ||
      !is_literal_int(get_arg(*activation_dq, "quant_min", 5), -128) ||
      !is_literal_int(get_arg(*activation_dq, "quant_max", 6), 127) ||
      !is_literal_int(get_arg(*weight_dq, "quant_min", 5), -8) ||
      !is_literal_int(get_arg(*weight_dq, "quant_max", 6), 7)) {
    return std::nullopt;
  }

  const TensorMeta& dequantized_weight =
      graph.value(*weight_dq_id).tensor_meta();
  const TensorMeta& scales = graph.value(*weight_scale_id).tensor_meta();
  const Value& weight = graph.value(*weight_id);
  const Value& zero_points = graph.value(*weight_zero_id);
  if (dequantized_weight.sizes.size() != 2 || scales.sizes.size() != 2 ||
      (dequantized_weight.dtype != ScalarType::Float &&
       dequantized_weight.dtype != ScalarType::Half) ||
      scales.dtype != dequantized_weight.dtype || !is_constant(weight) ||
      !is_constant(graph.value(*weight_scale_id)) ||
      !is_constant(zero_points)) {
    return std::nullopt;
  }
  const int64_t out_channels = dequantized_weight.sizes[0];
  const int64_t in_channels = dequantized_weight.sizes[1];
  const int64_t groups = scales.sizes[1];
  const int64_t group_size = weight_block->values[1];
  if (out_channels <= 0 || in_channels <= 0 || groups <= 0 || group_size <= 0 ||
      in_channels % 4 != 0 || group_size % 4 != 0 ||
      in_channels != groups * group_size || scales.sizes[0] != out_channels ||
      activation_block->values.empty() ||
      activation_block->values.back() != in_channels) {
    return std::nullopt;
  }

  const NodeId activation_dq_node_id =
      graph.value(*activation_dq_id).producer_id;
  const NodeId quantize_node_id =
      graph.value(*quantized_activation_id).producer_id;
  const NodeId weight_dq_node_id = graph.value(*weight_dq_id).producer_id;
  const Argument* bias = get_arg(linear, "bias", 2);
  return Match{
      .linear_id = linear_id,
      .activation_quantize_id = quantize_node_id,
      .activation_dequantize_id = activation_dq_node_id,
      .weight_dequantize_id = weight_dq_node_id,
      .activation_id = *activation_id,
      .input_scale_id = *input_scale_id,
      .input_zero_id = *input_zero_id,
      .weight_id = *weight_id,
      .weight_scale_id = *weight_scale_id,
      .weight_zero_id = *weight_zero_id,
      .out_channels = out_channels,
      .in_channels = in_channels,
      .groups = groups,
      .bias = bias == nullptr ? Argument{NoneArg{}} : *bias,
  };
}

bool rewrite_match(
    Method& method,
    const Match& match,
    std::vector<std::pair<ValueId, ValueId>>& sums_by_weight) {
  Graph& graph = method.graph;
  Value& weight = graph.value(match.weight_id);
  if (!mark_transform(
          weight,
          {
              .kind = Q4ConstantTransformKind::PackWeight,
              .source_id = match.weight_id,
              .zero_points_id = match.weight_zero_id,
              .group_size = match.in_channels / match.groups,
          },
          TensorMeta{
              .dtype = ScalarType::Byte,
              .sizes = {match.out_channels, match.in_channels / 2},
          })) {
    return false;
  }

  Value& scales = graph.value(match.weight_scale_id);
  if (!mark_transform(
          scales,
          {
              .kind = Q4ConstantTransformKind::TransposeScales,
              .source_id = match.weight_scale_id,
          },
          TensorMeta{
              .dtype = scales.tensor_meta().dtype,
              .sizes = {match.groups, align_up(match.out_channels, 4)},
          })) {
    return false;
  }

  ValueId sums_id;
  const auto existing_sums = std::ranges::find_if(
      sums_by_weight,
      [&](const auto& entry) { return entry.first == match.weight_id; });
  if (existing_sums != sums_by_weight.end()) {
    sums_id = existing_sums->second;
  } else {
    Value sums(
        weight.name + "_q4_group_sums",
        ScalarType::Int,
        {match.groups, align_up(match.out_channels, 8)});
    sums.role = ValueRole::ConstantTensor;
    sums.attrs.emplace(
        kQ4ConstantTransformAttr,
        Q4ConstantTransform{
            .kind = Q4ConstantTransformKind::WeightSums,
            .source_id = match.weight_id,
            .zero_points_id = match.weight_zero_id,
            .group_size = match.in_channels / match.groups,
        });
    sums_id = graph.append_value(std::move(sums));
    graph.insert_node_before(
        graph.schedule.front(),
        Node{
            .name = graph.value(sums_id).name,
            .op_kind = OpKind::Placeholder,
            .outputs = {{.value_id = sums_id}},
        });
    sums_by_weight.emplace_back(match.weight_id, sums_id);
  }

  Node& linear = graph.node(match.linear_id);
  linear.target = kDynamicQ4Linear;
  linear.inputs = {
      {.name = "input", .arg = TensorArg{match.activation_id}},
      {.name = "input_scale", .arg = TensorArg{match.input_scale_id}},
      {.name = "input_zero_point", .arg = TensorArg{match.input_zero_id}},
      {.name = "weight", .arg = TensorArg{match.weight_id}},
      {.name = "weight_sums", .arg = TensorArg{sums_id}},
      {.name = "weight_scales", .arg = TensorArg{match.weight_scale_id}},
      {.name = "group_size", .arg = IntArg{match.in_channels / match.groups}},
      {.name = "bias", .arg = match.bias},
  };

  graph.rebuild_def_use();
  graph.erase_node(match.activation_dequantize_id);
  graph.erase_node(match.activation_quantize_id);
  graph.erase_node(match.weight_dequantize_id);
  return true;
}

} // namespace

size_t fuse_quantized_linears(Method& method) {
  Graph& graph = method.graph;
  graph.rebuild_def_use();
  const std::vector<NodeId> schedule = graph.schedule;
  std::vector<Match> matches;
  for (const NodeId node_id : schedule) {
    const std::optional<Match> match = match_dynamic_q4_linear(graph, node_id);
    if (match.has_value()) {
      matches.push_back(*match);
    }
  }

  std::vector<std::pair<ValueId, ValueId>> sums_by_weight;
  size_t count = 0;
  for (const Match& match : matches) {
    if (rewrite_match(method, match, sums_by_weight)) {
      // cppcheck-suppress useStlAlgorithm
      ++count;
    }
  }
  validate_graph(graph);
  return count;
}

} // namespace ptn::vulkan
