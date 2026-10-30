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
#include <variant>
#include <vector>

#include <executorch/backends/native/runtime/graph/Argument.h>
#include <executorch/backends/native/runtime/graph/Node.h>
#include <executorch/backends/native/runtime/graph/ScalarType.h>
#include <executorch/backends/native/runtime/graph/utils/GraphUtils.h>

namespace ptn::vulkan {
namespace {

int32_t packed_q4_group_sum(const uint8_t* row, size_t start, size_t count) {
  const size_t end = start + count;
  size_t col = start;
  int32_t sum = 0;
  if (col % 2 != 0) {
    sum += row[col / 2] >> 4;
    ++col;
  }
  const uint8_t* bytes = row + col / 2;
  const size_t whole_bytes = (end - col) / 2;
  for (size_t i = 0; i < whole_bytes; ++i) {
    sum += (bytes[i] & 0x0F) + (bytes[i] >> 4);
  }
  col += 2 * whole_bytes;
  if (col < end) {
    sum += row[col / 2] & 0x0F;
  }
  return sum - 8 * static_cast<int32_t>(count);
}

int32_t int8_group_sum(const uint8_t* row, size_t start, size_t count) {
  int32_t sum = 0;
  for (size_t col = start; col < start + count; ++col) {
    sum += static_cast<int8_t>(row[col]);
  }
  return sum;
}

constexpr std::string_view kChooseQParams =
    "torch.ops.torchao.choose_qparams_affine.default";
constexpr std::string_view kQuantize =
    "torch.ops.torchao.quantize_affine.default";
constexpr std::string_view kDequantize =
    "torch.ops.torchao.dequantize_affine.default";
constexpr std::string_view kLinear = "torch.ops.aten.linear.default";
constexpr const char* kDynamicQ4Linear =
    "torch.ops.et_vk.linear_dq8ca_q4gsw.default";
constexpr const char* kWeightOnlyQ4Linear =
    "torch.ops.et_vk.linear_q4gsw.default";

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

// A q4 weight as seen by one linear: either the output of an explicit
// dequantize_affine node, or a constant carrying AffineGroupQuant that the
// linear reads directly and that denotes its decoded value.
struct WeightMatch {
  NodeId dequantize_id = kInvalid; // kInvalid when read directly
  ValueId weight_id = kInvalid;
  ValueId scale_id = kInvalid;
  ValueId zero_id = kInvalid; // kInvalid when symmetric
  ScalarType dtype = ScalarType::Float; // decoded weight dtype
  int64_t out_channels = 0;
  int64_t in_channels = 0;
  int64_t group_size = 0;
  int64_t groups = 0;
};

struct DynamicActivation {
  NodeId quantize_id;
  NodeId dequantize_id;
  ValueId input_id;
  ValueId scale_id;
  ValueId zero_id;
};

struct Match {
  NodeId linear_id;
  WeightMatch weight;
  // nullopt when the linear reads a floating-point activation directly.
  std::optional<DynamicActivation> dynamic;
  ValueId activation_id;
  Argument bias;
};

std::optional<WeightMatch> match_dequantized_weight(
    const Graph& graph,
    NodeId linear_id,
    ValueId dequantized_id) {
  const Node* dequantize = producer(graph, dequantized_id);
  if (dequantize == nullptr || dequantize->target != kDequantize ||
      !solely_consumed_by(graph, dequantized_id, linear_id)) {
    return std::nullopt;
  }
  const std::optional<ValueId> weight_id =
      tensor_id(get_arg(*dequantize, "input", 0));
  const std::optional<ValueId> scale_id =
      tensor_id(get_arg(*dequantize, "scale", 2));
  const std::optional<ValueId> zero_id =
      tensor_id(get_arg(*dequantize, "zero_point", 3));
  const IntListArg* block =
      literal_int_list(get_arg(*dequantize, "block_size", 1));
  if (!weight_id.has_value() || !scale_id.has_value() || !zero_id.has_value() ||
      block == nullptr || block->values.size() != 2 || block->values[0] != 1 ||
      !is_dtype(get_arg(*dequantize, "input_dtype", 4), ScalarType::Char) ||
      !is_literal_int(get_arg(*dequantize, "quant_min", 5), -8) ||
      !is_literal_int(get_arg(*dequantize, "quant_max", 6), 7) ||
      !is_constant(graph.value(*weight_id)) ||
      !is_constant(graph.value(*zero_id))) {
    return std::nullopt;
  }
  const TensorMeta& decoded = graph.value(dequantized_id).tensor_meta();
  if (decoded.sizes.size() != 2) {
    return std::nullopt;
  }
  return WeightMatch{
      .dequantize_id = graph.value(dequantized_id).producer_id,
      .weight_id = *weight_id,
      .scale_id = *scale_id,
      .zero_id = *zero_id,
      .dtype = decoded.dtype,
      .out_channels = decoded.sizes[0],
      .in_channels = decoded.sizes[1],
      .group_size = block->values[1],
  };
}

std::optional<WeightMatch> match_quantized_weight(
    const Method& method,
    ValueId weight_id) {
  const Value& weight = method.graph.value(weight_id);
  const TensorMeta& meta = weight.tensor_meta();
  const auto* quant = meta.quant.has_value()
      ? std::get_if<AffineGroupQuant>(&*meta.quant)
      : nullptr;
  if (quant == nullptr || !is_constant(weight) ||
      meta.dtype != ScalarType::Byte || meta.sizes.size() != 2 ||
      quant->quant_min != -8 || quant->quant_max != 7) {
    return std::nullopt;
  }
  const DataBinding* scales = find_data_binding(method, quant->scale_data_key);
  const DataBinding* zeros = quant->zero_point_data_key.empty()
      ? nullptr
      : find_data_binding(method, quant->zero_point_data_key);
  if (scales == nullptr ||
      (zeros == nullptr && !quant->zero_point_data_key.empty())) {
    return std::nullopt;
  }
  return WeightMatch{
      .weight_id = weight_id,
      .scale_id = scales->value_id,
      .zero_id = zeros == nullptr ? kInvalid : zeros->value_id,
      .dtype = quant->scale_dtype,
      .out_channels = meta.sizes[0],
      .in_channels = meta.sizes[1],
      .group_size = quant->group_size,
  };
}

std::optional<WeightMatch>
match_q4_weight(const Method& method, NodeId linear_id, ValueId weight_id) {
  const Graph& graph = method.graph;
  std::optional<WeightMatch> match = producer(graph, weight_id) != nullptr &&
          producer(graph, weight_id)->target == kDequantize
      ? match_dequantized_weight(graph, linear_id, weight_id)
      : match_quantized_weight(method, weight_id);
  if (!match.has_value() || match->out_channels <= 0 ||
      match->in_channels <= 0 || match->group_size <= 0 ||
      match->in_channels % 4 != 0 || match->group_size % 4 != 0 ||
      match->in_channels % match->group_size != 0) {
    return std::nullopt;
  }
  match->groups = match->in_channels / match->group_size;
  const Value& scales = graph.value(match->scale_id);
  const TensorMeta& scale_meta = scales.tensor_meta();
  if ((match->dtype != ScalarType::Float && match->dtype != ScalarType::Half) ||
      !is_constant(scales) || scale_meta.dtype != match->dtype ||
      scale_meta.sizes !=
          std::vector<int64_t>{match->out_channels, match->groups}) {
    return std::nullopt;
  }
  return match;
}

std::optional<DynamicActivation> match_dynamic_activation(
    const Graph& graph,
    NodeId linear_id,
    ValueId dequantized_id,
    int64_t in_channels) {
  const Node* dequantize = producer(graph, dequantized_id);
  if (dequantize == nullptr || dequantize->target != kDequantize ||
      !solely_consumed_by(graph, dequantized_id, linear_id)) {
    return std::nullopt;
  }
  const std::optional<ValueId> quantized_id =
      tensor_id(get_arg(*dequantize, "input", 0));
  if (!quantized_id.has_value()) {
    return std::nullopt;
  }
  const Node* quantize = producer(graph, *quantized_id);
  if (quantize == nullptr || quantize->target != kQuantize ||
      !solely_consumed_by(
          graph, *quantized_id, graph.value(dequantized_id).producer_id)) {
    return std::nullopt;
  }

  const std::optional<ValueId> input_id =
      tensor_id(get_arg(*quantize, "input", 0));
  const std::optional<ValueId> scale_id =
      tensor_id(get_arg(*quantize, "scale", 2));
  const std::optional<ValueId> zero_id =
      tensor_id(get_arg(*quantize, "zero_point", 3));
  if (!input_id.has_value() || !scale_id.has_value() || !zero_id.has_value() ||
      !same_tensor(
          get_arg(*dequantize, "scale", 2), get_arg(*quantize, "scale", 2)) ||
      !same_tensor(
          get_arg(*dequantize, "zero_point", 3),
          get_arg(*quantize, "zero_point", 3))) {
    return std::nullopt;
  }

  const Node* choose = producer(graph, *scale_id);
  if (choose == nullptr || choose != producer(graph, *zero_id) ||
      choose->target != kChooseQParams ||
      !same_tensor(
          get_arg(*choose, "input", 0), get_arg(*quantize, "input", 0))) {
    return std::nullopt;
  }

  const IntListArg* block =
      literal_int_list(get_arg(*quantize, "block_size", 1));
  const IntListArg* dequantize_block =
      literal_int_list(get_arg(*dequantize, "block_size", 1));
  const IntListArg* choose_block =
      literal_int_list(get_arg(*choose, "block_size", 2));
  if (block == nullptr || dequantize_block == nullptr ||
      choose_block == nullptr || block->values != dequantize_block->values ||
      block->values != choose_block->values || block->values.empty() ||
      block->values.back() != in_channels ||
      !is_dtype(get_arg(*quantize, "output_dtype", 4), ScalarType::Char) ||
      !is_dtype(get_arg(*dequantize, "input_dtype", 4), ScalarType::Char) ||
      !is_literal_int(get_arg(*quantize, "quant_min", 5), -128) ||
      !is_literal_int(get_arg(*quantize, "quant_max", 6), 127) ||
      !is_literal_int(get_arg(*dequantize, "quant_min", 5), -128) ||
      !is_literal_int(get_arg(*dequantize, "quant_max", 6), 127)) {
    return std::nullopt;
  }
  return DynamicActivation{
      .quantize_id = graph.value(*quantized_id).producer_id,
      .dequantize_id = graph.value(dequantized_id).producer_id,
      .input_id = *input_id,
      .scale_id = *scale_id,
      .zero_id = *zero_id,
  };
}

std::optional<Match> match_q4_linear(const Method& method, NodeId linear_id) {
  const Graph& graph = method.graph;
  const Node& linear = graph.node(linear_id);
  if (!linear.is_call() || linear.target != kLinear) {
    return std::nullopt;
  }
  const std::optional<ValueId> input_id =
      tensor_id(get_arg(linear, "input", 0));
  const std::optional<ValueId> weight_id =
      tensor_id(get_arg(linear, "weight", 1));
  if (!input_id.has_value() || !weight_id.has_value()) {
    return std::nullopt;
  }
  const std::optional<WeightMatch> weight =
      match_q4_weight(method, linear_id, *weight_id);
  if (!weight.has_value()) {
    return std::nullopt;
  }
  const std::optional<DynamicActivation> dynamic = match_dynamic_activation(
      graph, linear_id, *input_id, weight->in_channels);
  // linear_q4gsw computes in the activation dtype and packs N in fours.
  if (!dynamic.has_value() &&
      (graph.value(*input_id).tensor_meta().dtype != weight->dtype ||
       weight->out_channels % 4 != 0)) {
    return std::nullopt;
  }
  const Argument* bias = get_arg(linear, "bias", 2);
  return Match{
      .linear_id = linear_id,
      .weight = *weight,
      .dynamic = dynamic,
      .activation_id = dynamic.has_value() ? dynamic->input_id : *input_id,
      .bias = bias == nullptr ? Argument{NoneArg{}} : *bias,
  };
}

// Constant transforms rewrite a value's metadata for every reader, so a weight
// is fused only if each reader of it and of its scales is fused along with it.
void drop_partially_fused_weights(
    const Graph& graph,
    std::vector<Match>& matches) {
  for (bool changed = true; changed;) {
    std::vector<NodeId> fused;
    for (const Match& match : matches) {
      fused.push_back(match.linear_id);
      if (valid(match.weight.dequantize_id)) {
        fused.push_back(match.weight.dequantize_id);
      }
    }
    const auto all_readers_fused = [&](ValueId id) {
      return std::ranges::all_of(
          graph.value(id).consumer_ids, [&](NodeId consumer) {
            return std::ranges::find(fused, consumer) != fused.end();
          });
    };
    changed = std::erase_if(matches, [&](const Match& match) {
                return !all_readers_fused(match.weight.weight_id) ||
                    !all_readers_fused(match.weight.scale_id);
              }) > 0;
  }
}

bool rewrite_match(
    Method& method,
    const Match& match,
    std::vector<std::pair<ValueId, ValueId>>& sums_by_weight) {
  Graph& graph = method.graph;
  const WeightMatch& w = match.weight;
  Value& weight = graph.value(w.weight_id);
  if (!mark_transform(
          weight,
          {
              .kind = Q4ConstantTransformKind::PackWeight,
              .source_id = w.weight_id,
              .zero_points_id = w.zero_id,
              .group_size = w.group_size,
          },
          TensorMeta{
              .dtype = ScalarType::Byte,
              .sizes = {w.out_channels, w.in_channels / 2},
          })) {
    return false;
  }

  Value& scales = graph.value(w.scale_id);
  if (!mark_transform(
          scales,
          {
              .kind = Q4ConstantTransformKind::TransposeScales,
              .source_id = w.scale_id,
          },
          TensorMeta{
              .dtype = scales.tensor_meta().dtype,
              .sizes = {w.groups, align_up(w.out_channels, 4)},
          })) {
    return false;
  }

  if (!match.dynamic.has_value()) {
    Node& linear = graph.node(match.linear_id);
    linear.target = kWeightOnlyQ4Linear;
    linear.inputs = {
        {.name = "input", .arg = TensorArg{match.activation_id}},
        {.name = "weight", .arg = TensorArg{w.weight_id}},
        {.name = "weight_scales", .arg = TensorArg{w.scale_id}},
        {.name = "group_size", .arg = IntArg{w.group_size}},
        {.name = "bias", .arg = match.bias},
    };
  } else {
    ValueId sums_id;
    const auto existing_sums = std::ranges::find_if(
        sums_by_weight,
        [&](const auto& entry) { return entry.first == w.weight_id; });
    if (existing_sums != sums_by_weight.end()) {
      sums_id = existing_sums->second;
    } else {
      Value sums(
          weight.name + "_q4_group_sums",
          ScalarType::Int,
          {w.groups, align_up(w.out_channels, 8)});
      sums.role = ValueRole::ConstantTensor;
      sums.attrs.emplace(
          kQ4ConstantTransformAttr,
          Q4ConstantTransform{
              .kind = Q4ConstantTransformKind::WeightSums,
              .source_id = w.weight_id,
              .zero_points_id = w.zero_id,
              .group_size = w.group_size,
          });
      sums_id = graph.append_value(std::move(sums));
      graph.insert_node_before(
          graph.schedule.front(),
          Node{
              .name = graph.value(sums_id).name,
              .op_kind = OpKind::Placeholder,
              .outputs = {{.value_id = sums_id}},
          });
      sums_by_weight.emplace_back(w.weight_id, sums_id);
    }

    Node& linear = graph.node(match.linear_id);
    linear.target = kDynamicQ4Linear;
    linear.inputs = {
        {.name = "input", .arg = TensorArg{match.activation_id}},
        {.name = "input_scale", .arg = TensorArg{match.dynamic->scale_id}},
        {.name = "input_zero_point", .arg = TensorArg{match.dynamic->zero_id}},
        {.name = "weight", .arg = TensorArg{w.weight_id}},
        {.name = "weight_sums", .arg = TensorArg{sums_id}},
        {.name = "weight_scales", .arg = TensorArg{w.scale_id}},
        {.name = "group_size", .arg = IntArg{w.group_size}},
        {.name = "bias", .arg = match.bias},
    };
  }

  graph.rebuild_def_use();
  if (match.dynamic.has_value()) {
    graph.erase_node(match.dynamic->dequantize_id);
    graph.erase_node(match.dynamic->quantize_id);
  }
  if (valid(w.dequantize_id)) {
    graph.erase_node(w.dequantize_id);
  }
  return true;
}

} // namespace

std::vector<int32_t> q4_group_sums(
    std::span<const uint8_t> weight,
    const Q4GroupSumsLayout& layout) {
  const size_t groups = layout.cols / layout.group_size;
  const size_t stored_cols = layout.packed ? layout.cols / 2 : layout.cols;
  std::vector<int32_t> sums(groups * layout.output_cols, 0);
  for (size_t row = 0; row < layout.rows; ++row) {
    const uint8_t* row_bytes = weight.data() + row * stored_cols;
    for (size_t group = 0; group < groups; ++group) {
      const size_t start = group * layout.group_size;
      sums[group * layout.output_cols + row] = layout.packed
          ? packed_q4_group_sum(row_bytes, start, layout.group_size)
          : int8_group_sum(row_bytes, start, layout.group_size);
    }
  }
  return sums;
}

size_t fuse_quantized_linears(Method& method) {
  Graph& graph = method.graph;
  graph.rebuild_def_use();
  const std::vector<NodeId> schedule = graph.schedule;
  std::vector<Match> matches;
  for (const NodeId node_id : schedule) {
    const std::optional<Match> match = match_q4_linear(method, node_id);
    if (match.has_value()) {
      matches.push_back(*match);
    }
  }
  drop_partially_fused_weights(graph, matches);

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
