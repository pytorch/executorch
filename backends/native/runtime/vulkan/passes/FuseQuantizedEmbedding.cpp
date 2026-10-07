// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/vulkan/passes/FuseQuantizedEmbedding.h>

#include <algorithm>
#include <limits>
#include <string_view>
#include <variant>
#include <vector>

#include <executorch/backends/native/runtime/graph/Argument.h>
#include <executorch/backends/native/runtime/graph/Node.h>
#include <executorch/backends/native/runtime/graph/utils/GraphUtils.h>
#include <executorch/backends/native/runtime/vulkan/passes/FuseQuantizedLinear.h>

namespace ptn::vulkan {
namespace {

constexpr std::string_view kEmbedding4Bit =
    "torch.ops.quantized_decomposed.embedding_4bit.dtype";
constexpr std::string_view kEmbedding = "torch.ops.aten.embedding.default";
constexpr const char* kEmbeddingQ4gsw =
    "torch.ops.et_vk.embedding_q4gsw.default";

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

const TensorMeta* tensor_meta(const Graph& graph, const Argument* arg) {
  if (arg == nullptr || arg->kind() != ArgKind::Tensor ||
      !valid(arg->as_tensor().id)) {
    return nullptr;
  }
  const Value& value = graph.value(arg->as_tensor().id);
  return value.is_tensor() ? &value.tensor_meta() : nullptr;
}

bool is_int(const Argument* arg, int64_t value) {
  return arg != nullptr && arg->kind() == ArgKind::Int &&
      !valid(arg->as_int().id) && arg->as_int().value == value;
}

bool is_constant(const Value& value) {
  return value.role == ValueRole::Parameter ||
      value.role == ValueRole::ConstantTensor;
}

bool is_index_dtype(ScalarType dtype) {
  return dtype == ScalarType::Int || dtype == ScalarType::Long;
}

bool rewrite_embedding_4bit(const Graph& graph, Node& node) {
  if (!node.is_call() || node.target != kEmbedding4Bit) {
    return false;
  }

  const Argument* weight = get_arg(node, "weight", 0);
  const Argument* scales = get_arg(node, "weight_scales", 1);
  const Argument* zero_points = get_arg(node, "weight_zero_points", 2);
  const Argument* quant_min = get_arg(node, "weight_quant_min", 3);
  const Argument* quant_max = get_arg(node, "weight_quant_max", 4);
  const Argument* indices = get_arg(node, "indices", 5);
  if (weight == nullptr || scales == nullptr || indices == nullptr ||
      zero_points == nullptr || zero_points->kind() != ArgKind::None ||
      !is_int(quant_min, -8) || !is_int(quant_max, 7)) {
    return false;
  }

  const TensorMeta* weight_meta = tensor_meta(graph, weight);
  const TensorMeta* scales_meta = tensor_meta(graph, scales);
  const TensorMeta* indices_meta = tensor_meta(graph, indices);
  if (weight_meta == nullptr || scales_meta == nullptr ||
      indices_meta == nullptr || weight_meta->sizes.size() != 2 ||
      scales_meta->sizes.size() != 2 ||
      weight_meta->dtype != ScalarType::Byte ||
      !is_index_dtype(indices_meta->dtype) ||
      weight_meta->sizes[0] != scales_meta->sizes[0] ||
      weight_meta->sizes[1] <= 0 || scales_meta->sizes[1] <= 0) {
    return false;
  }

  if (weight_meta->sizes[1] > std::numeric_limits<int64_t>::max() / 2) {
    return false;
  }
  const int64_t embed_dim = weight_meta->sizes[1] * 2;
  const int64_t groups_per_row = scales_meta->sizes[1];
  if (embed_dim % 32 != 0 || embed_dim % groups_per_row != 0 ||
      (embed_dim / groups_per_row) % 4 != 0) {
    return false;
  }

  node.target = kEmbeddingQ4gsw;
  node.inputs = {
      {.name = "weight", .arg = *weight},
      {.name = "weight_scales", .arg = *scales},
      {.name = "group_size", .arg = IntArg{embed_dim / groups_per_row}},
      {.name = "indices", .arg = *indices},
      {.name = "is_linear_weight", .arg = BoolArg{false}},
  };
  return true;
}

// aten.embedding over a constant carrying AffineGroupQuant, which denotes its
// decoded value. The constant is stored in the q4 linear weight format, so the
// kernel reads it through the linear weight prepack.
bool rewrite_quantized_embedding(Method& method, NodeId node_id) {
  Graph& graph = method.graph;
  Node& node = graph.node(node_id);
  if (!node.is_call() || node.target != kEmbedding) {
    return false;
  }
  const Argument* weight = get_arg(node, "weight", 0);
  const Argument* indices = get_arg(node, "indices", 1);
  const TensorMeta* weight_meta = tensor_meta(graph, weight);
  const TensorMeta* indices_meta = tensor_meta(graph, indices);
  if (weight_meta == nullptr || indices_meta == nullptr ||
      !weight_meta->quant.has_value() || !is_index_dtype(indices_meta->dtype)) {
    return false;
  }
  const auto* quant = std::get_if<AffineGroupQuant>(&*weight_meta->quant);
  const ValueId weight_id = weight->as_tensor().id;
  // The transform rewrites the constant for every reader.
  if (quant == nullptr || !is_constant(graph.value(weight_id)) ||
      graph.value(weight_id).consumer_ids != std::vector<NodeId>{node_id} ||
      graph.value(weight_id).attrs.contains(kQ4ConstantTransformAttr) ||
      weight_meta->dtype != ScalarType::Byte ||
      weight_meta->sizes.size() != 2 || quant->quant_min != -8 ||
      quant->quant_max != 7) {
    return false;
  }
  const DataBinding* scales = find_data_binding(method, quant->scale_data_key);
  const DataBinding* zeros = quant->zero_point_data_key.empty()
      ? nullptr
      : find_data_binding(method, quant->zero_point_data_key);
  if (scales == nullptr ||
      (zeros == nullptr && !quant->zero_point_data_key.empty())) {
    return false;
  }

  const int64_t rows = weight_meta->sizes[0];
  const int64_t embed_dim = weight_meta->sizes[1];
  const int64_t group_size = quant->group_size;
  if (rows <= 0 || embed_dim <= 0 || embed_dim % 32 != 0 || group_size <= 0 ||
      group_size % 4 != 0 || embed_dim % group_size != 0) {
    return false;
  }
  const Value& scale_value = graph.value(scales->value_id);
  if (!is_constant(scale_value) ||
      (quant->scale_dtype != ScalarType::Float &&
       quant->scale_dtype != ScalarType::Half) ||
      scale_value.tensor_meta().dtype != quant->scale_dtype ||
      scale_value.tensor_meta().sizes !=
          std::vector<int64_t>{rows, embed_dim / group_size} ||
      !std::ranges::all_of(scale_value.consumer_ids, [node_id](NodeId id) {
        return id == node_id;
      })) {
    return false;
  }

  const Argument indices_arg = *indices;
  Value& weight_value = graph.value(weight_id);
  weight_value.attrs.emplace(
      kQ4ConstantTransformAttr,
      Q4ConstantTransform{
          .kind = Q4ConstantTransformKind::PackWeight,
          .source_id = weight_id,
          .zero_points_id = zeros == nullptr ? kInvalid : zeros->value_id,
          .group_size = group_size,
      });
  weight_value.tensor_meta() = TensorMeta{
      .dtype = ScalarType::Byte,
      .sizes = {rows, embed_dim / 2},
  };
  node.target = kEmbeddingQ4gsw;
  node.inputs = {
      {.name = "weight", .arg = TensorArg{weight_id}},
      {.name = "weight_scales", .arg = TensorArg{scales->value_id}},
      {.name = "group_size", .arg = IntArg{group_size}},
      {.name = "indices", .arg = indices_arg},
      {.name = "is_linear_weight", .arg = BoolArg{true}},
  };
  return true;
}

} // namespace

// cppcheck-suppress unusedFunction
size_t fuse_quantized_embeddings(Method& method) {
  Graph& graph = method.graph;
  graph.rebuild_def_use();
  const size_t count = static_cast<size_t>(
      std::ranges::count_if(graph.schedule, [&](const NodeId node_id) {
        return rewrite_embedding_4bit(graph, graph.node(node_id)) ||
            rewrite_quantized_embedding(method, node_id);
      }));
  if (count > 0) {
    graph.rebuild_def_use();
    validate_graph(graph);
  }
  return count;
}

} // namespace ptn::vulkan
