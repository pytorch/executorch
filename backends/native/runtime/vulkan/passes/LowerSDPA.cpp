// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/vulkan/passes/LowerSDPA.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <executorch/backends/native/runtime/graph/Argument.h>
#include <executorch/backends/native/runtime/graph/Node.h>
#include <executorch/backends/native/runtime/graph/ScalarType.h>
#include <executorch/backends/native/runtime/graph/Value.h>
#include <executorch/backends/native/runtime/graph/utils/GraphUtils.h>

namespace ptn::vulkan {
namespace {

constexpr std::string_view kAtenSDPA =
    "torch.ops.aten.scaled_dot_product_attention.default";
constexpr const char* kVulkanSDPA = "torch.ops.et_vk.sdpa.default";

const Argument* get_arg(const Node& node, std::string_view name, size_t pos) {
  const auto found = std::ranges::find_if(
      node.inputs,
      [name](const NamedArgument& input) { return input.name == name; });
  if (found != node.inputs.end()) {
    return &found->arg;
  }
  return pos < node.inputs.size() ? &node.inputs[pos].arg : nullptr;
}

std::optional<ValueId> tensor_id(const Argument* arg) {
  if (arg == nullptr || arg->kind() != ArgKind::Tensor ||
      !valid(arg->as_tensor().id)) {
    return std::nullopt;
  }
  return arg->as_tensor().id;
}

// Absent optional arguments take their schema default.
bool is_bool_or_absent(const Argument* arg, bool value) {
  return arg == nullptr ||
      (arg->kind() == ArgKind::Bool && !valid(arg->as_bool().id) &&
       arg->as_bool().value == value);
}

bool is_float_or_absent(const Argument* arg, double value) {
  return arg == nullptr ||
      (arg->kind() == ArgKind::Float && !valid(arg->as_float().id) &&
       arg->as_float().value == value);
}

bool is_float_dtype(ScalarType dtype) {
  return dtype == ScalarType::Float || dtype == ScalarType::Half;
}

ValueId add_tensor(Graph& graph, std::string name, TensorMeta meta) {
  meta.dim_order_hint.clear();
  meta.quant.reset();
  return graph.append_value(Value(std::move(name), std::move(meta)));
}

// Select rather than multiply: 0 * -inf would make kept positions NaN.
ValueId bool_mask_bias(
    Graph& graph,
    NodeId before,
    const std::string& name,
    ValueId mask_id,
    ScalarType dtype) {
  TensorMeta float_meta = graph.value(mask_id).tensor_meta();
  float_meta.dtype = dtype;
  std::vector<ValueId> fills;
  fills.reserve(2);
  for (const auto& [suffix, fill] :
       {std::pair{"_zero", 0.0},
        std::pair{"_neg_inf", -std::numeric_limits<double>::infinity()}}) {
    const ValueId id = add_tensor(
        graph, name + suffix, TensorMeta{.dtype = dtype, .sizes = {1}});
    graph.insert_node_before(
        before,
        Node{
            .name = name + suffix,
            .target = "torch.ops.aten.full.default",
            .inputs =
                {{.name = "size", .arg = IntListArg{{1}, {}}},
                 {.name = "fill_value", .arg = FloatArg{fill}},
                 {.name = "dtype", .arg = ScalarTypeArg{dtype}},
                 {.name = "layout", .arg = NoneArg{}},
                 {.name = "device", .arg = NoneArg{}},
                 {.name = "pin_memory", .arg = NoneArg{}}},
            .outputs = {{.value_id = id}},
        });
    fills.push_back(id);
  }
  const ValueId bias = add_tensor(graph, name + "_bias", float_meta);
  graph.insert_node_before(
      before,
      Node{
          .name = name + "_bias",
          .target = "torch.ops.aten.where.self",
          .inputs =
              {{.name = "condition", .arg = TensorArg{mask_id}},
               {.name = "self", .arg = TensorArg{fills.at(0)}},
               {.name = "other", .arg = TensorArg{fills.at(1)}}},
          .outputs = {{.value_id = bias}},
      });
  return bias;
}

} // namespace

size_t lower_sdpa(Graph& graph) {
  size_t count = 0;
  // Bool-mask conversion appends nodes and values, so iterate a copy of the
  // schedule and re-resolve the node after inserting.
  for (const NodeId node_id : std::vector<NodeId>(graph.schedule)) {
    const Node& node = graph.node(node_id);
    if (!node.is_call() || node.target != kAtenSDPA ||
        node.outputs.size() != 1 || !valid(node.outputs[0].value_id)) {
      continue;
    }

    const Argument* query = get_arg(node, "query", 0);
    const Argument* key = get_arg(node, "key", 1);
    const Argument* value = get_arg(node, "value", 2);
    const Argument* attn_mask = get_arg(node, "attn_mask", 3);
    const Argument* dropout_p = get_arg(node, "dropout_p", 4);
    const Argument* is_causal = get_arg(node, "is_causal", 5);
    const Argument* scale = get_arg(node, "scale", 6);
    const Argument* enable_gqa = get_arg(node, "enable_gqa", 7);

    const std::optional<ValueId> q_id = tensor_id(query);
    const std::optional<ValueId> k_id = tensor_id(key);
    const std::optional<ValueId> v_id = tensor_id(value);
    if (!q_id || !k_id || !v_id || !is_float_or_absent(dropout_p, 0.0) ||
        !is_bool_or_absent(is_causal, false) ||
        !is_bool_or_absent(enable_gqa, false)) {
      continue;
    }

    Argument scale_arg = NoneArg{};
    if (scale != nullptr && scale->kind() == ArgKind::Float &&
        !valid(scale->as_float().id)) {
      scale_arg = *scale;
    } else if (scale != nullptr && scale->kind() != ArgKind::None) {
      continue;
    }

    const Value& q_value = graph.value(*q_id);
    const Value& k_value = graph.value(*k_id);
    const Value& v_value = graph.value(*v_id);
    const Value& out_value = graph.value(node.outputs[0].value_id);
    if (!q_value.is_tensor() || !k_value.is_tensor() || !v_value.is_tensor() ||
        !out_value.is_tensor()) {
      continue;
    }
    // Copies: inserting the mask conversion reallocates the graph's lists.
    const TensorMeta q_meta = q_value.tensor_meta();
    const TensorMeta k_meta = k_value.tensor_meta();
    const TensorMeta v_meta = v_value.tensor_meta();
    const TensorMeta out_meta = out_value.tensor_meta();
    // The fused kernel requires equal head widths and has no GQA broadcast.
    if (q_meta.sizes.size() != 4 || k_meta.sizes.size() != 4 ||
        v_meta.sizes.size() != 4 || out_meta.sizes.size() != 4 ||
        !is_float_dtype(q_meta.dtype) || k_meta.dtype != q_meta.dtype ||
        v_meta.dtype != q_meta.dtype || out_meta.dtype != q_meta.dtype ||
        k_meta.sizes[0] != q_meta.sizes[0] ||
        v_meta.sizes[0] != q_meta.sizes[0] ||
        k_meta.sizes[1] != q_meta.sizes[1] ||
        v_meta.sizes[1] != q_meta.sizes[1] ||
        k_meta.sizes[3] != q_meta.sizes[3] ||
        v_meta.sizes[3] != q_meta.sizes[3] ||
        v_meta.sizes[2] != k_meta.sizes[2] ||
        out_meta.sizes !=
            std::vector<int64_t>{
                q_meta.sizes[0],
                q_meta.sizes[1],
                q_meta.sizes[2],
                v_meta.sizes[3]}) {
      continue;
    }

    // The fused kernel reads the mask as an additive float bias. A float
    // mask passes through; a bool keep-mask is converted; anything else
    // would change semantics, so leave it to fail dispatch by name.
    Argument mask_arg = NoneArg{};
    if (attn_mask != nullptr && attn_mask->kind() == ArgKind::Tensor) {
      const std::optional<ValueId> mask_id = tensor_id(attn_mask);
      if (!mask_id || !graph.value(*mask_id).is_tensor()) {
        continue;
      }
      const TensorMeta& mask_meta = graph.value(*mask_id).tensor_meta();
      const std::vector<int64_t> score_sizes{
          q_meta.sizes[0], q_meta.sizes[1], q_meta.sizes[2], k_meta.sizes[2]};
      if (mask_meta.sizes.empty() || mask_meta.sizes.size() > 4 ||
          mask_meta.sizes.back() != score_sizes.back()) {
        continue;
      }
      bool broadcastable = true;
      const size_t offset = score_sizes.size() - mask_meta.sizes.size();
      for (size_t i = 0; i < mask_meta.sizes.size(); ++i) {
        broadcastable &= mask_meta.sizes[i] == 1 ||
            mask_meta.sizes[i] == score_sizes[offset + i];
      }
      if (!broadcastable) {
        continue;
      }
      const ScalarType mask_dtype = mask_meta.dtype;
      if (is_float_dtype(mask_dtype)) {
        mask_arg = *attn_mask;
      } else if (mask_dtype == ScalarType::Bool) {
        mask_arg = TensorArg{bool_mask_bias(
            graph, node_id, node.name + "_mask", *mask_id, q_meta.dtype)};
      } else {
        continue;
      }
    } else if (attn_mask != nullptr && attn_mask->kind() != ArgKind::None) {
      continue;
    }

    Node& lowered = graph.node(node_id);
    lowered.target = kVulkanSDPA;
    lowered.inputs = {
        {.name = "query", .arg = TensorArg{*q_id}},
        {.name = "key", .arg = TensorArg{*k_id}},
        {.name = "value", .arg = TensorArg{*v_id}},
        {.name = "attn_mask", .arg = mask_arg},
        {.name = "scale", .arg = scale_arg},
    };
    ++count;
  }
  if (count > 0) {
    graph.rebuild_def_use();
    validate_graph(graph);
  }
  return count;
}

} // namespace ptn::vulkan
