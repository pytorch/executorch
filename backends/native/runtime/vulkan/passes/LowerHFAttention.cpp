// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/vulkan/passes/LowerHFAttention.h>

#include <algorithm>
#include <any>
#include <array>
#include <cmath>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <executorch/backends/native/runtime/graph/Argument.h>
#include <executorch/backends/native/runtime/graph/Node.h>
#include <executorch/backends/native/runtime/graph/ScalarType.h>
#include <executorch/backends/native/runtime/graph/utils/GraphUtils.h>

namespace ptn::vulkan {
namespace {

constexpr std::string_view kSDPA =
    "torch.ops.aten.scaled_dot_product_attention.default";
constexpr std::string_view kRope = "torch.ops.native.rope.default";
constexpr std::string_view kIndexPut = "torch.ops.aten.index_put_.default";
constexpr std::string_view kAdd = "torch.ops.aten.add.Tensor";
constexpr std::string_view kArange = "torch.ops.aten.arange.start_step";
constexpr std::string_view kLe = "torch.ops.aten.le.Tensor";
constexpr std::array<std::string_view, 2> kPermutes{
    "torch.ops.aten.permute.default",
    "torch.ops.aten.permute_copy.default"};
constexpr std::array<std::string_view, 2> kUnsqueezes{
    "torch.ops.aten.unsqueeze.default",
    "torch.ops.aten.unsqueeze_copy.default"};
constexpr std::array<std::string_view, 2> kViews{
    "torch.ops.aten.view.default",
    "torch.ops.aten.view_copy.default"};
constexpr std::array<std::string_view, 2> kExpands{
    "torch.ops.aten.expand.default",
    "torch.ops.aten.expand_copy.default"};
constexpr std::array<std::string_view, 2> kAliases{
    "torch.ops.aten.alias.default",
    "torch.ops.aten.alias_copy.default"};
constexpr const char* kSelectAsSymInt =
    "torch.ops.et_vk.select_as_symint.default";
constexpr const char* kPrepack = "torch.ops.et_vk.prepack.default";
constexpr const char* kRopeHF = "torch.ops.et_vk.apply_rotary_emb_hf.default";
constexpr const char* kUpdateCache = "torch.ops.llama.update_cache.default";
constexpr const char* kCustomSDPA = "torch.ops.llama.custom_sdpa.default";

template <size_t N>
bool is_any(
    std::string_view target,
    const std::array<std::string_view, N>& names) {
  return std::ranges::find(names, target) != names.end();
}

const NamedArgument*
get_arg(const Node& node, std::string_view name, size_t position) {
  const auto input = std::ranges::find_if(
      node.inputs, [&](const NamedArgument& arg) { return arg.name == name; });
  if (input != node.inputs.end()) {
    return &*input;
  }
  return position < node.inputs.size() ? &node.inputs[position] : nullptr;
}

std::optional<ValueId> tensor_id(const NamedArgument* arg) {
  if (arg == nullptr || arg->arg.kind() != ArgKind::Tensor ||
      !valid(arg->arg.as_tensor().id)) {
    return std::nullopt;
  }
  return arg->arg.as_tensor().id;
}

bool is_int(const NamedArgument* arg, int64_t value) {
  return arg != nullptr && arg->arg.kind() == ArgKind::Int &&
      !valid(arg->arg.as_int().id) && arg->arg.as_int().value == value;
}

bool is_int_list(const NamedArgument* arg, const std::vector<int64_t>& values) {
  return arg != nullptr && arg->arg.kind() == ArgKind::IntList &&
      arg->arg.as_int_list().ids.empty() &&
      arg->arg.as_int_list().values == values;
}

// Absent optional arguments take their schema default.
bool is_int_or_absent(const NamedArgument* arg, int64_t value) {
  return arg == nullptr || is_int(arg, value);
}

bool is_bool_or_absent(const NamedArgument* arg, bool value) {
  return arg == nullptr ||
      (arg->arg.kind() == ArgKind::Bool && !valid(arg->arg.as_bool().id) &&
       arg->arg.as_bool().value == value);
}

std::optional<double> float_or_absent(const NamedArgument* arg, double value) {
  if (arg == nullptr) {
    return value;
  }
  if (arg->arg.kind() != ArgKind::Float || valid(arg->arg.as_float().id)) {
    return std::nullopt;
  }
  return arg->arg.as_float().value;
}

bool is_graph_input(const Graph& graph, ValueId id) {
  return std::ranges::find(graph.input_ids, id) != graph.input_ids.end();
}

const Node* producer(const Graph& graph, ValueId id) {
  const NodeId producer_id = graph.value(id).producer_id;
  if (!valid(producer_id) || !graph.node(producer_id).is_call()) {
    return nullptr;
  }
  return &graph.node(producer_id);
}

// Graph outputs are consumed by the output node; they are replaced separately.
bool only_consumer(const Graph& graph, ValueId id, NodeId consumer) {
  return std::ranges::all_of(
      graph.value(id).consumer_ids, [&](NodeId consumer_id) {
        return consumer_id == consumer || graph.node(consumer_id).is_output();
      });
}

bool outputs_are_dead(const Graph& graph, const Node& node) {
  return !node.output_value_ids().empty() &&
      std::ranges::all_of(node.output_value_ids(), [&](ValueId id) {
        return graph.value(id).consumer_ids.empty() &&
            std::ranges::find(graph.output_ids, id) == graph.output_ids.end();
      });
}

const std::vector<int64_t>& sizes_of(const Graph& graph, ValueId id) {
  return graph.value(id).tensor_meta().sizes;
}

// Returns x for permute(x, [0, 2, 1, 3]).
std::optional<ValueId> unpermuted(const Graph& graph, ValueId id) {
  const Node* node = producer(graph, id);
  if (node == nullptr || !is_any(node->target, kPermutes) ||
      !is_int_list(get_arg(*node, "dims", 1), {0, 2, 1, 3})) {
    return std::nullopt;
  }
  return tensor_id(get_arg(*node, "self", 0));
}

struct Rope {
  NodeId node_id;
  ValueId input; // [1, T, H, D], before the head-major permute
  ValueId position_ids;
  ValueId inv_freq;
  double attention_scale;
};

std::optional<Rope> match_rope(const Graph& graph, ValueId id) {
  const Node* node = producer(graph, id);
  if (node == nullptr || node->target != kRope ||
      !is_bool_or_absent(get_arg(*node, "interleaved", 3), false)) {
    return std::nullopt;
  }
  const std::optional<ValueId> input = tensor_id(get_arg(*node, "input", 0));
  const std::optional<ValueId> position_ids =
      tensor_id(get_arg(*node, "position_ids", 1));
  const std::optional<ValueId> inv_freq =
      tensor_id(get_arg(*node, "inv_freq", 2));
  const std::optional<double> scale =
      float_or_absent(get_arg(*node, "attention_scale", 4), 1.0);
  if (!input || !position_ids || !inv_freq || !scale) {
    return std::nullopt;
  }
  const std::optional<ValueId> unpermuted_input = unpermuted(graph, *input);
  if (!unpermuted_input) {
    return std::nullopt;
  }
  return Rope{
      graph.value(id).producer_id,
      *unpermuted_input,
      *position_ids,
      *inv_freq,
      *scale};
}

struct CacheWrite {
  NodeId node_id;
  ValueId cache;
  ValueId cache_position;
  ValueId values;
};

// index_put_(cache, [None, None, cache_position], values) on a [1, H, S, D]
// buffer.
std::optional<CacheWrite> match_cache_write(const Graph& graph, ValueId id) {
  const Node* node = producer(graph, id);
  if (node == nullptr || node->target != kIndexPut ||
      !is_bool_or_absent(get_arg(*node, "accumulate", 3), false)) {
    return std::nullopt;
  }
  const std::optional<ValueId> cache = tensor_id(get_arg(*node, "self", 0));
  const NamedArgument* indices = get_arg(*node, "indices", 1);
  const std::optional<ValueId> values = tensor_id(get_arg(*node, "values", 2));
  if (!cache || !values || indices == nullptr ||
      indices->arg.kind() != ArgKind::OptionalTensorList ||
      graph.value(*cache).role != ValueRole::Buffer) {
    return std::nullopt;
  }
  const std::vector<ValueId>& ids = indices->arg.as_optional_tensor_list().ids;
  if (ids.size() != 3 || valid(ids[0]) || valid(ids[1]) || !valid(ids[2])) {
    return std::nullopt;
  }
  return CacheWrite{graph.value(id).producer_id, *cache, ids[2], *values};
}

bool is_unsqueezed(
    const Graph& graph,
    ValueId id,
    ValueId source,
    int64_t dim) {
  const Node* node = producer(graph, id);
  return node != nullptr && is_any(node->target, kUnsqueezes) &&
      tensor_id(get_arg(*node, "self", 0)) == source &&
      is_int(get_arg(*node, "dim", 1), dim);
}

// Returns x for op(x, ...) when op is one of names.
template <size_t N>
std::optional<ValueId> operand_of(
    const Graph& graph,
    ValueId id,
    const std::array<std::string_view, N>& names) {
  const Node* node = producer(graph, id);
  if (node == nullptr || !is_any(node->target, names)) {
    return std::nullopt;
  }
  return tensor_id(get_arg(*node, "self", 0));
}

// arange(0, cache_len), optionally plus HF's kv_offset, a one-element constant
// returned in `offset`. HF static caches set it to 0; the pass cannot read it,
// so the rewrite marks it for the engine to check.
bool is_kv_index(
    const Graph& graph,
    ValueId id,
    int64_t cache_len,
    std::optional<ValueId>& offset) {
  const Node* node = producer(graph, id);
  if (node != nullptr && node->target == kAdd) {
    const std::optional<ValueId> self = tensor_id(get_arg(*node, "self", 0));
    offset = tensor_id(get_arg(*node, "other", 1));
    if (!self || !offset || !is_int_or_absent(get_arg(*node, "alpha", 2), 1) ||
        is_graph_input(graph, *offset) || producer(graph, *offset) != nullptr ||
        graph.value(*offset).tensor_meta().numel() != 1) {
      return false;
    }
    node = producer(graph, *self);
  }
  return node != nullptr && node->target == kArange &&
      is_int(get_arg(*node, "start", 0), 0) &&
      is_int(get_arg(*node, "end", 1), cache_len) &&
      is_int_or_absent(get_arg(*node, "step", 2), 1);
}

// HF's static-cache causal mask:
//   expand(unsqueeze(unsqueeze(le(kv_index, view(cache_position, [-1, 1])),
//          0), 1), [1, -1, -1, -1]),
// possibly behind aliases. Sliding-window and padding masks add terms and do
// not match.
bool is_causal_mask(
    const Graph& graph,
    ValueId mask,
    ValueId cache_position,
    int64_t cache_len,
    std::optional<ValueId>& kv_offset) {
  while (const std::optional<ValueId> aliased =
             operand_of(graph, mask, kAliases)) {
    mask = *aliased;
  }
  const Node* expand = producer(graph, mask);
  if (expand == nullptr || !is_any(expand->target, kExpands) ||
      !is_int_list(get_arg(*expand, "size", 1), {1, -1, -1, -1})) {
    return false;
  }
  const std::optional<ValueId> unsqueezed_1 =
      tensor_id(get_arg(*expand, "self", 0));
  const std::optional<ValueId> unsqueezed_0 = unsqueezed_1
      ? operand_of(graph, *unsqueezed_1, kUnsqueezes)
      : std::nullopt;
  const std::optional<ValueId> le = unsqueezed_0
      ? operand_of(graph, *unsqueezed_0, kUnsqueezes)
      : std::nullopt;
  if (!le || !is_unsqueezed(graph, *unsqueezed_1, *unsqueezed_0, 1) ||
      !is_unsqueezed(graph, *unsqueezed_0, *le, 0)) {
    return false;
  }
  const Node* compare = producer(graph, *le);
  if (compare == nullptr || compare->target != kLe) {
    return false;
  }
  const std::optional<ValueId> kv_index =
      tensor_id(get_arg(*compare, "self", 0));
  const std::optional<ValueId> query_index =
      tensor_id(get_arg(*compare, "other", 1));
  const Node* view = query_index ? producer(graph, *query_index) : nullptr;
  return kv_index && view != nullptr && is_any(view->target, kViews) &&
      tensor_id(get_arg(*view, "self", 0)) == cache_position &&
      is_int_list(get_arg(*view, "size", 1), {-1, 1}) &&
      is_kv_index(graph, *kv_index, cache_len, kv_offset);
}

struct Match {
  NodeId sdpa_id;
  NodeId output_permute_id;
  ValueId output; // [1, T, Hq, D], after the token-major permute
  Rope q_rope;
  Rope k_rope;
  CacheWrite k_write;
  CacheWrite v_write;
  ValueId v_input; // [1, T, Hkv, D], before the head-major permute
  std::optional<ValueId> kv_offset;
};

bool compatible_shapes(const Graph& graph, const Match& m) {
  const std::vector<int64_t>& q = sizes_of(graph, m.q_rope.input);
  const std::vector<int64_t>& k = sizes_of(graph, m.k_rope.input);
  const std::vector<int64_t>& v = sizes_of(graph, m.v_input);
  const TensorMeta& kc = graph.value(m.k_write.cache).tensor_meta();
  const TensorMeta& vc = graph.value(m.v_write.cache).tensor_meta();
  const TensorMeta& inv_freq = graph.value(m.q_rope.inv_freq).tensor_meta();
  const ScalarType dtype = graph.value(m.q_rope.input).tensor_meta().dtype;
  if (q.size() != 4 || k.size() != 4 || k != v || kc.sizes != vc.sizes ||
      kc.sizes.size() != 4 || kc.sizes[2] <= 0 || !kc.lower_bounds.empty() ||
      !vc.lower_bounds.empty() || inv_freq.sizes.size() != 1 ||
      inv_freq.sizes[0] <= 0 || inv_freq.dtype != ScalarType::Float ||
      dtype != ScalarType::Float) {
    return false;
  }
  const int64_t head_dim = q[3];
  const int64_t rotary_dim = 2 * inv_freq.sizes[0];
  return q[0] == 1 && k[0] == 1 && q[1] == k[1] && k[3] == head_dim &&
      kc.sizes[0] == 1 && kc.sizes[1] == k[2] && kc.sizes[3] == head_dim &&
      kc.dtype == dtype && vc.dtype == dtype &&
      graph.value(m.k_rope.input).tensor_meta().dtype == dtype &&
      graph.value(m.v_input).tensor_meta().dtype == dtype &&
      rotary_dim <= head_dim && rotary_dim % 8 == 0;
}

bool supported_sdpa_args(const Graph& graph, const Node& sdpa, Match& m) {
  const int64_t head_dim = sizes_of(graph, m.q_rope.input)[3];
  const std::optional<ValueId> mask = tensor_id(get_arg(sdpa, "attn_mask", 3));
  const bool causal_mask = mask &&
      is_causal_mask(graph,
                     *mask,
                     m.k_write.cache_position,
                     sizes_of(graph, m.k_write.cache)[2],
                     m.kv_offset);
  const std::optional<double> dropout =
      float_or_absent(get_arg(sdpa, "dropout_p", 4), 0.0);
  const NamedArgument* scale = get_arg(sdpa, "scale", 6);
  const bool default_scale = scale == nullptr ||
      scale->arg.kind() == ArgKind::None ||
      (scale->arg.kind() == ArgKind::Float &&
       !valid(scale->arg.as_float().id) &&
       std::abs(
           scale->arg.as_float().value -
           1.0 / std::sqrt(static_cast<double>(head_dim))) < 1e-6);
  return causal_mask && dropout == 0.0 && default_scale &&
      is_bool_or_absent(get_arg(sdpa, "is_causal", 5), false);
}

std::optional<Match> match(const Graph& graph, NodeId sdpa_id) {
  const Node& sdpa = graph.node(sdpa_id);
  if (!sdpa.is_call() || sdpa.target != kSDPA || sdpa.outputs.size() != 1) {
    return std::nullopt;
  }
  const std::optional<ValueId> query = tensor_id(get_arg(sdpa, "query", 0));
  const std::optional<ValueId> key = tensor_id(get_arg(sdpa, "key", 1));
  const std::optional<ValueId> value = tensor_id(get_arg(sdpa, "value", 2));
  const ValueId sdpa_out = sdpa.outputs[0].value_id;
  if (!query || !key || !value || !valid(sdpa_out) ||
      graph.value(sdpa_out).consumer_ids.size() != 1) {
    return std::nullopt;
  }
  const NodeId permute_id = graph.value(sdpa_out).consumer_ids.front();
  const Node& permute = graph.node(permute_id);
  if (permute.outputs.size() != 1 ||
      unpermuted(graph, permute.outputs[0].value_id) != sdpa_out) {
    return std::nullopt;
  }

  const std::optional<Rope> q_rope = match_rope(graph, *query);
  const std::optional<CacheWrite> k_write = match_cache_write(graph, *key);
  const std::optional<CacheWrite> v_write = match_cache_write(graph, *value);
  if (!q_rope || !k_write || !v_write) {
    return std::nullopt;
  }
  const std::optional<Rope> k_rope = match_rope(graph, k_write->values);
  const std::optional<ValueId> v_input = unpermuted(graph, v_write->values);
  if (!k_rope || !v_input) {
    return std::nullopt;
  }
  if (!is_graph_input(graph, k_write->cache_position) ||
      v_write->cache_position != k_write->cache_position ||
      k_write->cache == v_write->cache ||
      k_rope->position_ids != q_rope->position_ids ||
      k_rope->inv_freq != q_rope->inv_freq ||
      k_rope->attention_scale != q_rope->attention_scale ||
      !is_unsqueezed(graph, q_rope->position_ids, k_write->cache_position, 0)) {
    return std::nullopt;
  }

  // Each intermediate must feed only this block, since all of it is erased.
  if (!only_consumer(graph, *query, sdpa_id) ||
      !only_consumer(graph, *key, sdpa_id) ||
      !only_consumer(graph, *value, sdpa_id) ||
      !only_consumer(graph, k_write->values, k_write->node_id) ||
      !only_consumer(graph, k_write->cache, k_write->node_id) ||
      !only_consumer(graph, v_write->cache, v_write->node_id)) {
    return std::nullopt;
  }

  Match m{
      .sdpa_id = sdpa_id,
      .output_permute_id = permute_id,
      .output = permute.outputs[0].value_id,
      .q_rope = *q_rope,
      .k_rope = *k_rope,
      .k_write = *k_write,
      .v_write = *v_write,
      .v_input = *v_input,
  };
  if (!compatible_shapes(graph, m) || !supported_sdpa_args(graph, sdpa, m)) {
    return std::nullopt;
  }
  return m;
}

ValueId add_tensor(Graph& graph, std::string name, TensorMeta meta) {
  meta.dim_order_hint.clear();
  return graph.append_value(Value(std::move(name), std::move(meta)));
}

class Lowering {
 public:
  explicit Lowering(Graph& graph) : graph_(graph) {}

  void rewrite(const Match& m) {
    const ValueId start_pos =
        start_pos_for(m.k_write.cache_position, m.sdpa_id);
    const int64_t max_seq_len = sizes_of(graph_, m.k_write.cache)[2];
    mark_cache_position(m.k_write.cache_position, max_seq_len);
    if (m.kv_offset) {
      graph_.value(*m.kv_offset).attrs.try_emplace(kZeroConstantAttr, true);
    }
    const auto [cos, sin] = tables_for(m.q_rope, max_seq_len, m.sdpa_id);
    const IntArg start_pos_arg{.id = start_pos};
    // Copies: appending nodes and values reallocates the graph's lists.
    const std::string name = graph_.node(m.sdpa_id).name;
    const TensorMeta q_meta = graph_.value(m.q_rope.input).tensor_meta();
    const TensorMeta k_meta = graph_.value(m.k_rope.input).tensor_meta();
    const ValueId q = add_tensor(graph_, name + "_rope_q", q_meta);
    const ValueId k = add_tensor(graph_, name + "_rope_k", k_meta);
    insert(
        m.sdpa_id,
        Node{
            .name = name + "_rope",
            .target = kRopeHF,
            .inputs =
                {{.name = "xq", .arg = TensorArg{m.q_rope.input}},
                 {.name = "xk", .arg = TensorArg{m.k_rope.input}},
                 {.name = "freqs_cos", .arg = TensorArg{cos}},
                 {.name = "freqs_sin", .arg = TensorArg{sin}},
                 {.name = "start_pos", .arg = start_pos_arg}},
            .outputs = {{.value_id = q}, {.value_id = k}},
        });
    update_cache(
        m.sdpa_id, name + "_k_update", k, m.k_write.cache, start_pos_arg);
    update_cache(
        m.sdpa_id,
        name + "_v_update",
        m.v_input,
        m.v_write.cache,
        start_pos_arg);

    const ValueId output = add_tensor(
        graph_, name + "_custom_sdpa", graph_.value(m.output).tensor_meta());
    insert(
        m.sdpa_id,
        Node{
            .name = name + "_custom_sdpa",
            .target = kCustomSDPA,
            .inputs =
                {{.name = "query", .arg = TensorArg{q}},
                 {.name = "key", .arg = TensorArg{m.k_write.cache}},
                 {.name = "value", .arg = TensorArg{m.v_write.cache}},
                 {.name = "start_pos", .arg = start_pos_arg},
                 {.name = "attn_mask", .arg = NoneArg{}},
                 {.name = "drpout_p", .arg = FloatArg{.value = 0.0}},
                 {.name = "is_causal", .arg = BoolArg{.value = true}},
                 {.name = "scale", .arg = NoneArg{}}},
            .outputs = {{.value_id = output}},
        });

    for (const ValueId cache : {m.k_write.cache, m.v_write.cache}) {
      TensorMeta& meta = graph_.value(cache).tensor_meta();
      meta.sizes = {meta.sizes[0], meta.sizes[2], meta.sizes[1], meta.sizes[3]};
      meta.dim_order_hint.clear();
      graph_.value(cache).attrs.try_emplace(kWrittenBeforeReadAttr, true);
    }
    const ValueId k_written =
        graph_.node(m.k_write.node_id).outputs.at(0).value_id;
    const ValueId v_written =
        graph_.node(m.v_write.node_id).outputs.at(0).value_id;
    graph_.replace_all_uses(m.output, output);
    graph_.replace_all_uses(k_written, m.k_write.cache, m.k_write.node_id);
    graph_.replace_all_uses(v_written, m.v_write.cache, m.v_write.node_id);
    for (const NodeId node_id :
         {m.output_permute_id,
          m.sdpa_id,
          m.k_write.node_id,
          m.v_write.node_id,
          m.q_rope.node_id,
          m.k_rope.node_id}) {
      graph_.erase_node(node_id);
    }
  }

 private:
  void insert(NodeId before, Node node) {
    graph_.insert_node_before(before, std::move(node));
  }

  void update_cache(
      NodeId before,
      const std::string& name,
      ValueId value,
      ValueId cache,
      const IntArg& start_pos) {
    const ValueId out = graph_.append_value(Value(
        name,
        graph_.value(cache).tensor_meta().dtype,
        std::vector<int64_t>{1}));
    insert(
        before,
        Node{
            .name = name,
            .target = kUpdateCache,
            .inputs =
                {{.name = "value", .arg = TensorArg{value}},
                 {.name = "cache", .arg = TensorArg{cache}, .mutated = true},
                 {.name = "start_pos", .arg = start_pos}},
            .outputs = {{.value_id = out}},
        });
  }

  // Blocks may read caches of different lengths; the shortest bounds them all.
  void mark_cache_position(ValueId source, int64_t cache_len) {
    const auto [it, inserted] = graph_.value(source).attrs.try_emplace(
        kCachePositionAttr, CachePosition{.cache_len = cache_len});
    if (!inserted) {
      auto& bound = std::any_cast<CachePosition&>(it->second);
      bound.cache_len = std::min(bound.cache_len, cache_len);
    }
  }

  // The first block's start_pos is scheduled before it, so later blocks reuse
  // it; the position input is never mutated.
  ValueId start_pos_for(ValueId source, NodeId before) {
    const auto found = start_pos_.find(source);
    if (found != start_pos_.end()) {
      return found->second;
    }
    const std::string name = graph_.value(source).name + "_start_pos";
    const ValueId start_pos = graph_.append_value(Value(name));
    insert(
        before,
        Node{
            .name = name,
            .target = kSelectAsSymInt,
            .inputs =
                {{.name = "x", .arg = TensorArg{source}},
                 {.name = "dim", .arg = IntArg{.value = 0}},
                 {.name = "index", .arg = IntArg{.value = 0}}},
            .outputs = {{.kind = OutputValueKind::Int, .value_id = start_pos}},
        });
    start_pos_.emplace(source, start_pos);
    return start_pos;
  }

  std::pair<ValueId, ValueId>
  tables_for(const Rope& rope, int64_t max_seq_len, NodeId before) {
    const auto key =
        std::make_tuple(rope.inv_freq, rope.attention_scale, max_seq_len);
    const auto found = tables_.find(key);
    if (found != tables_.end()) {
      return found->second;
    }
    const auto pair = std::make_pair(
        table(rope, max_seq_len, false, before),
        table(rope, max_seq_len, true, before));
    tables_.emplace(key, pair);
    return pair;
  }

  ValueId
  table(const Rope& rope, int64_t max_seq_len, bool use_sin, NodeId before) {
    const std::string name = graph_.value(rope.inv_freq).name +
        (use_sin ? "_rope_sin_" : "_rope_cos_") +
        std::to_string(tables_.size());
    const TensorMeta meta{
        .dtype = graph_.value(rope.input).tensor_meta().dtype,
        .sizes =
            {max_seq_len,
             2 * graph_.value(rope.inv_freq).tensor_meta().sizes[0]},
    };
    Value source(name, meta);
    source.role = ValueRole::ConstantTensor;
    source.attrs.emplace(
        kRopeTableAttr,
        RopeTable{
            .inv_freq_id = rope.inv_freq,
            .use_sin = use_sin,
            .attention_scale = rope.attention_scale});
    const ValueId source_id = graph_.append_value(std::move(source));
    graph_.insert_node_before(
        graph_.schedule.front(),
        Node{
            .name = name,
            .op_kind = OpKind::Placeholder,
            .outputs = {{.value_id = source_id}},
        });
    // ET-VK's rotary kernel reads tensors, not tensor references.
    Value prepacked(name + "_vulkan_prepacked", meta);
    prepacked.role = ValueRole::ConstantTensor;
    const ValueId prepacked_id = graph_.append_value(std::move(prepacked));
    insert(
        before,
        Node{
            .name = name + "_vulkan_prepack",
            .target = kPrepack,
            .inputs = {{.name = "self", .arg = TensorArg{source_id}}},
            .outputs = {{.value_id = prepacked_id}},
        });
    return prepacked_id;
  }

  Graph& graph_;
  std::map<ValueId, ValueId> start_pos_;
  std::map<std::tuple<ValueId, double, int64_t>, std::pair<ValueId, ValueId>>
      tables_;
};

bool has_mutated_input(const Node& node) {
  return std::ranges::any_of(
      node.inputs, [](const NamedArgument& arg) { return arg.mutated; });
}

// Drops call nodes whose outputs nothing reads, such as the mask and position
// subgraphs the rewritten blocks no longer use. Nodes that mutate an input or
// have no outputs run for their effects and stay.
void erase_dead_nodes(Graph& graph) {
  bool changed = true;
  while (changed) {
    changed = false;
    graph.rebuild_def_use();
    const std::vector<NodeId> schedule = graph.schedule;
    for (auto it = schedule.rbegin(); it != schedule.rend(); ++it) {
      const Node& node = graph.node(*it);
      if (node.is_call() && !has_mutated_input(node) &&
          outputs_are_dead(graph, node)) {
        graph.erase_node(*it);
        changed = true;
      }
    }
  }
}

} // namespace

// cppcheck-suppress unusedFunction
void check_cache_positions(
    std::span<const int64_t> positions,
    const CachePosition& bound,
    int64_t& rows_written) {
  if (positions.empty()) {
    return;
  }
  const int64_t start = positions.front();
  const auto count = static_cast<int64_t>(positions.size());
  if (start < 0 || count > bound.cache_len || start > bound.cache_len - count) {
    throw std::runtime_error(
        "vulkan: cache_position starting at " + std::to_string(start) +
        " with " + std::to_string(count) + " positions does not fit the " +
        std::to_string(bound.cache_len) + "-row KV cache");
  }
  for (int64_t i = 0; i < count; ++i) {
    if (positions[static_cast<size_t>(i)] != start + i) {
      throw std::runtime_error(
          "vulkan: cache_position must hold contiguous positions");
    }
  }
  if (start > rows_written) {
    throw std::runtime_error(
        "vulkan: cache_position starting at " + std::to_string(start) +
        " skips KV cache rows; only " + std::to_string(rows_written) +
        " have been written");
  }
  rows_written = std::max(rows_written, start + count);
}

// cppcheck-suppress unusedFunction
size_t lower_hf_attention(Graph& graph) {
  graph.rebuild_def_use();
  Lowering lowering(graph);
  size_t count = 0;
  for (const NodeId node_id : std::vector<NodeId>(graph.schedule)) {
    const std::optional<Match> m = match(graph, node_id);
    if (m) {
      lowering.rewrite(*m);
      ++count;
    }
  }
  if (count > 0) {
    erase_dead_nodes(graph);
    validate_graph(graph);
  }
  return count;
}

} // namespace ptn::vulkan
