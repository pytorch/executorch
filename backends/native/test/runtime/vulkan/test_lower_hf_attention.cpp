// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/vulkan/passes/LowerHFAttention.h>

#include <algorithm>
#include <any>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <executorch/backends/native/runtime/graph/Argument.h>
#include <executorch/backends/native/runtime/graph/Node.h>
#include <executorch/backends/native/runtime/graph/ScalarType.h>
#include <executorch/backends/native/runtime/graph/Value.h>
#include <executorch/backends/native/runtime/graph/utils/GraphUtils.h>

namespace ptn::vulkan {
namespace {

constexpr int64_t kTokens = 4;
constexpr int64_t kHeads = 4;
constexpr int64_t kKvHeads = 2;
constexpr int64_t kDim = 16;
constexpr int64_t kCache = 32;

Node placeholder(const char* name, ValueId output) {
  return Node{
      .name = name,
      .op_kind = OpKind::Placeholder,
      .outputs = {{.value_id = output}},
  };
}

Node call(
    const char* name,
    const char* target,
    std::vector<NamedArgument> inputs,
    ValueId output) {
  return Node{
      .name = name,
      .target = target,
      .inputs = std::move(inputs),
      .outputs = {{.value_id = output}},
  };
}

Node permute(const char* name, ValueId input, ValueId output) {
  return call(
      name,
      "torch.ops.aten.permute_copy.default",
      {{.name = "self", .arg = TensorArg{input}},
       {.name = "dims", .arg = IntListArg{{0, 2, 1, 3}}}},
      output);
}

Node rope(const char* name, ValueId input, ValueId output) {
  return call(
      name,
      "torch.ops.native.rope.default",
      {{.name = "input", .arg = TensorArg{input}},
       {.name = "position_ids", .arg = TensorArg{9}},
       {.name = "inv_freq", .arg = TensorArg{4}},
       {.name = "interleaved", .arg = BoolArg{false}},
       {.name = "attention_scale", .arg = FloatArg{1.0}}},
      output);
}

Node cache_write(const char* name, ValueId cache, ValueId values, ValueId out) {
  return call(
      name,
      "torch.ops.aten.index_put_.default",
      {{.name = "self", .arg = TensorArg{cache}, .mutated = true},
       {.name = "indices",
        .arg = OptionalTensorListArg{{kInvalid, kInvalid, 3}}},
       {.name = "values", .arg = TensorArg{values}},
       {.name = "accumulate", .arg = BoolArg{false}}},
      out);
}

// One HF static-cache attention block with its causal mask, as exported.
Graph make_graph() {
  Graph graph;
  graph.values = {
      Value("q", ScalarType::Float, {1, kTokens, kHeads, kDim}),
      Value("k", ScalarType::Float, {1, kTokens, kKvHeads, kDim}),
      Value("v", ScalarType::Float, {1, kTokens, kKvHeads, kDim}),
      Value("cache_position", ScalarType::Long, {kTokens}),
      Value("inv_freq", ScalarType::Float, {kDim / 2}),
      Value("key_cache", ScalarType::Float, {1, kKvHeads, kCache, kDim}),
      Value("value_cache", ScalarType::Float, {1, kKvHeads, kCache, kDim}),
      Value("kv_offset", ScalarType::Long, {}),
      Value("kv_arange", ScalarType::Long, {kCache}),
      Value("position_ids", ScalarType::Long, {1, kTokens}),
      Value("q_perm", ScalarType::Float, {1, kHeads, kTokens, kDim}),
      Value("q_rope", ScalarType::Float, {1, kHeads, kTokens, kDim}),
      Value("k_perm", ScalarType::Float, {1, kKvHeads, kTokens, kDim}),
      Value("k_rope", ScalarType::Float, {1, kKvHeads, kTokens, kDim}),
      Value("v_perm", ScalarType::Float, {1, kKvHeads, kTokens, kDim}),
      Value("k_all", ScalarType::Float, {1, kKvHeads, kCache, kDim}),
      Value("v_all", ScalarType::Float, {1, kKvHeads, kCache, kDim}),
      Value("kv_index", ScalarType::Long, {kCache}),
      Value("query_index", ScalarType::Long, {kTokens, 1}),
      Value("le", ScalarType::Bool, {kTokens, kCache}),
      Value("mask_0", ScalarType::Bool, {1, kTokens, kCache}),
      Value("mask_1", ScalarType::Bool, {1, 1, kTokens, kCache}),
      Value("mask_2", ScalarType::Bool, {1, 1, kTokens, kCache}),
      Value("mask", ScalarType::Bool, {1, 1, kTokens, kCache}),
      Value("attention", ScalarType::Float, {1, kHeads, kTokens, kDim}),
      Value("out", ScalarType::Float, {1, kTokens, kHeads, kDim}),
  };
  graph.value(4).role = ValueRole::ConstantTensor;
  graph.value(5).role = ValueRole::Buffer;
  graph.value(6).role = ValueRole::Buffer;
  graph.value(7).role = ValueRole::Buffer;
  graph.nodes = {
      placeholder("q", 0),
      placeholder("k", 1),
      placeholder("v", 2),
      placeholder("cache_position", 3),
      placeholder("inv_freq", 4),
      placeholder("key_cache", 5),
      placeholder("value_cache", 6),
      placeholder("kv_offset", 7),
      call(
          "position_ids",
          "torch.ops.aten.unsqueeze_copy.default",
          {{.name = "self", .arg = TensorArg{3}},
           {.name = "dim", .arg = IntArg{0}}},
          9),
      permute("q_perm", 0, 10),
      rope("q_rope", 10, 11),
      permute("k_perm", 1, 12),
      rope("k_rope", 12, 13),
      permute("v_perm", 2, 14),
      cache_write("k_all", 5, 13, 15),
      cache_write("v_all", 6, 14, 16),
      call(
          "kv_arange",
          "torch.ops.aten.arange.start_step",
          {{.name = "start", .arg = IntArg{0}},
           {.name = "end", .arg = IntArg{kCache}}},
          8),
      call(
          "kv_index",
          "torch.ops.aten.add.Tensor",
          {{.name = "self", .arg = TensorArg{8}},
           {.name = "other", .arg = TensorArg{7}}},
          17),
      call(
          "query_index",
          "torch.ops.aten.view_copy.default",
          {{.name = "self", .arg = TensorArg{3}},
           {.name = "size", .arg = IntListArg{{-1, 1}}}},
          18),
      call(
          "le",
          "torch.ops.aten.le.Tensor",
          {{.name = "self", .arg = TensorArg{17}},
           {.name = "other", .arg = TensorArg{18}}},
          19),
      call(
          "mask_0",
          "torch.ops.aten.unsqueeze_copy.default",
          {{.name = "self", .arg = TensorArg{19}},
           {.name = "dim", .arg = IntArg{0}}},
          20),
      call(
          "mask_1",
          "torch.ops.aten.unsqueeze_copy.default",
          {{.name = "self", .arg = TensorArg{20}},
           {.name = "dim", .arg = IntArg{1}}},
          21),
      call(
          "mask_2",
          "torch.ops.aten.expand_copy.default",
          {{.name = "self", .arg = TensorArg{21}},
           {.name = "size", .arg = IntListArg{{1, -1, -1, -1}}}},
          22),
      call(
          "mask",
          "torch.ops.aten.alias_copy.default",
          {{.name = "self", .arg = TensorArg{22}}},
          23),
      call(
          "attention",
          "torch.ops.aten.scaled_dot_product_attention.default",
          {{.name = "query", .arg = TensorArg{11}},
           {.name = "key", .arg = TensorArg{15}},
           {.name = "value", .arg = TensorArg{16}},
           {.name = "attn_mask", .arg = TensorArg{23}},
           {.name = "dropout_p", .arg = FloatArg{0.0}},
           {.name = "is_causal", .arg = BoolArg{false}},
           {.name = "scale", .arg = FloatArg{0.25}},
           {.name = "enable_gqa", .arg = BoolArg{true}}},
          24),
      permute("out", 24, 25),
      Node{
          .name = "output",
          .op_kind = OpKind::Output,
          .inputs = {{.name = "output", .arg = TensorArg{25}}},
      },
  };
  for (NodeId id = 0; id < static_cast<NodeId>(graph.nodes.size()); ++id) {
    graph.schedule.push_back(id);
  }
  graph.input_ids = {0, 1, 2, 3};
  graph.output_ids = {25};
  graph.rebuild_def_use();
  return graph;
}

// One Whisper static-cache attention block: learned positions (no RoPE),
// .contiguous() clones after each head-major permute, the broadcast form
// of HF's causal mask, and scale=1 with q pre-scaled upstream.
Graph make_whisper_graph(int64_t cache_len = kCache) {
  Graph graph;
  graph.values = {
      Value("q", ScalarType::Float, {1, kTokens, kHeads, kDim}),
      Value("k", ScalarType::Float, {1, kTokens, kHeads, kDim}),
      Value("v", ScalarType::Float, {1, kTokens, kHeads, kDim}),
      Value("cache_position", ScalarType::Long, {kTokens}),
      Value("key_cache", ScalarType::Float, {1, kHeads, cache_len, kDim}),
      Value("value_cache", ScalarType::Float, {1, kHeads, cache_len, kDim}),
      Value("kv_offset", ScalarType::Long, {}),
      Value("kv_arange", ScalarType::Long, {cache_len}),
      Value("kv_index", ScalarType::Long, {cache_len}),
      Value("q_perm", ScalarType::Float, {1, kHeads, kTokens, kDim}),
      Value("q_clone", ScalarType::Float, {1, kHeads, kTokens, kDim}),
      Value("k_perm", ScalarType::Float, {1, kHeads, kTokens, kDim}),
      Value("k_clone", ScalarType::Float, {1, kHeads, kTokens, kDim}),
      Value("v_perm", ScalarType::Float, {1, kHeads, kTokens, kDim}),
      Value("v_clone", ScalarType::Float, {1, kHeads, kTokens, kDim}),
      Value("k_all", ScalarType::Float, {1, kHeads, cache_len, kDim}),
      Value("v_all", ScalarType::Float, {1, kHeads, cache_len, kDim}),
      Value("kv_u0", ScalarType::Long, {1, cache_len}),
      Value("kv_u1", ScalarType::Long, {1, 1, cache_len}),
      Value("kv_u2", ScalarType::Long, {1, 1, 1, cache_len}),
      Value("q_u0", ScalarType::Long, {1, kTokens}),
      Value("q_u1", ScalarType::Long, {1, 1, kTokens}),
      Value("q_u2", ScalarType::Long, {1, 1, kTokens, 1}),
      Value("le", ScalarType::Bool, {1, 1, kTokens, cache_len}),
      Value("mask", ScalarType::Bool, {1, 1, kTokens, cache_len}),
      Value("attention", ScalarType::Float, {1, kHeads, kTokens, kDim}),
      Value("out", ScalarType::Float, {1, kTokens, kHeads, kDim}),
  };
  graph.value(4).role = ValueRole::Buffer;
  graph.value(5).role = ValueRole::Buffer;
  graph.value(6).role = ValueRole::Buffer;
  const auto clone = [](const char* name, ValueId in, ValueId out) {
    return call(
        name,
        "torch.ops.aten.clone.default",
        {{.name = "self", .arg = TensorArg{in}}},
        out);
  };
  const auto unsqueeze =
      [](const char* name, ValueId in, int64_t dim, ValueId out) {
        return call(
            name,
            "torch.ops.aten.unsqueeze_copy.default",
            {{.name = "self", .arg = TensorArg{in}},
             {.name = "dim", .arg = IntArg{dim}}},
            out);
      };
  graph.nodes = {
      placeholder("q", 0),
      placeholder("k", 1),
      placeholder("v", 2),
      placeholder("cache_position", 3),
      placeholder("key_cache", 4),
      placeholder("value_cache", 5),
      placeholder("kv_offset", 6),
      permute("q_perm", 0, 9),
      clone("q_clone", 9, 10),
      permute("k_perm", 1, 11),
      clone("k_clone", 11, 12),
      permute("v_perm", 2, 13),
      clone("v_clone", 13, 14),
      cache_write("k_all", 4, 12, 15),
      cache_write("v_all", 5, 14, 16),
      call(
          "kv_arange",
          "torch.ops.aten.arange.start_step",
          {{.name = "start", .arg = IntArg{0}},
           {.name = "end", .arg = IntArg{cache_len}}},
          7),
      call(
          "kv_index",
          "torch.ops.aten.add.Tensor",
          {{.name = "self", .arg = TensorArg{7}},
           {.name = "other", .arg = TensorArg{6}}},
          8),
      unsqueeze("kv_u0", 8, 0, 17),
      unsqueeze("kv_u1", 17, 1, 18),
      unsqueeze("kv_u2", 18, 2, 19),
      unsqueeze("q_u0", 3, 0, 20),
      unsqueeze("q_u1", 20, 1, 21),
      unsqueeze("q_u2", 21, 3, 22),
      call(
          "le",
          "torch.ops.aten.le.Tensor",
          {{.name = "self", .arg = TensorArg{19}},
           {.name = "other", .arg = TensorArg{22}}},
          23),
      call(
          "mask",
          "torch.ops.aten.expand_copy.default",
          {{.name = "self", .arg = TensorArg{23}},
           {.name = "size",
            .arg = IntListArg{{1, -1, kTokens, cache_len}, {}}}},
          24),
      call(
          "attention",
          "torch.ops.aten.scaled_dot_product_attention.default",
          {{.name = "query", .arg = TensorArg{10}},
           {.name = "key", .arg = TensorArg{15}},
           {.name = "value", .arg = TensorArg{16}},
           {.name = "attn_mask", .arg = TensorArg{24}},
           {.name = "dropout_p", .arg = FloatArg{0.0}},
           {.name = "is_causal", .arg = BoolArg{false}},
           {.name = "scale", .arg = FloatArg{1.0}},
           {.name = "enable_gqa", .arg = BoolArg{false}}},
          25),
      permute("out", 25, 26),
      Node{
          .name = "output",
          .op_kind = OpKind::Output,
          .inputs = {{.name = "output", .arg = TensorArg{26}}},
      },
  };
  for (NodeId id = 0; id < static_cast<NodeId>(graph.nodes.size()); ++id) {
    graph.schedule.push_back(id);
  }
  graph.input_ids = {0, 1, 2, 3};
  graph.output_ids = {26};
  graph.rebuild_def_use();
  return graph;
}

std::vector<std::string> scheduled_targets(const Graph& graph) {
  std::vector<std::string> targets;
  for (const NodeId id : graph.schedule) {
    if (graph.node(id).is_call()) {
      targets.push_back(graph.node(id).target);
    }
  }
  return targets;
}

const Node& producer_of(const Graph& graph, ValueId id) {
  return graph.node(graph.value(id).producer_id);
}

// cppcheck-suppress-begin syntaxError
TEST(LowerHFAttentionTest, RewritesStaticCacheAttention) {
  Graph graph = make_graph();

  EXPECT_EQ(lower_hf_attention(graph), 1);

  EXPECT_EQ(
      scheduled_targets(graph),
      (std::vector<std::string>{
          "torch.ops.et_vk.select_as_symint.default",
          "torch.ops.et_vk.prepack.default",
          "torch.ops.et_vk.prepack.default",
          "torch.ops.et_vk.apply_rotary_emb_hf.default",
          "torch.ops.llama.update_cache.default",
          "torch.ops.llama.update_cache.default",
          "torch.ops.llama.custom_sdpa.default",
      }));
  EXPECT_EQ(
      graph.value(5).tensor_meta().sizes,
      (std::vector<int64_t>{1, kCache, kKvHeads, kDim}));
  EXPECT_EQ(
      graph.value(6).tensor_meta().sizes,
      (std::vector<int64_t>{1, kCache, kKvHeads, kDim}));

  const ValueId output = graph.output_ids.at(0);
  const Node& sdpa = producer_of(graph, output);
  EXPECT_EQ(sdpa.target, "torch.ops.llama.custom_sdpa.default");
  EXPECT_EQ(
      graph.value(output).tensor_meta().sizes,
      (std::vector<int64_t>{1, kTokens, kHeads, kDim}));
  EXPECT_EQ(sdpa.inputs[1].arg.as_tensor().id, 5);
  EXPECT_EQ(sdpa.inputs[2].arg.as_tensor().id, 6);
  EXPECT_TRUE(sdpa.inputs[6].arg.as_bool().value);
  EXPECT_DOUBLE_EQ(sdpa.inputs[7].arg.as_float().value, 0.25);

  const Node& rotary = producer_of(graph, sdpa.inputs[0].arg.as_tensor().id);
  EXPECT_EQ(rotary.inputs[0].arg.as_tensor().id, 0);
  EXPECT_EQ(rotary.inputs[1].arg.as_tensor().id, 1);
  const ValueId start_pos = rotary.inputs[4].arg.as_int().id;
  EXPECT_EQ(producer_of(graph, start_pos).inputs[0].arg.as_tensor().id, 3);
  EXPECT_EQ(sdpa.inputs[3].arg.as_int().id, start_pos);
  const auto position = graph.value(3).attrs.find(kCachePositionAttr);
  ASSERT_NE(position, graph.value(3).attrs.end());
  EXPECT_EQ(
      std::any_cast<const CachePosition&>(position->second).cache_len, kCache);
  EXPECT_TRUE(graph.value(5).attrs.contains(kWrittenBeforeReadAttr));
  EXPECT_TRUE(graph.value(6).attrs.contains(kWrittenBeforeReadAttr));
  EXPECT_TRUE(graph.value(7).attrs.contains(kZeroConstantAttr));

  for (const size_t i : {size_t{2}, size_t{3}}) {
    const Node& prepack =
        producer_of(graph, rotary.inputs[i].arg.as_tensor().id);
    const Value& table = graph.value(prepack.inputs[0].arg.as_tensor().id);
    EXPECT_EQ(table.role, ValueRole::ConstantTensor);
    EXPECT_EQ(table.tensor_meta().sizes, (std::vector<int64_t>{kCache, kDim}));
    const auto attr = table.attrs.find(kRopeTableAttr);
    ASSERT_NE(attr, table.attrs.end());
    const auto& spec = std::any_cast<const RopeTable&>(attr->second);
    EXPECT_EQ(spec.inv_freq_id, 4);
    EXPECT_EQ(spec.use_sin, i == 3);
    EXPECT_DOUBLE_EQ(spec.attention_scale, 1.0);
  }
}

TEST(LowerHFAttentionTest, RewritesWhisperStaticCacheAttention) {
  Graph graph = make_whisper_graph();

  EXPECT_EQ(lower_hf_attention(graph), 1);

  // No RoPE node and no leftover SDPA / cache writes.
  EXPECT_EQ(
      scheduled_targets(graph),
      (std::vector<std::string>{
          "torch.ops.et_vk.select_as_symint.default",
          "torch.ops.llama.update_cache.default",
          "torch.ops.llama.update_cache.default",
          "torch.ops.llama.custom_sdpa.default",
      }));

  const ValueId output = graph.output_ids.at(0);
  const Node& sdpa = producer_of(graph, output);
  EXPECT_EQ(sdpa.target, "torch.ops.llama.custom_sdpa.default");
  EXPECT_EQ(sdpa.inputs[1].arg.as_tensor().id, 4);
  EXPECT_EQ(sdpa.inputs[2].arg.as_tensor().id, 5);
  EXPECT_EQ(sdpa.inputs[0].arg.as_tensor().id, 0);
  EXPECT_DOUBLE_EQ(sdpa.inputs[7].arg.as_float().value, 1.0);

  EXPECT_EQ(
      graph.value(4).tensor_meta().sizes,
      (std::vector<int64_t>{1, kCache, kHeads, kDim}));
  EXPECT_TRUE(graph.value(3).attrs.contains(kCachePositionAttr));
  EXPECT_TRUE(graph.value(4).attrs.contains(kWrittenBeforeReadAttr));
  EXPECT_NO_THROW(validate_graph(graph));
}

TEST(LowerHFAttentionTest, RewritesClonedRopeOutputs) {
  for (const std::vector<ValueId>& sources :
       {std::vector<ValueId>{11},
        std::vector<ValueId>{13},
        std::vector<ValueId>{11, 13}}) {
    for (const int depth : {1, 2}) {
      Graph graph = make_graph();
      for (const ValueId source : sources) {
        const NodeId consumer = source == 11 ? 24 : 14;
        const size_t input = source == 11 ? 0 : 2;
        ValueId value = source;
        for (int i = 0; i < depth; ++i) {
          const std::string name =
              "clone_" + std::to_string(source) + "_" + std::to_string(i);
          const TensorMeta meta = graph.value(source).tensor_meta();
          const ValueId cloned = graph.append_value(Value(name, meta));
          graph.insert_node_before(
              consumer,
              call(
                  name.c_str(),
                  "torch.ops.aten.clone.default",
                  {{.name = "self", .arg = TensorArg{value}}},
                  cloned));
          value = cloned;
        }
        graph.node(consumer).inputs[input].arg = TensorArg{value};
      }
      graph.rebuild_def_use();

      EXPECT_EQ(lower_hf_attention(graph), 1);
      const auto targets = scheduled_targets(graph);
      EXPECT_EQ(
          std::ranges::count(targets, "torch.ops.native.rope.default"), 0);
      EXPECT_EQ(std::ranges::count(targets, "torch.ops.aten.clone.default"), 0);
      EXPECT_EQ(
          std::ranges::count(targets, "torch.ops.llama.custom_sdpa.default"),
          1);
      EXPECT_NO_THROW(validate_graph(graph));
    }
  }
}

TEST(LowerHFAttentionTest, ChecksBroadcastMaskAxesForFullCachePrefill) {
  for (const bool transposed : {false, true}) {
    SCOPED_TRACE(transposed);
    Graph graph = make_whisper_graph(kTokens);
    if (transposed) {
      graph.node(19).inputs[1].arg = IntArg{3};
      graph.value(19).tensor_meta().sizes = {1, 1, kTokens, 1};
      graph.node(22).inputs[1].arg = IntArg{2};
      graph.value(22).tensor_meta().sizes = {1, 1, 1, kTokens};
    }
    const auto schedule = graph.schedule;

    EXPECT_EQ(lower_hf_attention(graph), transposed ? 0 : 1);
    if (transposed) {
      EXPECT_EQ(graph.schedule, schedule);
      EXPECT_EQ(
          graph.node(25).target,
          "torch.ops.aten.scaled_dot_product_attention.default");
    }
    EXPECT_NO_THROW(validate_graph(graph));
  }
}

TEST(LowerHFAttentionTest, RewritesMaskWithInterleavedExpands) {
  Graph graph = make_graph();
  graph.node(21) = call(
      "mask_1",
      "torch.ops.aten.expand_copy.default",
      {{.name = "self", .arg = TensorArg{20}},
       {.name = "size", .arg = IntListArg{{1, kTokens, kCache}, {}}}},
      21);
  graph.value(21).tensor_meta().sizes = {1, kTokens, kCache};
  graph.node(22) = call(
      "mask_2",
      "torch.ops.aten.unsqueeze_copy.default",
      {{.name = "self", .arg = TensorArg{21}},
       {.name = "dim", .arg = IntArg{1}}},
      22);
  const ValueId expanded = graph.append_value(
      Value("expanded", ScalarType::Bool, {1, 1, kTokens, kCache}));
  graph.insert_node_before(
      23,
      call(
          "expanded",
          "torch.ops.aten.expand_copy.default",
          {{.name = "self", .arg = TensorArg{22}},
           {.name = "size", .arg = IntListArg{{1, 1, kTokens, kCache}, {}}}},
          expanded));
  graph.node(23).inputs[0].arg = TensorArg{expanded};
  graph.rebuild_def_use();

  EXPECT_EQ(lower_hf_attention(graph), 1);
  const auto targets = scheduled_targets(graph);
  EXPECT_EQ(std::ranges::count(targets, "torch.ops.aten.le.Tensor"), 0);
  EXPECT_EQ(
      std::ranges::count(targets, "torch.ops.llama.custom_sdpa.default"), 1);
  EXPECT_NO_THROW(validate_graph(graph));
}

TEST(LowerHFAttentionTest, ChecksCachePositions) {
  const auto check = [](std::vector<int64_t> positions) {
    int64_t rows_written = 4;
    check_cache_positions(
        positions, CachePosition{.cache_len = 4}, rows_written);
  };
  EXPECT_NO_THROW(check({}));
  EXPECT_NO_THROW((check({0, 1, 2})));
  EXPECT_NO_THROW(check({3}));
  EXPECT_NO_THROW((check({0, 1, 2, 3})));
  EXPECT_THROW(check({4}), std::runtime_error);
  EXPECT_THROW((check({2, 3, 4})), std::runtime_error);
  EXPECT_THROW((check({0, 1, 2, 3, 4})), std::runtime_error);
  EXPECT_THROW(check({-1}), std::runtime_error);
  EXPECT_THROW((check({0, 2})), std::runtime_error);
  EXPECT_THROW((check({1, 0})), std::runtime_error);
}

TEST(LowerHFAttentionTest, RejectsCachePositionsPastWrittenRows) {
  const CachePosition bound{.cache_len = 8};
  int64_t rows_written = 0;
  EXPECT_THROW(
      check_cache_positions(std::vector<int64_t>{1}, bound, rows_written),
      std::runtime_error);
  EXPECT_NO_THROW(check_cache_positions(
      std::vector<int64_t>{0, 1, 2}, bound, rows_written));
  EXPECT_EQ(rows_written, 3);
  EXPECT_NO_THROW(
      check_cache_positions(std::vector<int64_t>{3}, bound, rows_written));
  EXPECT_EQ(rows_written, 4);
  EXPECT_THROW(
      check_cache_positions(std::vector<int64_t>{5}, bound, rows_written),
      std::runtime_error);
  EXPECT_NO_THROW(
      check_cache_positions(std::vector<int64_t>{0, 1}, bound, rows_written));
  EXPECT_EQ(rows_written, 4);
}

TEST(LowerHFAttentionTest, KeepsNodesWithoutOutputs) {
  Graph graph = make_graph();
  graph.insert_node_before(
      24,
      Node{
          .name = "check",
          .target = "torch.ops.aten._assert_async.msg",
          .inputs = {{.name = "self", .arg = TensorArg{3}}},
      });
  graph.rebuild_def_use();

  EXPECT_EQ(lower_hf_attention(graph), 1);
  EXPECT_EQ(
      std::ranges::count(
          scheduled_targets(graph), "torch.ops.aten._assert_async.msg"),
      1);
}

TEST(LowerHFAttentionTest, RejectsNonRank4KeyValue) {
  Graph graph = make_graph();
  for (const ValueId id : {ValueId{1}, ValueId{2}}) {
    graph.value(id).tensor_meta().sizes = {kTokens, kKvHeads * kDim};
  }

  EXPECT_EQ(lower_hf_attention(graph), 0);
}

TEST(LowerHFAttentionTest, RejectsEmptyInvFreq) {
  Graph graph = make_graph();
  graph.value(4).tensor_meta().sizes = {0};

  EXPECT_EQ(lower_hf_attention(graph), 0);
}

TEST(LowerHFAttentionTest, RejectsInterleavedRope) {
  Graph graph = make_graph();
  graph.node(10).inputs[3].arg = BoolArg{true};

  EXPECT_EQ(lower_hf_attention(graph), 0);
  EXPECT_EQ(
      producer_of(graph, 25).target, "torch.ops.aten.permute_copy.default");
  EXPECT_NO_THROW(validate_graph(graph));
}

// Exported graphs omit default arguments, so attention_scale can sit at the
// position of an omitted interleaved.
TEST(LowerHFAttentionTest, ReadsNamedRopeInputsAfterOmittedDefault) {
  Graph graph = make_graph();
  std::vector<NamedArgument>& rope_inputs = graph.node(10).inputs;
  rope_inputs.erase(rope_inputs.begin() + 3);

  EXPECT_EQ(lower_hf_attention(graph), 1);
}

TEST(LowerHFAttentionTest, ReadsUnnamedInputsByPosition) {
  Graph graph = make_graph();
  for (Node& node : graph.nodes) {
    for (NamedArgument& input : node.inputs) {
      input.name.clear();
    }
  }

  EXPECT_EQ(lower_hf_attention(graph), 1);
}

TEST(LowerHFAttentionTest, RejectsSdpaInputOfUnknownName) {
  Graph graph = make_graph();
  graph.node(24).inputs[5] = {.name = "causal", .arg = BoolArg{true}};

  EXPECT_EQ(lower_hf_attention(graph), 0);
}

// HF passes head_dim**-0.5 explicitly; any positive literal scale is passed
// through to llama.custom_sdpa without rescaling q.
TEST(LowerHFAttentionTest, PassesExplicitScaleThrough) {
  for (const double scale : {0.5, std::pow(128.0, -0.5)}) {
    SCOPED_TRACE(scale);
    Graph graph = make_graph();
    graph.node(24).inputs[6].arg = FloatArg{scale};

    EXPECT_EQ(lower_hf_attention(graph), 1);

    const Node& sdpa = producer_of(graph, graph.output_ids.at(0));
    EXPECT_DOUBLE_EQ(sdpa.inputs[7].arg.as_float().value, scale);
    EXPECT_EQ(
        producer_of(graph, sdpa.inputs[0].arg.as_tensor().id).target,
        "torch.ops.et_vk.apply_rotary_emb_hf.default");
    EXPECT_EQ(
        std::ranges::count(
            scheduled_targets(graph), "torch.ops.aten.mul.Scalar"),
        0);
    EXPECT_NO_THROW(validate_graph(graph));
  }
}

TEST(LowerHFAttentionTest, RejectsUnsupportedScale) {
  for (const double scale :
       {0.0,
        -0.25,
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity()}) {
    SCOPED_TRACE(scale);
    Graph graph = make_graph();
    graph.node(24).inputs[6].arg = FloatArg{scale};

    EXPECT_EQ(lower_hf_attention(graph), 0);
    EXPECT_EQ(scheduled_targets(graph).size(), 18);
  }
}

// RoPE is optional only as a pair.
TEST(LowerHFAttentionTest, RejectsRopeOnOnlyOneOfQueryAndKey) {
  for (const bool rotate_query : {false, true}) {
    SCOPED_TRACE(rotate_query);
    Graph graph = make_graph();
    if (rotate_query) {
      graph.node(14).inputs[2].arg = TensorArg{12};
    } else {
      graph.node(24).inputs[0].arg = TensorArg{10};
    }
    graph.rebuild_def_use();

    EXPECT_EQ(lower_hf_attention(graph), 0);
    EXPECT_EQ(
        graph.node(24).target,
        "torch.ops.aten.scaled_dot_product_attention.default");
  }
}

// Appends neg(`id`) as a graph output, giving `id` a reader outside the block.
void read_elsewhere(Graph& graph, ValueId id) {
  const ValueId negated =
      graph.append_value(Value("negated", graph.value(id).tensor_meta()));
  graph.insert_node_before(
      graph.schedule.back(),
      call(
          "negated",
          "torch.ops.aten.neg.default",
          {{.name = "self", .arg = TensorArg{id}}},
          negated));
  graph.node(graph.schedule.back())
      .inputs.push_back({.name = "output", .arg = TensorArg{negated}});
  graph.output_ids.push_back(negated);
  graph.rebuild_def_use();
}

// The rewrite reads q from before its permute, which stays for its other
// reader while the clone feeding SDPA is dropped.
TEST(LowerHFAttentionTest, RewritesWhenPermutedQueryIsReadElsewhere) {
  Graph graph = make_whisper_graph();
  read_elsewhere(graph, 9);

  EXPECT_EQ(lower_hf_attention(graph), 1);
  const std::vector<std::string> targets = scheduled_targets(graph);
  EXPECT_EQ(
      std::ranges::count(targets, "torch.ops.aten.permute_copy.default"), 1);
  EXPECT_EQ(std::ranges::count(targets, "torch.ops.aten.clone.default"), 0);
  EXPECT_EQ(std::ranges::count(targets, "torch.ops.aten.neg.default"), 1);
}

// The rewrite transposes the caches and erases the writes.
TEST(LowerHFAttentionTest, RejectsCachesReadElsewhere) {
  for (const ValueId id : {ValueId{4}, ValueId{15}}) {
    SCOPED_TRACE(id);
    Graph graph = make_whisper_graph();
    read_elsewhere(graph, id);

    EXPECT_EQ(lower_hf_attention(graph), 0);
  }
}

// Without RoPE, the mask's query operand is the block's only link to
// cache_position, so only le(kv_index, cache_position) broadcast along
// unchanged query and key axes matches.
TEST(LowerHFAttentionTest, RejectsWhisperMaskVariants) {
  const auto insert_wrapper = [](Graph& graph,
                                 const char* target,
                                 const char* param,
                                 std::vector<int64_t> values) {
    const ValueId wrapped =
        graph.append_value(Value("wrapped", graph.value(23).tensor_meta()));
    graph.insert_node_before(
        24,
        call(
            "wrapped",
            target,
            {{.name = "self", .arg = TensorArg{23}},
             {.name = param, .arg = IntListArg{std::move(values), {}}}},
            wrapped));
    graph.node(24).inputs[0].arg = TensorArg{wrapped};
  };
  for (const auto& [name, edit] :
       std::vector<std::pair<std::string, std::function<void(Graph&)>>>{
           {"ge",
            [](Graph& graph) {
              graph.node(23).target = "torch.ops.aten.ge.Tensor";
            }},
           {"lt",
            [](Graph& graph) {
              graph.node(23).target = "torch.ops.aten.lt.Tensor";
            }},
           {"swapped operands",
            [](Graph& graph) {
              std::swap(
                  graph.node(23).inputs[0].arg, graph.node(23).inputs[1].arg);
            }},
           {"expand along the query axis",
            [](Graph& graph) {
              graph.value(22).tensor_meta().sizes = {1, 1, 1, 1};
              graph.value(23).tensor_meta().sizes = {1, 1, 1, kCache};
            }},
           {"view between le and the mask",
            [&](Graph& graph) {
              insert_wrapper(
                  graph,
                  "torch.ops.aten.view_copy.default",
                  "size",
                  {1, 1, kTokens, kCache});
            }},
           {"permute between le and the mask",
            [&](Graph& graph) {
              insert_wrapper(
                  graph,
                  "torch.ops.aten.permute_copy.default",
                  "dims",
                  {0, 1, 2, 3});
            }},
       }) {
    SCOPED_TRACE(name);
    Graph graph = make_whisper_graph();
    edit(graph);
    graph.rebuild_def_use();

    EXPECT_EQ(lower_hf_attention(graph), 0);
  }
}

// The mask's query operand may be a view, but only of cache_position.
TEST(LowerHFAttentionTest, ChecksSourceOfViewedMaskQueryOperand) {
  for (const bool from_cache_position : {true, false}) {
    SCOPED_TRACE(from_cache_position);
    Graph graph = make_whisper_graph();
    ValueId source = 3;
    if (!from_cache_position) {
      source = graph.append_value(
          Value("other_position", ScalarType::Long, {kTokens}));
      graph.insert_node_before(
          graph.schedule.front(), placeholder("other_position", source));
      graph.input_ids.push_back(source);
    }
    const ValueId viewed = graph.append_value(
        Value("viewed", ScalarType::Long, {1, 1, kTokens, 1}));
    graph.insert_node_before(
        23,
        call(
            "viewed",
            "torch.ops.aten.view_copy.default",
            {{.name = "self", .arg = TensorArg{source}},
             {.name = "size", .arg = IntListArg{{1, 1, kTokens, 1}, {}}}},
            viewed));
    graph.node(23).inputs[1].arg = TensorArg{viewed};
    graph.rebuild_def_use();

    EXPECT_EQ(lower_hf_attention(graph), from_cache_position ? 1 : 0);
  }
}

// Buffer rows are padded to whole texels, so any head width works.
TEST(LowerHFAttentionTest, RewritesHeadDimNotMultipleOfFour) {
  Graph graph = make_whisper_graph();
  for (Value& value : graph.values) {
    if (value.is_tensor() && value.tensor_meta().dtype == ScalarType::Float) {
      value.tensor_meta().sizes.back() = 6;
    }
  }

  EXPECT_EQ(lower_hf_attention(graph), 1);
}

TEST(LowerHFAttentionTest, RejectsUnmaskedAttention) {
  Graph graph = make_graph();
  graph.node(24).inputs[3].arg = NoneArg{};

  EXPECT_EQ(lower_hf_attention(graph), 0);
}

// Gemma-style sliding-window layers AND the causal mask with
// kv_index > cache_position - window.
TEST(LowerHFAttentionTest, RejectsSlidingWindowMask) {
  Graph graph = make_graph();
  const ValueId window_start =
      graph.append_value(Value("window_start", ScalarType::Long, {kTokens, 1}));
  const ValueId in_window = graph.append_value(
      Value("in_window", ScalarType::Bool, {kTokens, kCache}));
  const ValueId sliding =
      graph.append_value(Value("sliding", ScalarType::Bool, {kTokens, kCache}));
  graph.insert_node_before(
      20,
      call(
          "window_start",
          "torch.ops.aten.sub.Tensor",
          {{.name = "self", .arg = TensorArg{18}},
           {.name = "other", .arg = TensorArg{7}}},
          window_start));
  graph.insert_node_before(
      20,
      call(
          "in_window",
          "torch.ops.aten.gt.Tensor",
          {{.name = "self", .arg = TensorArg{17}},
           {.name = "other", .arg = TensorArg{window_start}}},
          in_window));
  graph.insert_node_before(
      20,
      call(
          "sliding",
          "torch.ops.aten.mul.Tensor",
          {{.name = "self", .arg = TensorArg{19}},
           {.name = "other", .arg = TensorArg{in_window}}},
          sliding));
  graph.node(20).inputs[0].arg = TensorArg{sliding};
  graph.rebuild_def_use();

  EXPECT_EQ(lower_hf_attention(graph), 0);
}
// cppcheck-suppress-end syntaxError

} // namespace
} // namespace ptn::vulkan
