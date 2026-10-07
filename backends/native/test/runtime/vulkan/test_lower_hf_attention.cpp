// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/vulkan/passes/LowerHFAttention.h>

#include <algorithm>
#include <any>
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

TEST(LowerHFAttentionTest, RejectsNonDefaultScale) {
  Graph graph = make_graph();
  graph.node(24).inputs[6].arg = FloatArg{0.5};

  EXPECT_EQ(lower_hf_attention(graph), 0);
  EXPECT_EQ(scheduled_targets(graph).size(), 18);
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
