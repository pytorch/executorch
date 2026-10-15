// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// cppcheck-suppress-file syntaxError

#include <executorch/backends/native/runtime/vulkan/passes/LowerSDPA.h>

#include <algorithm>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <executorch/backends/native/runtime/Method.h>
#include <executorch/backends/native/runtime/graph/Argument.h>
#include <executorch/backends/native/runtime/graph/Node.h>
#include <executorch/backends/native/runtime/graph/ScalarType.h>
#include <executorch/backends/native/runtime/graph/Value.h>
#include <executorch/backends/native/runtime/graph/utils/GraphUtils.h>

namespace ptn::vulkan {
namespace {

constexpr const char* kAtenSDPA =
    "torch.ops.aten.scaled_dot_product_attention.default";

// q [1, 4, 8, 16], k/v [1, 4, 12, 16], out [1, 4, 8, 16], plus an optional
// mask value (id 4) typed by `mask_dtype`. The SDPA node starts with only
// query/key/value inputs; tests append the optional arguments they exercise.
Method make_sdpa(ScalarType mask_dtype = ScalarType::Float) {
  Method method;
  method.name = "forward";
  Graph& graph = method.graph;
  graph.values = {
      Value("query", ScalarType::Float, {1, 4, 8, 16}),
      Value("key", ScalarType::Float, {1, 4, 12, 16}),
      Value("value", ScalarType::Float, {1, 4, 12, 16}),
      Value("output", ScalarType::Float, {1, 4, 8, 16}),
      Value("mask", mask_dtype, {1, 1, 8, 12}),
  };
  graph.nodes = {
      Node{
          .name = "query",
          .op_kind = OpKind::Placeholder,
          .outputs = {{.value_id = 0}},
      },
      Node{
          .name = "key",
          .op_kind = OpKind::Placeholder,
          .outputs = {{.value_id = 1}},
      },
      Node{
          .name = "value",
          .op_kind = OpKind::Placeholder,
          .outputs = {{.value_id = 2}},
      },
      Node{
          .name = "mask",
          .op_kind = OpKind::Placeholder,
          .outputs = {{.value_id = 4}},
      },
      Node{
          .name = "sdpa",
          .target = kAtenSDPA,
          .inputs =
              {
                  {.name = "query", .arg = TensorArg{0}},
                  {.name = "key", .arg = TensorArg{1}},
                  {.name = "value", .arg = TensorArg{2}},
              },
          .outputs = {{.value_id = 3}},
      },
      Node{
          .name = "output",
          .op_kind = OpKind::Output,
          .inputs = {{.arg = TensorArg{3}}},
      },
  };
  graph.schedule = {0, 1, 2, 3, 4, 5};
  graph.input_ids = {0, 1, 2, 4};
  graph.output_ids = {3};
  graph.rebuild_def_use();
  return method;
}

TEST(LowerSDPATest, RewritesPlainSDPAWithDefaults) {
  Method method = make_sdpa();

  EXPECT_EQ(lower_sdpa(method.graph), 1);

  const Node& sdpa = method.graph.node(4);
  EXPECT_EQ(sdpa.target, "torch.ops.et_vk.sdpa.default");
  ASSERT_EQ(sdpa.inputs.size(), 5);
  EXPECT_EQ(sdpa.inputs[0].arg.as_tensor().id, 0);
  EXPECT_EQ(sdpa.inputs[1].arg.as_tensor().id, 1);
  EXPECT_EQ(sdpa.inputs[2].arg.as_tensor().id, 2);
  EXPECT_EQ(sdpa.inputs[3].arg.kind(), ArgKind::None);
  EXPECT_EQ(sdpa.inputs[4].arg.kind(), ArgKind::None);
  EXPECT_NO_THROW(validate_graph(method.graph));
}

TEST(LowerSDPATest, RewritesFloatMaskAndExplicitScale) {
  Method method = make_sdpa();
  Node& sdpa = method.graph.node(4);
  sdpa.inputs.push_back({.name = "attn_mask", .arg = TensorArg{4}});
  sdpa.inputs.push_back({.name = "dropout_p", .arg = FloatArg{0.0}});
  sdpa.inputs.push_back({.name = "is_causal", .arg = BoolArg{false}});
  sdpa.inputs.push_back({.name = "scale", .arg = FloatArg{1.0}});
  method.graph.rebuild_def_use();

  EXPECT_EQ(lower_sdpa(method.graph), 1);

  const Node& lowered = method.graph.node(4);
  EXPECT_EQ(lowered.target, "torch.ops.et_vk.sdpa.default");
  ASSERT_EQ(lowered.inputs.size(), 5);
  EXPECT_EQ(lowered.inputs[3].arg.as_tensor().id, 4);
  EXPECT_DOUBLE_EQ(lowered.inputs[4].arg.as_float().value, 1.0);
  EXPECT_NO_THROW(validate_graph(method.graph));
}

TEST(LowerSDPATest, RejectsCausalFlag) {
  Method method = make_sdpa();
  method.graph.node(4).inputs.push_back(
      {.name = "is_causal", .arg = BoolArg{true}});

  EXPECT_EQ(lower_sdpa(method.graph), 0);
  EXPECT_EQ(method.graph.node(4).target, kAtenSDPA);
}

TEST(LowerSDPATest, RejectsDropout) {
  Method method = make_sdpa();
  method.graph.node(4).inputs.push_back(
      {.name = "dropout_p", .arg = FloatArg{0.1}});

  EXPECT_EQ(lower_sdpa(method.graph), 0);
}

TEST(LowerSDPATest, ConvertsBoolMaskToAdditiveBias) {
  Method method = make_sdpa(ScalarType::Bool);
  method.graph.node(4).inputs.push_back(
      {.name = "attn_mask", .arg = TensorArg{4}});
  method.graph.rebuild_def_use();

  EXPECT_EQ(lower_sdpa(method.graph), 1);

  const Node& lowered = method.graph.node(4);
  EXPECT_EQ(lowered.target, "torch.ops.et_vk.sdpa.default");
  ASSERT_EQ(lowered.inputs.size(), 5);
  // The mask operand is the converted bias, not the bool mask itself.
  const ValueId bias_id = lowered.inputs[3].arg.as_tensor().id;
  EXPECT_NE(bias_id, 4);
  EXPECT_EQ(method.graph.value(bias_id).tensor_meta().dtype, ScalarType::Float);
  EXPECT_EQ(
      method.graph.value(bias_id).tensor_meta().sizes,
      method.graph.value(4).tensor_meta().sizes);

  // Scalar fills and the selection must precede SDPA.
  std::vector<std::string> scheduled;
  scheduled.reserve(method.graph.schedule.size());
  for (const NodeId id : method.graph.schedule) {
    scheduled.push_back(method.graph.node(id).target);
  }
  const auto sdpa_pos = std::find(
      scheduled.begin(), scheduled.end(), "torch.ops.et_vk.sdpa.default");
  ASSERT_NE(sdpa_pos, scheduled.end());
  for (const char* conversion :
       {"torch.ops.aten.full.default", "torch.ops.aten.where.self"}) {
    const auto pos = std::find(scheduled.begin(), scheduled.end(), conversion);
    ASSERT_NE(pos, scheduled.end()) << conversion;
    EXPECT_LT(pos, sdpa_pos) << conversion;
  }
  EXPECT_NO_THROW(validate_graph(method.graph));
}

TEST(LowerSDPATest, RejectsGqaHeadMismatch) {
  Method method = make_sdpa();
  method.graph.value(1).tensor_meta().sizes = {1, 2, 12, 16};
  method.graph.value(2).tensor_meta().sizes = {1, 2, 12, 16};

  EXPECT_EQ(lower_sdpa(method.graph), 0);
}

TEST(LowerSDPATest, RejectsUnequalValueHeadWidth) {
  Method method = make_sdpa();
  method.graph.value(2).tensor_meta().sizes[3] = 32;
  method.graph.value(3).tensor_meta().sizes[3] = 32;

  EXPECT_EQ(lower_sdpa(method.graph), 0);
  EXPECT_EQ(method.graph.node(4).target, kAtenSDPA);
}

TEST(LowerSDPATest, RejectsUnsupportedMaskShapesWithoutInsertingNodes) {
  for (const std::vector<int64_t>& shape :
       {std::vector<int64_t>{1, 1, 8, 1},
        std::vector<int64_t>{1, 2, 8, 12},
        std::vector<int64_t>{1, 1, 3, 12},
        std::vector<int64_t>{1, 1, 1, 8, 12},
        std::vector<int64_t>{}}) {
    for (const ScalarType dtype : {ScalarType::Bool, ScalarType::Float}) {
      Method method = make_sdpa(dtype);
      method.graph.value(4).tensor_meta().sizes = shape;
      method.graph.node(4).inputs.push_back(
          {.name = "attn_mask", .arg = TensorArg{4}});
      const auto schedule = method.graph.schedule;

      EXPECT_EQ(lower_sdpa(method.graph), 0);
      EXPECT_EQ(method.graph.node(4).target, kAtenSDPA);
      EXPECT_EQ(method.graph.schedule, schedule);
    }
  }
}

TEST(LowerSDPATest, AcceptsMaskBroadcastAcrossBatchHeadsAndQueries) {
  for (const std::vector<int64_t>& shape :
       {std::vector<int64_t>{12},
        std::vector<int64_t>{8, 12},
        std::vector<int64_t>{1, 1, 1, 12}}) {
    Method method = make_sdpa();
    method.graph.value(4).tensor_meta().sizes = shape;
    method.graph.node(4).inputs.push_back(
        {.name = "attn_mask", .arg = TensorArg{4}});

    EXPECT_EQ(lower_sdpa(method.graph), 1);
    EXPECT_NO_THROW(validate_graph(method.graph));
  }
}

TEST(LowerSDPATest, RejectsEnableGqa) {
  Method method = make_sdpa();
  method.graph.node(4).inputs.push_back(
      {.name = "enable_gqa", .arg = BoolArg{true}});

  EXPECT_EQ(lower_sdpa(method.graph), 0);
}

} // namespace
} // namespace ptn::vulkan
