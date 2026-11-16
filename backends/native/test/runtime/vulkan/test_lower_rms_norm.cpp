// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// cppcheck-suppress-file syntaxError

#include <executorch/backends/native/runtime/vulkan/passes/LowerRMSNorm.h>

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

Method make_rms_norm(ScalarType dtype = ScalarType::Float, int64_t width = 4) {
  Method method;
  method.name = "forward";
  Graph& graph = method.graph;
  graph.values = {
      Value("input", dtype, {2, width}),
      Value("weight", dtype, {width}),
      Value("output", dtype, {2, width}),
  };
  graph.value(1).role = ValueRole::Parameter;
  graph.nodes = {
      Node{
          .name = "input",
          .op_kind = OpKind::Placeholder,
          .outputs = {{.value_id = 0}},
      },
      Node{
          .name = "weight",
          .op_kind = OpKind::Placeholder,
          .outputs = {{.value_id = 1}},
      },
      Node{
          .name = "rms_norm",
          .target = "torch.ops.aten.rms_norm.default",
          .inputs =
              {
                  {.name = "input", .arg = TensorArg{0}},
                  {.name = "normalized_shape", .arg = IntListArg{{width}}},
                  {.name = "weight", .arg = TensorArg{1}},
                  {.name = "eps", .arg = FloatArg{1e-5}},
              },
          .outputs = {{.value_id = 2}},
      },
      Node{
          .name = "output",
          .op_kind = OpKind::Output,
          .inputs = {{.arg = TensorArg{2}}},
      },
  };
  graph.schedule = {0, 1, 2, 3};
  graph.input_ids = {0};
  graph.output_ids = {2};
  graph.rebuild_def_use();
  return method;
}

TEST(LowerRMSNormTest, RewritesPortableWeightedNorm) {
  Method method = make_rms_norm();

  EXPECT_EQ(lower_rms_norms(method.graph), 1);

  const Node& norm = method.graph.node(2);
  EXPECT_EQ(norm.target, "torch.ops.et_vk.rms_norm.default");
  ASSERT_EQ(norm.inputs.size(), 3);
  EXPECT_EQ(norm.inputs[0].arg.as_tensor().id, 0);
  EXPECT_EQ(norm.inputs[1].arg.as_tensor().id, 1);
  EXPECT_DOUBLE_EQ(norm.inputs[2].arg.as_float().value, 1e-5);
  EXPECT_NO_THROW(validate_graph(method.graph));
}

TEST(LowerRMSNormTest, ReadsUnnamedInputsByPosition) {
  Method method = make_rms_norm();
  for (NamedArgument& input : method.graph.node(2).inputs) {
    input.name.clear();
  }

  EXPECT_EQ(lower_rms_norms(method.graph), 1);
}

TEST(LowerRMSNormTest, RewritesBFloat16Norm) {
  Method method = make_rms_norm(ScalarType::BFloat16);

  EXPECT_EQ(lower_rms_norms(method.graph), 1);

  const Node& norm = method.graph.node(2);
  EXPECT_EQ(norm.target, "torch.ops.et_vk.rms_norm.default");
  ASSERT_EQ(norm.inputs.size(), 3);
  EXPECT_EQ(norm.inputs[0].arg.as_tensor().id, 0);
  EXPECT_EQ(norm.inputs[1].arg.as_tensor().id, 1);
  EXPECT_NO_THROW(validate_graph(method.graph));
}

TEST(LowerRMSNormTest, RewritesNormWithoutWeight) {
  Method method = make_rms_norm();
  method.graph.node(2).inputs[2].arg = NoneArg{};
  method.graph.rebuild_def_use();

  EXPECT_EQ(lower_rms_norms(method.graph), 1);

  const Node& norm = method.graph.node(2);
  EXPECT_EQ(norm.target, "torch.ops.et_vk.rms_norm.default");
  ASSERT_EQ(norm.inputs.size(), 3);
  EXPECT_EQ(norm.inputs[1].arg.kind(), ArgKind::None);
  EXPECT_NO_THROW(validate_graph(method.graph));
}

TEST(LowerRMSNormTest, RejectsMismatchedNormalizedShape) {
  Method method = make_rms_norm();
  method.graph.node(2).inputs[1].arg = IntListArg{{8}};

  EXPECT_EQ(lower_rms_norms(method.graph), 0);
  EXPECT_EQ(method.graph.node(2).target, "torch.ops.aten.rms_norm.default");
}

TEST(LowerRMSNormTest, RejectsInvalidWeight) {
  struct Case {
    const char* name;
    ValueRole role;
    ScalarType dtype;
    std::vector<int64_t> sizes;
  };
  const std::vector<Case> cases = {
      {"intermediate", ValueRole::Intermediate, ScalarType::Float, {4}},
      {"user input", ValueRole::UserInput, ScalarType::Float, {4}},
      {"wrong shape", ValueRole::Parameter, ScalarType::Float, {8}},
      {"wrong dtype", ValueRole::Parameter, ScalarType::Half, {4}},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.name);
    Method method = make_rms_norm();
    method.graph.value(1) = Value("weight", c.dtype, c.sizes);
    method.graph.value(1).role = c.role;
    method.graph.rebuild_def_use();

    EXPECT_EQ(lower_rms_norms(method.graph), 0);
    EXPECT_EQ(method.graph.node(2).target, "torch.ops.aten.rms_norm.default");
  }
}

TEST(LowerRMSNormTest, RejectsZeroWidthNorm) {
  Method method = make_rms_norm(ScalarType::Float, /*width=*/0);
  method.graph.node(2).inputs[2].arg = NoneArg{};
  method.graph.rebuild_def_use();

  EXPECT_EQ(lower_rms_norms(method.graph), 0);
  EXPECT_EQ(method.graph.node(2).target, "torch.ops.aten.rms_norm.default");
}

} // namespace
} // namespace ptn::vulkan
