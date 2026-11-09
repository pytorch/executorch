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

Method make_rms_norm() {
  Method method;
  method.name = "forward";
  Graph& graph = method.graph;
  graph.values = {
      Value("input", ScalarType::Float, {2, 4}),
      Value("weight", ScalarType::Float, {4}),
      Value("output", ScalarType::Float, {2, 4}),
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
                  {.name = "normalized_shape", .arg = IntListArg{{4}}},
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

TEST(LowerRMSNormTest, RejectsMismatchedNormalizedShape) {
  Method method = make_rms_norm();
  method.graph.node(2).inputs[1].arg = IntListArg{{8}};

  EXPECT_EQ(lower_rms_norms(method.graph), 0);
  EXPECT_EQ(method.graph.node(2).target, "torch.ops.aten.rms_norm.default");
}

} // namespace
} // namespace ptn::vulkan
