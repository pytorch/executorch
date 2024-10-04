/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/xnnpack/runtime/XNNExecutor.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <gtest/gtest.h>
#include <xnnpack/subgraph.h>

using executorch::backends::xnnpack::delegate::XNNExecutor;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::testing::TensorFactory;

TEST(XNNExecutorTest, ArgumentWithTooManyDimensions) {
  XNNExecutor executor;
  xnn_subgraph_t subgraph = nullptr;
  xnn_runtime_t rt = nullptr;
  et_pal_init();
  ASSERT_EQ(xnn_initialize(nullptr), xnn_status_success);
  ASSERT_EQ(xnn_create_subgraph(2, 0, &subgraph), xnn_status_success);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  auto input_id = XNN_INVALID_NODE_ID;
  std::vector<size_t> dims = {
      1,
  };
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_quantized_tensor_value(
          subgraph,
          xnn_datatype_qint8,
          0,
          1,
          dims.size(),
          dims.data(),
          nullptr,
          /*external_id=*/0,
          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT,
          &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  auto output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_quantized_tensor_value(
          subgraph,
          xnn_datatype_qint8,
          0,
          1,
          dims.size(),
          dims.data(),
          nullptr,
          /*external_id=*/0,
          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
          &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
      xnn_status_success,
      xnn_define_clamp(subgraph, 1, 2, input_id, output_id, 0));

  ASSERT_EQ(xnn_create_runtime(subgraph, &rt), xnn_status_success);
  EXPECT_EQ(
      executor.initialize(
          rt,
          {
              0,
          },
          {
              1,
          }),
      Error::Ok);
  TensorFactory<executorch::aten::ScalarType::Int> tf;
  auto input_tensor = tf.make({1, 1, 1, 1, 1, 1, 1, 1, 1}, {42});
  ASSERT_EQ(input_tensor.dim(), 9);
  auto output_tensor = tf.make(
      {
          1,
      },
      {
          1,
      });
  EValue input_ev(input_tensor);
  EValue output_ev(output_tensor);
  std::array<EValue*, 2> args = {&input_ev, &output_ev};
  // Check for invalid number of dimensions should fail without stack overflow.
  EXPECT_EQ(executor.prepare_args(args.data()), Error::InvalidArgument);
}
