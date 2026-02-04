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
#include <xnnpack.h>

using executorch::aten::Tensor;
using executorch::backends::xnnpack::delegate::XNNExecutor;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::Span;
using executorch::runtime::testing::TensorFactory;

TEST(XNNExecutorTest, ArgumentWithTooManyDimensions) {
  XNNExecutor executor({});
  xnn_subgraph_t subgraph = nullptr;
  xnn_runtime_t rt = nullptr;
  et_pal_init();
  ASSERT_EQ(xnn_initialize(nullptr), xnn_status_success);
  ASSERT_EQ(xnn_create_subgraph(2, 0, &subgraph), xnn_status_success);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  auto input_id = XNN_INVALID_VALUE_ID;
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
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  auto output_id = XNN_INVALID_VALUE_ID;
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
          /*external_id=*/1,
          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
          &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

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
          },
          {}),
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
  Span<EValue*> stack_args(args.data(), 2);
  // Check for invalid number of dimensions should fail without stack overflow.
  EXPECT_EQ(executor.prepare_args(stack_args), Error::InvalidArgument);
}

// Tests that resize_outputs correctly converts int32 indices to int64.
TEST(XNNExecutorTest, ResizeOutputsWithLongTensorConvertsInt32ToInt64) {
  XNNExecutor executor({});
  xnn_runtime_t rt = nullptr;
  et_pal_init();
  ASSERT_EQ(xnn_initialize(nullptr), xnn_status_success);

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_create_subgraph(3, 0, &subgraph), xnn_status_success);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  std::vector<size_t> in_dims = {1, 4, 4, 1}, out_dims = {1, 2, 2, 1};
  uint32_t input_id = XNN_INVALID_VALUE_ID;
  uint32_t value_id = XNN_INVALID_VALUE_ID;
  uint32_t index_id = XNN_INVALID_VALUE_ID;

  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
          subgraph,
          xnn_datatype_fp32,
          in_dims.size(),
          in_dims.data(),
          nullptr,
          0,
          XNN_VALUE_FLAG_EXTERNAL_INPUT,
          &input_id));
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
          subgraph,
          xnn_datatype_fp32,
          out_dims.size(),
          out_dims.data(),
          nullptr,
          1,
          XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
          &value_id));
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
          subgraph,
          xnn_datatype_int32,
          out_dims.size(),
          out_dims.data(),
          nullptr,
          2,
          XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
          &index_id));
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_argmax_pooling_2d(
          subgraph, 0, 0, 0, 0, 2, 2, input_id, value_id, index_id, 0));

  ASSERT_EQ(xnn_create_runtime(subgraph, &rt), xnn_status_success);
  ASSERT_EQ(executor.initialize(rt, {0}, {1, 2}, {}), Error::Ok);

  TensorFactory<executorch::aten::ScalarType::Float> tf_float;
  TensorFactory<executorch::aten::ScalarType::Long> tf_long;

  auto input = tf_float.make(
      {1, 4, 4, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  auto out_value = tf_float.make({1, 2, 2, 1}, {0, 0, 0, 0});
  auto out_index = tf_long.make({1, 2, 2, 1}, {0, 0, 0, 0});

  EValue ev_in(input), ev_val(out_value), ev_idx(out_index);
  std::array<EValue*, 3> args = {&ev_in, &ev_val, &ev_idx};
  Span<EValue*> span(args.data(), 3);

  ASSERT_EQ(executor.prepare_args(span), Error::Ok);
  executorch::ET_RUNTIME_NAMESPACE::BackendExecutionContext context;
  ASSERT_EQ(executor.forward(context), Error::Ok);
  ASSERT_EQ(executor.resize_outputs(span), Error::Ok);

  Tensor& result = args[2]->toTensor();
  ASSERT_EQ(result.scalar_type(), executorch::aten::ScalarType::Long);

  /*
  Input 4x4:          Output values:   Output indices:
  1  2  | 3  4        6  | 8           3 | 3
  5  6  | 7  8        14 | 16          3 | 3
  ------|-----
  9  10 |11 12
  13 14 |15 16

  Each 2x2 quadrant → max value + index of max (3 = bottom-right).
  */
  for (ssize_t i = 0; i < result.numel(); ++i) {
    EXPECT_EQ(result.const_data_ptr<int64_t>()[i], 3);
  }
}
