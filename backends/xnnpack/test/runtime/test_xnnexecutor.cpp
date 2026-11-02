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

  std::vector<size_t> in_dims = {1, 4, 4, 4}, out_dims = {1, 2, 2, 4};
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

  // 4 channels to provide XNN_EXTRA_BYTES padding for SIMD reads
  std::vector<float> input_data(4 * 4 * 4, 0);
  for (int i = 0; i < 16; ++i) {
    input_data[i * 4] = static_cast<float>(i + 1); // channel 0 has values 1-16
  }
  auto input = tf_float.make({1, 4, 4, 4}, input_data);
  auto out_value = tf_float.make({1, 2, 2, 4}, std::vector<float>(16, 0));
  auto out_index = tf_long.make({1, 2, 2, 4}, std::vector<int64_t>(16, 0));

  EValue ev_in(input), ev_val(out_value), ev_idx(out_index);
  std::array<EValue*, 3> args = {&ev_in, &ev_val, &ev_idx};
  Span<EValue*> span(args.data(), 3);

  ASSERT_EQ(executor.prepare_args(span), Error::Ok);
  executorch::ET_RUNTIME_NAMESPACE::BackendExecutionContext context;
  ASSERT_EQ(executor.forward(context), Error::Ok);
  ASSERT_EQ(executor.convert_outputs(span), Error::Ok);

  Tensor& result = args[2]->toTensor();
  ASSERT_EQ(result.scalar_type(), executorch::aten::ScalarType::Long);

  /*
  Verify all 4 spatial positions for channel 0 (channels 1-3 have all zeros)
  Output is NHWC {1,2,2,4}, so channel 0 is at indices 0, 4, 8, 12

  Diagram for channel 0:
  Input 4x4:          Output values:   Output indices:
  1  2  | 3  4        6  | 8           3 | 3
  5  6  | 7  8        14 | 16          3 | 3
  ------|-----
  9  10 |11 12
  13 14 |15 16

  Each 2x2 quadrant → max value + index of max (3 = bottom-right).
  */
  for (ssize_t i = 0; i < 4; ++i) {
    EXPECT_EQ(result.const_data_ptr<int64_t>()[i * 4], 3);
  }
}

namespace {

using executorch::runtime::TensorShapeDynamism;

// Defines an fp32 external tensor with the given external id.
uint32_t define_external(
    xnn_subgraph_t subgraph,
    const std::vector<size_t>& dims,
    uint32_t external_id,
    uint32_t flags) {
  uint32_t id = XNN_INVALID_VALUE_ID;
  EXPECT_EQ(
      xnn_define_tensor_value(
          subgraph,
          xnn_datatype_fp32,
          dims.size(),
          dims.data(),
          nullptr,
          external_id,
          flags,
          &id),
      xnn_status_success);
  return id;
}

// Runs prepare_args and forward on one executor call.
Error run(XNNExecutor& executor, std::vector<EValue*> args) {
  Span<EValue*> span(args.data(), args.size());
  Error err = executor.prepare_args(span);
  if (err != Error::Ok) {
    return err;
  }
  executorch::ET_RUNTIME_NAMESPACE::BackendExecutionContext context;
  return executor.forward(context);
}

std::vector<float> values(const EValue& ev) {
  const Tensor& t = ev.toTensor();
  return std::vector<float>(
      t.const_data_ptr<float>(), t.const_data_ptr<float>() + t.numel());
}

} // namespace

// Tests that a changed input shape is propagated through the runtime, and that
// repeating a shape with new input data still computes on that data.
TEST(XNNExecutorTest, ReshapesWhenInputShapeChanges) {
  XNNExecutor executor({});
  et_pal_init();
  ASSERT_EQ(xnn_initialize(nullptr), xnn_status_success);
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_create_subgraph(2, 0, &subgraph), xnn_status_success);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);
  const uint32_t input_id =
      define_external(subgraph, {2, 3}, 0, XNN_VALUE_FLAG_EXTERNAL_INPUT);
  const uint32_t output_id =
      define_external(subgraph, {2, 3}, 1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT);
  ASSERT_EQ(
      xnn_define_clamp(subgraph, 0.0f, 10.0f, input_id, output_id, 0),
      xnn_status_success);
  xnn_runtime_t rt = nullptr;
  ASSERT_EQ(xnn_create_runtime(subgraph, &rt), xnn_status_success);
  ASSERT_EQ(executor.initialize(rt, {0}, {1}, {}), Error::Ok);

  TensorFactory<executorch::aten::ScalarType::Float> tf;
  // Sized for the largest shape; prepare_args resizes it to each input's shape.
  EValue out(tf.make(
      {4, 3},
      std::vector<float>(12, -1.0f),
      {},
      TensorShapeDynamism::DYNAMIC_BOUND));

  EValue in1(tf.make({2, 3}, {-1, 2, 3, 4, 5, 20}));
  ASSERT_EQ(run(executor, {&in1, &out}), Error::Ok);
  EXPECT_EQ(values(out), (std::vector<float>{0, 2, 3, 4, 5, 10}));

  EValue in2(tf.make({4, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -12}));
  ASSERT_EQ(run(executor, {&in2, &out}), Error::Ok);
  EXPECT_EQ(
      values(out), (std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 0}));

  EValue in3(tf.make({2, 3}, {6, 5, 4, 3, 2, 1}));
  ASSERT_EQ(run(executor, {&in3, &out}), Error::Ok);
  EXPECT_EQ(values(out), (std::vector<float>{6, 5, 4, 3, 2, 1}));

  EValue in4(tf.make({2, 3}, {7, 7, 7, -7, 70, 0}));
  ASSERT_EQ(run(executor, {&in4, &out}), Error::Ok);
  EXPECT_EQ(values(out), (std::vector<float>{7, 7, 7, 0, 10, 0}));
}

// Tests that a repeated input shape does not reshape the runtime: a shape set
// on the runtime behind the executor's back survives the next call.
TEST(XNNExecutorTest, SkipsReshapeForRepeatedInputShape) {
  XNNExecutor executor({});
  et_pal_init();
  ASSERT_EQ(xnn_initialize(nullptr), xnn_status_success);
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_create_subgraph(2, 0, &subgraph), xnn_status_success);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);
  const uint32_t input_id =
      define_external(subgraph, {2, 3}, 0, XNN_VALUE_FLAG_EXTERNAL_INPUT);
  const uint32_t output_id =
      define_external(subgraph, {2, 3}, 1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT);
  ASSERT_EQ(
      xnn_define_clamp(subgraph, 0.0f, 10.0f, input_id, output_id, 0),
      xnn_status_success);
  xnn_runtime_t rt = nullptr;
  ASSERT_EQ(xnn_create_runtime(subgraph, &rt), xnn_status_success);
  ASSERT_EQ(executor.initialize(rt, {0}, {1}, {}), Error::Ok);

  TensorFactory<executorch::aten::ScalarType::Float> tf;
  EValue out(tf.make(
      {4, 3},
      std::vector<float>(12, -1.0f),
      {},
      TensorShapeDynamism::DYNAMIC_BOUND));
  // A {4, 3} buffer viewed as {2, 3}, so running the runtime at {4, 3} stays in
  // bounds.
  Tensor input = tf.make(
      {4, 3},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
      {},
      TensorShapeDynamism::DYNAMIC_BOUND);
  const std::array<executorch::aten::SizesType, 2> view_sizes = {2, 3};
  ASSERT_EQ(
      executorch::ET_RUNTIME_NAMESPACE::resize_tensor(
          input, {view_sizes.data(), view_sizes.size()}),
      Error::Ok);
  EValue in(input);
  ASSERT_EQ(run(executor, {&in, &out}), Error::Ok);
  EXPECT_EQ(out.toTensor().size(0), 2);

  const std::vector<size_t> other_dims = {4, 3};
  ASSERT_EQ(
      xnn_reshape_external_value(rt, 0, other_dims.size(), other_dims.data()),
      xnn_status_success);
  ASSERT_EQ(xnn_reshape_runtime(rt), xnn_status_success);

  ASSERT_EQ(run(executor, {&in, &out}), Error::Ok);
  EXPECT_EQ(out.toTensor().size(0), 4);
}

// Tests that after a failed reshape the next call reshapes again, and that
// swapping the shapes of two inputs is seen as a shape change.
TEST(XNNExecutorTest, ReshapesAfterFailedReshapeAndSwappedInputShapes) {
  XNNExecutor executor({});
  et_pal_init();
  ASSERT_EQ(xnn_initialize(nullptr), xnn_status_success);
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_create_subgraph(3, 0, &subgraph), xnn_status_success);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);
  const uint32_t a_id =
      define_external(subgraph, {2, 3}, 0, XNN_VALUE_FLAG_EXTERNAL_INPUT);
  const uint32_t b_id =
      define_external(subgraph, {2, 3}, 1, XNN_VALUE_FLAG_EXTERNAL_INPUT);
  const uint32_t out_id =
      define_external(subgraph, {2, 3}, 2, XNN_VALUE_FLAG_EXTERNAL_OUTPUT);
  ASSERT_EQ(
      xnn_define_binary(
          subgraph, xnn_binary_add, nullptr, a_id, b_id, out_id, 0),
      xnn_status_success);
  xnn_runtime_t rt = nullptr;
  ASSERT_EQ(xnn_create_runtime(subgraph, &rt), xnn_status_success);
  ASSERT_EQ(executor.initialize(rt, {0, 1}, {2}, {}), Error::Ok);

  TensorFactory<executorch::aten::ScalarType::Float> tf;
  EValue out(tf.make(
      {4, 3},
      std::vector<float>(12, -1.0f),
      {},
      TensorShapeDynamism::DYNAMIC_BOUND));

  EValue a(tf.make({2, 3}, {1, 2, 3, 4, 5, 6}));
  EValue b(tf.make({2, 3}, {10, 20, 30, 40, 50, 60}));
  ASSERT_EQ(run(executor, {&a, &b, &out}), Error::Ok);
  EXPECT_EQ(values(out), (std::vector<float>{11, 22, 33, 44, 55, 66}));

  EValue b_bad(tf.make({4, 3}, std::vector<float>(12, 1.0f)));
  EXPECT_NE(run(executor, {&a, &b_bad, &out}), Error::Ok);

  EValue b2(tf.make({2, 3}, {1, 1, 1, 1, 1, 1}));
  ASSERT_EQ(run(executor, {&a, &b2, &out}), Error::Ok);
  EXPECT_EQ(values(out), (std::vector<float>{2, 3, 4, 5, 6, 7}));

  EValue row(tf.make({1, 3}, {100, 200, 300}));
  ASSERT_EQ(run(executor, {&row, &b2, &out}), Error::Ok);
  EXPECT_EQ(values(out), (std::vector<float>{101, 201, 301, 101, 201, 301}));

  ASSERT_EQ(run(executor, {&b2, &row, &out}), Error::Ok);
  EXPECT_EQ(values(out), (std::vector<float>{101, 201, 301, 101, 201, 301}));
  EValue a2(tf.make({2, 3}, {0, 0, 0, 1, 1, 1}));
  ASSERT_EQ(run(executor, {&a2, &row, &out}), Error::Ok);
  EXPECT_EQ(values(out), (std::vector<float>{100, 200, 300, 101, 201, 301}));
}

// Tests a rank-0 input in the last position, whose shape is the final entry of
// the cached dims.
TEST(XNNExecutorTest, ReshapesWithRankZeroLastInput) {
  XNNExecutor executor({});
  et_pal_init();
  ASSERT_EQ(xnn_initialize(nullptr), xnn_status_success);
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_create_subgraph(3, 0, &subgraph), xnn_status_success);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);
  const uint32_t a_id =
      define_external(subgraph, {2, 3}, 0, XNN_VALUE_FLAG_EXTERNAL_INPUT);
  const uint32_t s_id =
      define_external(subgraph, {}, 1, XNN_VALUE_FLAG_EXTERNAL_INPUT);
  const uint32_t out_id =
      define_external(subgraph, {2, 3}, 2, XNN_VALUE_FLAG_EXTERNAL_OUTPUT);
  ASSERT_EQ(
      xnn_define_binary(
          subgraph, xnn_binary_add, nullptr, a_id, s_id, out_id, 0),
      xnn_status_success);
  xnn_runtime_t rt = nullptr;
  ASSERT_EQ(xnn_create_runtime(subgraph, &rt), xnn_status_success);
  ASSERT_EQ(executor.initialize(rt, {0, 1}, {2}, {}), Error::Ok);

  TensorFactory<executorch::aten::ScalarType::Float> tf;
  EValue out(tf.make(
      {2, 3},
      std::vector<float>(6, -1.0f),
      {},
      TensorShapeDynamism::DYNAMIC_BOUND));
  EValue a(tf.make({2, 3}, {1, 2, 3, 4, 5, 6}));
  // TensorFactory gives a rank-0 tensor a null dim order, which
  // get_dim_order would pass to memcpy.
  float scalar = 10;
  executorch::aten::DimOrderType scalar_dim_order[1] = {0};
  executorch::aten::TensorImpl scalar_impl(
      executorch::aten::ScalarType::Float,
      0,
      nullptr,
      &scalar,
      scalar_dim_order);
  EValue s{Tensor(&scalar_impl)};
  ASSERT_EQ(run(executor, {&a, &s, &out}), Error::Ok);
  EXPECT_EQ(values(out), (std::vector<float>{11, 12, 13, 14, 15, 16}));

  scalar = -1;
  ASSERT_EQ(run(executor, {&a, &s, &out}), Error::Ok);
  EXPECT_EQ(values(out), (std::vector<float>{0, 1, 2, 3, 4, 5}));
}
