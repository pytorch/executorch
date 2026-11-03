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

namespace executorch::backends::xnnpack::delegate {

class XNNPackedLhsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    runtime::runtime_init();
    backend_ = runtime::get_backend_class(xnnpack_backend_key);
    ASSERT_NE(backend_, nullptr);
    ASSERT_EQ(
        original_options_.set_option(disable_packed_lhs_option_key, false),
        Error::Ok);
    ASSERT_EQ(
        runtime::get_option(xnnpack_backend_key, original_options_.view()),
        Error::Ok);

    std::vector<flatbuffers::Offset<fb_xnnpack::XValue>> values;
    for (uint32_t id = 0; id < 3; ++id) {
      const auto tensor = fb_xnnpack::CreateXNNTensorValue(
          builder_,
          fb_xnnpack::XNNDatatype::xnn_datatype_fp32,
          1,
          builder_.CreateVector<uint32_t>({1}),
          0,
          id,
          id == 2 ? XNN_VALUE_FLAG_EXTERNAL_OUTPUT
                  : XNN_VALUE_FLAG_EXTERNAL_INPUT,
          id);
      values.push_back(fb_xnnpack::CreateXValue(
          builder_, fb_xnnpack::XValueUnion::XNNTensorValue, tensor.Union()));
    }
    const auto add = fb_xnnpack::Create_XNNNode2x1(builder_, 0, 1, 2);
    const auto node = fb_xnnpack::CreateXNode(
        builder_, fb_xnnpack::XNodeUnion::XNNAdd, add.Union());
    const auto graph = fb_xnnpack::CreateXNNGraph(
        builder_,
        builder_.CreateString("0"),
        builder_.CreateVector(&node, 1),
        builder_.CreateVector(values),
        3,
        builder_.CreateVector<uint32_t>({0, 1}),
        builder_.CreateVector<uint32_t>({2}));
    fb_xnnpack::FinishXNNGraphBuffer(builder_, graph);
  }

  void TearDown() override {
    EXPECT_EQ(
        runtime::set_option(xnnpack_backend_key, original_options_.view()),
        Error::Ok);
  }

  runtime::Result<uint32_t> create_runtime_flags(
      Span<runtime::BackendOption> specs) {
    std::array<uint8_t, 4096> memory{};
    runtime::MemoryAllocator allocator(memory.size(), memory.data());
    runtime::BackendInitContext context(
        &allocator, nullptr, nullptr, nullptr, {specs.data(), specs.size()});
    runtime::FreeableBuffer processed(
        builder_.GetBufferPointer(), builder_.GetSize(), nullptr);
    auto handle = backend_->init(context, &processed, {});
    if (!handle.ok()) {
      return handle.error();
    }
    const auto* executor = static_cast<XNNExecutor*>(handle.get());
    const uint32_t flags = executor->runtime_->flags;
    backend_->destroy(handle.get());
    return flags;
  }

  runtime::BackendInterface* backend_{nullptr};
  runtime::BackendOptions<1> original_options_;
  flatbuffers::FlatBufferBuilder builder_;
};

TEST_F(XNNPackedLhsTest, RuntimeSpecOverridesGlobalDefault) {
  runtime::BackendOptions<1> global_options;
  ASSERT_EQ(
      global_options.set_option(disable_packed_lhs_option_key, false),
      Error::Ok);
  ASSERT_EQ(
      runtime::set_option(xnnpack_backend_key, global_options.view()),
      Error::Ok);
  runtime::BackendOptions<2> specs;
  ASSERT_EQ(specs.set_option(weight_cache_option_key, false), Error::Ok);
  const auto baseline = create_runtime_flags(specs.view());
  ASSERT_TRUE(baseline.ok());
#if defined(XNN_FLAG_NO_PACKED_LHS)
  ASSERT_EQ(baseline.get() & XNN_FLAG_NO_PACKED_LHS, 0);
#endif

  for (bool global_default : {false, true}) {
    SCOPED_TRACE(global_default);
    ASSERT_EQ(
        global_options.set_option(
            disable_packed_lhs_option_key, global_default),
        Error::Ok);
    ASSERT_EQ(
        runtime::set_option(xnnpack_backend_key, global_options.view()),
        Error::Ok);
    for (auto override_value :
         {std::optional<bool>{},
          std::optional<bool>{false},
          std::optional<bool>{true}}) {
      SCOPED_TRACE(::testing::PrintToString(override_value));
      runtime::BackendOptions<2> model_options;
      ASSERT_EQ(
          model_options.set_option(weight_cache_option_key, false), Error::Ok);
      if (override_value.has_value()) {
        ASSERT_EQ(
            model_options.set_option(
                disable_packed_lhs_option_key, *override_value),
            Error::Ok);
      }
      const auto flags = create_runtime_flags(model_options.view());
      ASSERT_TRUE(flags.ok());
      uint32_t expected_flags = baseline.get();
#if defined(XNN_FLAG_NO_PACKED_LHS)
      if (override_value.value_or(global_default)) {
        expected_flags |= XNN_FLAG_NO_PACKED_LHS;
      }
#endif
      EXPECT_EQ(flags.get(), expected_flags);
      ASSERT_EQ(
          runtime::get_option(xnnpack_backend_key, global_options.view()),
          Error::Ok);
      EXPECT_EQ(std::get<bool>(global_options.view()[0].value), global_default);
    }
  }
}

TEST(XNNCompilerTest, Bf16BatchMatrixMultiplyOutputWithoutDims) {
  runtime::runtime_init();
  auto* backend = runtime::get_backend_class(xnnpack_backend_key);
  ASSERT_NE(backend, nullptr);

  flatbuffers::FlatBufferBuilder builder;
  const std::vector<std::vector<uint32_t>> input_dims = {{2, 3, 4}, {2, 4, 6}};
  std::vector<flatbuffers::Offset<fb_xnnpack::XValue>> values;
  for (uint32_t id = 0; id < 3; ++id) {
    flatbuffers::Offset<flatbuffers::Vector<uint32_t>> dims;
    if (id < 2) {
      dims = builder.CreateVector(input_dims[id]);
    }
    const auto tensor = fb_xnnpack::CreateXNNTensorValue(
        builder,
        fb_xnnpack::XNNDatatype::xnn_datatype_bf16,
        id < 2 ? 3 : 0,
        dims,
        0,
        id,
        id == 2 ? XNN_VALUE_FLAG_EXTERNAL_OUTPUT
                : XNN_VALUE_FLAG_EXTERNAL_INPUT,
        id);
    values.push_back(fb_xnnpack::CreateXValue(
        builder, fb_xnnpack::XValueUnion::XNNTensorValue, tensor.Union()));
  }
  const auto bmm = fb_xnnpack::Create_XNNNode2x1(builder, 0, 1, 2);
  const auto node = fb_xnnpack::CreateXNode(
      builder, fb_xnnpack::XNodeUnion::XNNBatchMatrixMultiply, bmm.Union());
  const auto graph = fb_xnnpack::CreateXNNGraph(
      builder,
      builder.CreateString("0"),
      builder.CreateVector(&node, 1),
      builder.CreateVector(values),
      3,
      builder.CreateVector<uint32_t>({0, 1}),
      builder.CreateVector<uint32_t>({2}));
  fb_xnnpack::FinishXNNGraphBuffer(builder, graph);

  std::array<uint8_t, 4096> memory{};
  runtime::MemoryAllocator allocator(memory.size(), memory.data());
  runtime::BackendInitContext context(&allocator);
  runtime::FreeableBuffer processed(
      builder.GetBufferPointer(), builder.GetSize(), nullptr);
  // XNNPACK works out the output shape itself, so the graph still loads.
  auto handle = backend->init(context, &processed, {});
  ASSERT_TRUE(handle.ok());
  backend->destroy(handle.get());
}

} // namespace executorch::backends::xnnpack::delegate

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
