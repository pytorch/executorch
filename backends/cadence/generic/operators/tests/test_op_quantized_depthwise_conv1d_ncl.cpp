/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <optional>

#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <gtest/gtest.h>

namespace impl {
namespace generic {
namespace native {

// Forward declarations: the depthwise conv1d NCL entry points are defined in
// op_quantized_depthwise_conv1d_ncl.cpp but not exposed through a header.
::executorch::aten::Tensor& quantized_depthwise_conv1d_ncl_per_tensor_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& input,
    const ::executorch::aten::Tensor& weight,
    const ::executorch::aten::Tensor& bias,
    ::executorch::aten::IntArrayRef stride,
    ::executorch::aten::IntArrayRef padding,
    ::executorch::aten::IntArrayRef dilation,
    int64_t groups,
    int64_t input_zero_point,
    int64_t weight_zero_point,
    double bias_scale,
    double output_scale,
    int64_t output_zero_point,
    int64_t out_multiplier,
    int64_t out_shift,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor& quantized_depthwise_conv1d_ncl_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& input,
    const ::executorch::aten::Tensor& weight,
    const ::executorch::aten::Tensor& bias,
    ::executorch::aten::IntArrayRef stride,
    ::executorch::aten::IntArrayRef padding,
    ::executorch::aten::IntArrayRef dilation,
    int64_t groups,
    int64_t input_zero_point,
    const std::optional<::executorch::aten::Tensor>& weight_zero_point,
    const ::executorch::aten::Tensor& bias_scale,
    double output_scale,
    int64_t output_zero_point,
    const ::executorch::aten::Tensor& out_multiplier,
    const ::executorch::aten::Tensor& out_shift,
    ::executorch::aten::Tensor& out);

namespace {

using ::executorch::aten::IntArrayRef;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::testing::TensorFactory;

class GenericQuantizedDepthwiseConv1dNclPerChannelTest : public OperatorTest {
 protected:
  static IntArrayRef ref(const std::array<int64_t, 1>& a) {
    return IntArrayRef(a.data(), a.size());
  }
};

// Depthwise NCL layout: input [N, C, L], groups == C, weight [OC, 1, WW].
// With channel_multiplier = 1, out_channels = in_channels = groups.
// Setup (int8, all zero_points = 0, output_scale = 1.0):
//   input  [1, 3, 4] rows per input channel:
//     ic 0: [1, 2, 3, 4]
//     ic 1: [2, 3, 4, 5]
//     ic 2: [1, 1, 1, 1]
//   weight [3, 1, 1] = [[[2]], [[2]], [[2]]]           (kernel_size = 1)
//   bias   [3]       = [0, 0, 0]
//   weight_zero_point [3] = [1, 0, 2]
//     -> effective weight per output channel: 1, 2, 0
//   bias_scale [3]        = [1.0f, 2.0f, 5.0f]
//
// Because depthwise conv binds oc <-> ic, each output channel only reads its
// own input channel.
// Per-position accumulator per output channel:
//   oc 0 (uses ic 0, w=1): [1, 2, 3, 4]
//   oc 1 (uses ic 1, w=2): [4, 6, 8, 10]
//   oc 2 (uses ic 2, w=0): [0, 0, 0, 0]
//
// Output NCL [1, 3, 4] after bias_scale * acc, quantized with out_scale=1:
//   oc 0 (bs=1.0):  [1,  2,  3,  4]
//   oc 1 (bs=2.0):  [8, 12, 16, 20]
//   oc 2 (bs=5.0):  [0,  0,  0,  0]
//
// These per-channel bias_scales are DISTINCT so the test would fail if the
// kernel incorrectly broadcast bias_scale[0] to every output channel.
TEST_F(
    GenericQuantizedDepthwiseConv1dNclPerChannelTest,
    PerChannelCorrectnessInt8) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Int> tf_int32;
  TensorFactory<ScalarType::Float> tf_float;

  Tensor input = tf_int8.make(
      {1, 3, 4},
      {
          /*ic 0*/ 1,
          2,
          3,
          4,
          /*ic 1*/ 2,
          3,
          4,
          5,
          /*ic 2*/ 1,
          1,
          1,
          1,
      });
  Tensor weight = tf_int8.make({3, 1, 1}, {2, 2, 2});
  Tensor bias = tf_int32.zeros({3});
  Tensor weight_zero_point = tf_int32.make({3}, {1, 0, 2});
  Tensor bias_scale = tf_float.make({3}, {1.0f, 2.0f, 5.0f});
  Tensor out_multiplier = tf_int32.make({1}, {0});
  Tensor out_shift = tf_int32.make({1}, {0});
  Tensor out = tf_int8.zeros({1, 3, 4});

  std::array<int64_t, 1> stride_arr = {1};
  std::array<int64_t, 1> padding_arr = {0};
  std::array<int64_t, 1> dilation_arr = {1};

  quantized_depthwise_conv1d_ncl_out(
      context_,
      input,
      weight,
      bias,
      ref(stride_arr),
      ref(padding_arr),
      ref(dilation_arr),
      /*groups=*/3,
      /*input_zero_point=*/0,
      weight_zero_point,
      bias_scale,
      /*output_scale=*/1.0,
      /*output_zero_point=*/0,
      out_multiplier,
      out_shift,
      out);

  Tensor expected = tf_int8.make(
      {1, 3, 4},
      {
          /*oc 0*/ 1,
          2,
          3,
          4,
          /*oc 1*/ 8,
          12,
          16,
          20,
          /*oc 2*/ 0,
          0,
          0,
          0,
      });
  EXPECT_TENSOR_EQ(out, expected);
}

// Degeneracy: length-1 qparam tensors through the per-channel _out entry point
// must produce the same output as the per_tensor_out entry point with the
// equivalent scalar qparams.
TEST_F(
    GenericQuantizedDepthwiseConv1dNclPerChannelTest,
    LengthOneMatchesPerTensorOut) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Int> tf_int32;
  TensorFactory<ScalarType::Float> tf_float;

  Tensor input = tf_int8.make(
      {1, 3, 4},
      {
          /*ic 0*/ 1,
          2,
          3,
          4,
          /*ic 1*/ 2,
          3,
          4,
          5,
          /*ic 2*/ 1,
          1,
          1,
          1,
      });
  Tensor weight = tf_int8.make({3, 1, 1}, {2, 2, 2});
  Tensor bias = tf_int32.zeros({3});

  const int64_t scalar_wzp = 1;
  const double scalar_bs = 2.0;

  Tensor weight_zero_point =
      tf_int32.make({1}, {static_cast<int32_t>(scalar_wzp)});
  Tensor bias_scale = tf_float.make({1}, {static_cast<float>(scalar_bs)});
  Tensor out_multiplier = tf_int32.make({1}, {0});
  Tensor out_shift = tf_int32.make({1}, {0});

  std::array<int64_t, 1> stride_arr = {1};
  std::array<int64_t, 1> padding_arr = {0};
  std::array<int64_t, 1> dilation_arr = {1};

  Tensor out_pc = tf_int8.zeros({1, 3, 4});
  quantized_depthwise_conv1d_ncl_out(
      context_,
      input,
      weight,
      bias,
      ref(stride_arr),
      ref(padding_arr),
      ref(dilation_arr),
      /*groups=*/3,
      /*input_zero_point=*/0,
      weight_zero_point,
      bias_scale,
      /*output_scale=*/1.0,
      /*output_zero_point=*/0,
      out_multiplier,
      out_shift,
      out_pc);

  Tensor out_pt = tf_int8.zeros({1, 3, 4});
  quantized_depthwise_conv1d_ncl_per_tensor_out(
      context_,
      input,
      weight,
      bias,
      ref(stride_arr),
      ref(padding_arr),
      ref(dilation_arr),
      /*groups=*/3,
      /*input_zero_point=*/0,
      scalar_wzp,
      scalar_bs,
      /*output_scale=*/1.0,
      /*output_zero_point=*/0,
      /*out_multiplier=*/0,
      /*out_shift=*/0,
      out_pt);

  EXPECT_TENSOR_EQ(out_pc, out_pt);
}

// Absent weight_zero_point means a symmetric weight: every channel's zero point
// is zero, so the result must match an explicit all-zero tensor.
// Same input as above with weight 2 everywhere and bias_scale [1, 2, 5]:
//   oc 0 (bs=1.0): [2,  4,  6,  8]
//   oc 1 (bs=2.0): [8, 12, 16, 20]
//   oc 2 (bs=5.0): [10, 10, 10, 10]
TEST_F(
    GenericQuantizedDepthwiseConv1dNclPerChannelTest,
    AbsentWeightZeroPointMatchesAllZeros) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Int> tf_int32;
  TensorFactory<ScalarType::Float> tf_float;

  Tensor input = tf_int8.make(
      {1, 3, 4},
      {
          /*ic 0*/ 1,
          2,
          3,
          4,
          /*ic 1*/ 2,
          3,
          4,
          5,
          /*ic 2*/ 1,
          1,
          1,
          1,
      });
  Tensor weight = tf_int8.make({3, 1, 1}, {2, 2, 2});
  Tensor bias = tf_int32.zeros({3});
  Tensor bias_scale = tf_float.make({3}, {1.0f, 2.0f, 5.0f});
  Tensor out_multiplier = tf_int32.make({1}, {0});
  Tensor out_shift = tf_int32.make({1}, {0});

  std::array<int64_t, 1> stride_arr = {1};
  std::array<int64_t, 1> padding_arr = {0};
  std::array<int64_t, 1> dilation_arr = {1};

  const auto run = [&](const std::optional<Tensor>& weight_zero_point) {
    Tensor out = tf_int8.zeros({1, 3, 4});
    quantized_depthwise_conv1d_ncl_out(
        context_,
        input,
        weight,
        bias,
        ref(stride_arr),
        ref(padding_arr),
        ref(dilation_arr),
        /*groups=*/3,
        /*input_zero_point=*/0,
        weight_zero_point,
        bias_scale,
        /*output_scale=*/1.0,
        /*output_zero_point=*/0,
        out_multiplier,
        out_shift,
        out);
    return out;
  };

  Tensor expected = tf_int8.make(
      {1, 3, 4},
      {
          /*oc 0*/ 2,
          4,
          6,
          8,
          /*oc 1*/ 8,
          12,
          16,
          20,
          /*oc 2*/ 10,
          10,
          10,
          10,
      });
  EXPECT_TENSOR_EQ(run(std::nullopt), expected);
  EXPECT_TENSOR_EQ(run(tf_int32.zeros({3})), expected);
}

} // namespace
} // namespace native
} // namespace generic
} // namespace impl
