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

// Forward declarations: the depthwise conv1d NLC entry points are defined in
// op_quantized_depthwise_conv1d_nlc.cpp but not exposed through a header.
::executorch::aten::Tensor& quantized_depthwise_conv1d_nlc_per_tensor_out(
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

::executorch::aten::Tensor& quantized_depthwise_conv1d_nlc_out(
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

class GenericQuantizedDepthwiseConv1dNlcPerChannelTest : public OperatorTest {
 protected:
  static IntArrayRef ref(const std::array<int64_t, 1>& a) {
    return IntArrayRef(a.data(), a.size());
  }
};

// Depthwise NLC layout: input [N, L, C], groups == C, weight [OC, WW, 1].
// Same numeric construction as the NCL depthwise test, reinterpreted for NLC.
// Setup (int8, all zero_points = 0, output_scale = 1.0):
//   input NLC [1, 4, 3] (position-major, channel innermost):
//     pos 0: [ic0=1, ic1=2, ic2=1]
//     pos 1: [ic0=2, ic1=3, ic2=1]
//     pos 2: [ic0=3, ic1=4, ic2=1]
//     pos 3: [ic0=4, ic1=5, ic2=1]
//   weight [3, 1, 1] = [[[2]], [[2]], [[2]]]
//   bias   [3]       = [0, 0, 0]
//   weight_zero_point [3] = [1, 0, 2]
//     -> effective weight per output channel: 1, 2, 0
//   bias_scale [3]        = [1.0f, 2.0f, 5.0f]
//
// Per-position accumulator per output channel (each oc reads its own ic):
//   oc 0 (uses ic 0, w=1): [1, 2, 3, 4]
//   oc 1 (uses ic 1, w=2): [4, 6, 8, 10]
//   oc 2 (uses ic 2, w=0): [0, 0, 0, 0]
//
// Output NLC [1, 4, 3], laid out position-major:
//   pos 0: [oc0=1, oc1=8,  oc2=0]
//   pos 1: [oc0=2, oc1=12, oc2=0]
//   pos 2: [oc0=3, oc1=16, oc2=0]
//   pos 3: [oc0=4, oc1=20, oc2=0]
//
// These per-channel bias_scales are DISTINCT so the test would fail if the
// kernel incorrectly broadcast bias_scale[0] to every output channel.
TEST_F(
    GenericQuantizedDepthwiseConv1dNlcPerChannelTest,
    PerChannelCorrectnessInt8) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Int> tf_int32;
  TensorFactory<ScalarType::Float> tf_float;

  Tensor input = tf_int8.make(
      {1, 4, 3},
      {
          /*pos 0*/ 1,
          2,
          1,
          /*pos 1*/ 2,
          3,
          1,
          /*pos 2*/ 3,
          4,
          1,
          /*pos 3*/ 4,
          5,
          1,
      });
  Tensor weight = tf_int8.make({3, 1, 1}, {2, 2, 2});
  Tensor bias = tf_int32.zeros({3});
  Tensor weight_zero_point = tf_int32.make({3}, {1, 0, 2});
  Tensor bias_scale = tf_float.make({3}, {1.0f, 2.0f, 5.0f});
  Tensor out_multiplier = tf_int32.make({1}, {0});
  Tensor out_shift = tf_int32.make({1}, {0});
  Tensor out = tf_int8.zeros({1, 4, 3});

  std::array<int64_t, 1> stride_arr = {1};
  std::array<int64_t, 1> padding_arr = {0};
  std::array<int64_t, 1> dilation_arr = {1};

  quantized_depthwise_conv1d_nlc_out(
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
      {1, 4, 3},
      {
          /*pos 0*/ 1,
          8,
          0,
          /*pos 1*/ 2,
          12,
          0,
          /*pos 2*/ 3,
          16,
          0,
          /*pos 3*/ 4,
          20,
          0,
      });
  EXPECT_TENSOR_EQ(out, expected);
}

// Degeneracy: length-1 qparam tensors through the per-channel _out entry point
// must produce the same output as the per_tensor_out entry point with the
// equivalent scalar qparams.
TEST_F(
    GenericQuantizedDepthwiseConv1dNlcPerChannelTest,
    LengthOneMatchesPerTensorOut) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Int> tf_int32;
  TensorFactory<ScalarType::Float> tf_float;

  Tensor input = tf_int8.make(
      {1, 4, 3},
      {
          /*pos 0*/ 1,
          2,
          1,
          /*pos 1*/ 2,
          3,
          1,
          /*pos 2*/ 3,
          4,
          1,
          /*pos 3*/ 4,
          5,
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

  Tensor out_pc = tf_int8.zeros({1, 4, 3});
  quantized_depthwise_conv1d_nlc_out(
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

  Tensor out_pt = tf_int8.zeros({1, 4, 3});
  quantized_depthwise_conv1d_nlc_per_tensor_out(
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
//   pos 0: [oc0=2, oc1=8,  oc2=10]
//   pos 1: [oc0=4, oc1=12, oc2=10]
//   pos 2: [oc0=6, oc1=16, oc2=10]
//   pos 3: [oc0=8, oc1=20, oc2=10]
TEST_F(
    GenericQuantizedDepthwiseConv1dNlcPerChannelTest,
    AbsentWeightZeroPointMatchesAllZeros) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Int> tf_int32;
  TensorFactory<ScalarType::Float> tf_float;

  Tensor input = tf_int8.make(
      {1, 4, 3},
      {
          /*pos 0*/ 1,
          2,
          1,
          /*pos 1*/ 2,
          3,
          1,
          /*pos 2*/ 3,
          4,
          1,
          /*pos 3*/ 4,
          5,
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
    Tensor out = tf_int8.zeros({1, 4, 3});
    quantized_depthwise_conv1d_nlc_out(
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
      {1, 4, 3},
      {
          /*pos 0*/ 2,
          8,
          10,
          /*pos 1*/ 4,
          12,
          10,
          /*pos 2*/ 6,
          16,
          10,
          /*pos 3*/ 8,
          20,
          10,
      });
  EXPECT_TENSOR_EQ(run(std::nullopt), expected);
  EXPECT_TENSOR_EQ(run(tf_int32.zeros({3})), expected);
}

} // namespace
} // namespace native
} // namespace generic
} // namespace impl
