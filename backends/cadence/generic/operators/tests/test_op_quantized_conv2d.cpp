/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_quantized_conv2d.h>

#include <array>

#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <gtest/gtest.h>
#include <optional>

namespace impl {
namespace generic {
namespace native {
namespace {

using ::executorch::aten::IntArrayRef;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::testing::TensorFactory;

class GenericQuantizedConv2dPerChannelTest : public OperatorTest {
 protected:
  static IntArrayRef ref(const std::array<int64_t, 2>& a) {
    return IntArrayRef(a.data(), a.size());
  }
};

// ------------------------------------------------------------------
// NCHW per-channel correctness.
// ------------------------------------------------------------------
// Setup (int8, all input/output zero_points = 0, output_scale = 1.0):
//   input  [1,1,2,2] = [[[[1,2],[3,4]]]]
//   weight [3,1,1,1] = [[[[2]]],[[[2]]],[[[2]]]]      (3 output channels)
//   bias   [3]       = [0, 0, 0]
//   weight_zero_point [3] = [1, 0, 2]
//     -> effective weight per channel = 2 - wzp[oc]:  1, 2, 0
//   bias_scale [3]        = [1.0f, 5.0f, 10.0f]
//
// Per-pixel accumulator (single 1x1 kernel, single input channel):
//   ch 0: acc = input * 1   -> [1, 2, 3, 4]
//   ch 1: acc = input * 2   -> [2, 4, 6, 8]
//   ch 2: acc = input * 0   -> [0, 0, 0, 0]
//
// Requantize: out = clamp(nearbyint((bias_scale[oc] * acc) / output_scale))
//   ch 0 (bs=1.0):  [ 1,  2,  3,  4]
//   ch 1 (bs=5.0):  [10, 20, 30, 40]
//   ch 2 (bs=10.0): [ 0,  0,  0,  0]
//
// A regression that reads channel 0's qparams for every channel would give
// [1,2,3,4, 1,2,3,4, 1,2,3,4] instead of the expected output.
TEST_F(GenericQuantizedConv2dPerChannelTest, NchwPerChannelCorrectnessInt8) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Int> tf_int32;
  TensorFactory<ScalarType::Float> tf_float;

  Tensor input = tf_int8.make({1, 1, 2, 2}, {1, 2, 3, 4});
  Tensor weight = tf_int8.make({3, 1, 1, 1}, {2, 2, 2});
  Tensor bias = tf_int32.zeros({3});
  Tensor weight_zero_point = tf_int32.make({3}, {1, 0, 2});
  Tensor bias_scale = tf_float.make({3}, {1.0f, 5.0f, 10.0f});
  // Unused for conv.
  Tensor out_multiplier = tf_int32.make({1}, {0});
  Tensor out_shift = tf_int32.make({1}, {0});
  Tensor out = tf_int8.zeros({1, 3, 2, 2});

  std::array<int64_t, 2> stride_arr = {1, 1};
  std::array<int64_t, 2> padding_arr = {0, 0};
  std::array<int64_t, 2> dilation_arr = {1, 1};

  quantized_conv2d_nchw_out(
      context_,
      input,
      weight,
      bias,
      ref(stride_arr),
      ref(padding_arr),
      ref(dilation_arr),
      /*groups=*/1,
      /*in_zero_point=*/0,
      weight_zero_point,
      bias_scale,
      /*output_scale=*/1.0,
      /*output_zero_point=*/0,
      out_multiplier,
      out_shift,
      out);

  Tensor expected = tf_int8.make(
      {1, 3, 2, 2},
      {
          /*ch 0*/ 1,
          2,
          3,
          4,
          /*ch 1*/ 10,
          20,
          30,
          40,
          /*ch 2*/ 0,
          0,
          0,
          0,
      });
  EXPECT_TENSOR_EQ(out, expected);
}

// NCHW degeneracy: length-1 qparam tensors -> stride 0 -> same as per-tensor.
TEST_F(GenericQuantizedConv2dPerChannelTest, NchwLengthOneMatchesPerTensor) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Int> tf_int32;
  TensorFactory<ScalarType::Float> tf_float;

  Tensor input = tf_int8.make({1, 1, 2, 2}, {1, 2, 3, 4});
  Tensor weight = tf_int8.make({3, 1, 1, 1}, {2, 2, 2});
  Tensor bias = tf_int32.zeros({3});
  const int32_t scalar_wzp = 1;
  const float scalar_bs = 5.0f;

  Tensor weight_zero_point = tf_int32.make({1}, {scalar_wzp});
  Tensor bias_scale = tf_float.make({1}, {scalar_bs});
  Tensor out_multiplier = tf_int32.make({1}, {0});
  Tensor out_shift = tf_int32.make({1}, {0});

  std::array<int64_t, 2> stride_arr = {1, 1};
  std::array<int64_t, 2> padding_arr = {0, 0};
  std::array<int64_t, 2> dilation_arr = {1, 1};

  Tensor out_pc = tf_int8.zeros({1, 3, 2, 2});
  quantized_conv2d_nchw_out(
      context_,
      input,
      weight,
      bias,
      ref(stride_arr),
      ref(padding_arr),
      ref(dilation_arr),
      /*groups=*/1,
      /*in_zero_point=*/0,
      weight_zero_point,
      bias_scale,
      /*output_scale=*/1.0,
      /*output_zero_point=*/0,
      out_multiplier,
      out_shift,
      out_pc);

  Tensor out_pt = tf_int8.zeros({1, 3, 2, 2});
  quantized_conv2d_nchw_per_tensor_out(
      context_,
      input,
      weight,
      bias,
      ref(stride_arr),
      ref(padding_arr),
      ref(dilation_arr),
      /*groups=*/1,
      /*in_zero_point=*/0,
      /*weight_zero_point=*/scalar_wzp,
      /*bias_scale=*/scalar_bs,
      /*output_scale=*/1.0,
      /*output_zero_point=*/0,
      /*out_multiplier=*/0,
      /*out_shift=*/0,
      out_pt);

  EXPECT_TENSOR_EQ(out_pc, out_pt);
}

// ------------------------------------------------------------------
// NHWC per-channel correctness.
// ------------------------------------------------------------------
// Same numeric construction as NCHW, but interpreted in NHWC.
// input NHWC [1,2,2,1] = elements [1, 2, 3, 4]  (same values, single channel)
// weight (regular NHWC layout [OC, KH, KW, WC]) [3, 1, 1, 1] = [2, 2, 2]
// -> accumulator per pixel per channel same as NCHW test.
// Output NHWC [1, 2, 2, 3]:
//   pixel (0,0): [ 1, 10, 0]
//   pixel (0,1): [ 2, 20, 0]
//   pixel (1,0): [ 3, 30, 0]
//   pixel (1,1): [ 4, 40, 0]
TEST_F(GenericQuantizedConv2dPerChannelTest, NhwcPerChannelCorrectnessInt8) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Int> tf_int32;
  TensorFactory<ScalarType::Float> tf_float;

  Tensor input = tf_int8.make({1, 2, 2, 1}, {1, 2, 3, 4});
  // Regular NHWC weight layout: [OC, KH, KW, WC]
  Tensor weight = tf_int8.make({3, 1, 1, 1}, {2, 2, 2});
  Tensor bias = tf_int32.zeros({3});
  Tensor weight_zero_point = tf_int32.make({3}, {1, 0, 2});
  Tensor bias_scale = tf_float.make({3}, {1.0f, 5.0f, 10.0f});
  Tensor out_multiplier = tf_int32.make({1}, {0});
  Tensor out_shift = tf_int32.make({1}, {0});
  Tensor out = tf_int8.zeros({1, 2, 2, 3});

  std::array<int64_t, 2> stride_arr = {1, 1};
  std::array<int64_t, 2> padding_arr = {0, 0};
  std::array<int64_t, 2> dilation_arr = {1, 1};

  quantized_conv2d_nhwc_out(
      context_,
      input,
      weight,
      bias,
      ref(stride_arr),
      ref(padding_arr),
      ref(dilation_arr),
      /*groups=*/1,
      /*in_zero_point=*/0,
      weight_zero_point,
      bias_scale,
      /*output_scale=*/1.0,
      /*output_zero_point=*/0,
      out_multiplier,
      out_shift,
      out);

  Tensor expected = tf_int8.make(
      {1, 2, 2, 3},
      {
          /*pixel (0,0):*/ 1,
          10,
          0,
          /*pixel (0,1):*/ 2,
          20,
          0,
          /*pixel (1,0):*/ 3,
          30,
          0,
          /*pixel (1,1):*/ 4,
          40,
          0,
      });
  EXPECT_TENSOR_EQ(out, expected);
}

// NHWC degeneracy: length-1 qparam tensors -> same as per_tensor path.
TEST_F(GenericQuantizedConv2dPerChannelTest, NhwcLengthOneMatchesPerTensor) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Int> tf_int32;
  TensorFactory<ScalarType::Float> tf_float;

  Tensor input = tf_int8.make({1, 2, 2, 1}, {1, 2, 3, 4});
  Tensor weight = tf_int8.make({3, 1, 1, 1}, {2, 2, 2});
  Tensor bias = tf_int32.zeros({3});
  const int32_t scalar_wzp = 1;
  const float scalar_bs = 5.0f;

  Tensor weight_zero_point = tf_int32.make({1}, {scalar_wzp});
  Tensor bias_scale = tf_float.make({1}, {scalar_bs});
  Tensor out_multiplier = tf_int32.make({1}, {0});
  Tensor out_shift = tf_int32.make({1}, {0});

  std::array<int64_t, 2> stride_arr = {1, 1};
  std::array<int64_t, 2> padding_arr = {0, 0};
  std::array<int64_t, 2> dilation_arr = {1, 1};

  Tensor out_pc = tf_int8.zeros({1, 2, 2, 3});
  quantized_conv2d_nhwc_out(
      context_,
      input,
      weight,
      bias,
      ref(stride_arr),
      ref(padding_arr),
      ref(dilation_arr),
      /*groups=*/1,
      /*in_zero_point=*/0,
      weight_zero_point,
      bias_scale,
      /*output_scale=*/1.0,
      /*output_zero_point=*/0,
      out_multiplier,
      out_shift,
      out_pc);

  Tensor out_pt = tf_int8.zeros({1, 2, 2, 3});
  quantized_conv2d_nhwc_per_tensor_out(
      context_,
      input,
      weight,
      bias,
      ref(stride_arr),
      ref(padding_arr),
      ref(dilation_arr),
      /*groups=*/1,
      /*in_zero_point=*/0,
      /*weight_zero_point=*/scalar_wzp,
      /*bias_scale=*/scalar_bs,
      /*output_scale=*/1.0,
      /*output_zero_point=*/0,
      /*out_multiplier=*/0,
      /*out_shift=*/0,
      /*offset=*/std::nullopt,
      out_pt);

  EXPECT_TENSOR_EQ(out_pc, out_pt);
}

// ------------------------------------------------------------------
// Depthwise NHWC.
// ------------------------------------------------------------------
// NOTE: this entry point currently declares scalar (int64_t/double) qparams
// in its signature, so it is *not possible* to exercise real per-channel
// qparams through this API. We can only exercise the scalar (degeneracy)
// path here. The signature bug is called out in the report accompanying
// these tests.
//
// Depthwise is a specialisation of the general grouped convolution, so the
// general NHWC kernel with groups == channels is the reference. Note the
// weight layouts differ: depthwise packs it as [KH, KW, OC], the general path
// as [OC, KH, KW, IC/groups].
TEST_F(
    GenericQuantizedConv2dPerChannelTest,
    DepthwiseNhwcMatchesGeneralGroupedConv) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Int> tf_int32;

  // input NHWC [1, 2, 2, 3] -- 3 channels
  Tensor input = tf_int8.make(
      {1, 2, 2, 3},
      {
          1,
          2,
          3, // pixel (0,0)
          4,
          5,
          6, // pixel (0,1)
          7,
          8,
          9, // pixel (1,0)
          10,
          11,
          12, // pixel (1,1)
      });
  // Depthwise weight NHWC layout: [KH, KW, OC] (no IC dim).
  Tensor weight = tf_int8.make({1, 1, 3}, {2, 3, 4});
  Tensor bias = tf_int32.zeros({3});

  const int64_t scalar_wzp = 1;
  const double scalar_bs = 1.0;

  std::array<int64_t, 2> stride_arr = {1, 1};
  std::array<int64_t, 2> padding_arr = {0, 0};
  std::array<int64_t, 2> dilation_arr = {1, 1};

  Tensor out_pc = tf_int8.zeros({1, 2, 2, 3});
  quantized_conv2d_depthwise_nhwc_out(
      context_,
      input,
      weight,
      bias,
      ref(stride_arr),
      ref(padding_arr),
      ref(dilation_arr),
      /*groups=*/3,
      /*in_zero_point=*/0,
      scalar_wzp,
      scalar_bs,
      /*output_scale=*/1.0,
      /*output_zero_point=*/0,
      /*out_multiplier=*/0,
      /*out_shift=*/0,
      out_pc);

  Tensor weight_general = tf_int8.make({3, 1, 1, 1}, {2, 3, 4});
  Tensor out_ref = tf_int8.zeros({1, 2, 2, 3});
  quantized_conv2d_nhwc_per_tensor_out(
      context_,
      input,
      weight_general,
      bias,
      ref(stride_arr),
      ref(padding_arr),
      ref(dilation_arr),
      /*groups=*/3,
      /*in_zero_point=*/0,
      scalar_wzp,
      scalar_bs,
      /*output_scale=*/1.0,
      /*output_zero_point=*/0,
      /*out_multiplier=*/0,
      /*out_shift=*/0,
      /*offset=*/std::nullopt,
      out_ref);

  EXPECT_TENSOR_EQ(out_pc, out_ref);
}

// The uint8 typed NHWC entry. Expected values are aot/ref_implementations.py's:
// with both zero points 128, the two 2x2 windows sum to -6 and 48, plus the
// biases 10 and -4, times bias_scale 0.5, over output_scale 0.25, plus 100.
TEST_F(GenericQuantizedConv2dPerChannelTest, NhwcAsym8uPerTensor) {
  TensorFactory<ScalarType::Byte> tf_uint8;
  TensorFactory<ScalarType::Int> tf_int32;
  Tensor input = tf_uint8.make({1, 2, 2, 1}, {130, 126, 132, 120});
  Tensor weight =
      tf_uint8.make({2, 2, 2, 1}, {129, 130, 127, 128, 128, 128, 140, 128});
  Tensor bias = tf_int32.make({2}, {10, -4});
  Tensor out = tf_uint8.zeros({1, 1, 1, 2});
  std::array<int64_t, 2> stride_arr = {1, 1};
  std::array<int64_t, 2> padding_arr = {0, 0};
  std::array<int64_t, 2> dilation_arr = {1, 1};

  quantized_conv2d_nhwc_asym8uxsym8u_asym8u_per_tensor_out(
      context_,
      input,
      weight,
      bias,
      ref(stride_arr),
      ref(padding_arr),
      ref(dilation_arr),
      /*groups=*/1,
      /*in_zero_point=*/128,
      /*weight_zero_point=*/128,
      /*bias_scale=*/0.5,
      /*output_scale=*/0.25,
      /*output_zero_point=*/100,
      /*out_multiplier=*/0,
      /*out_shift=*/0,
      out);

  EXPECT_TENSOR_EQ(out, tf_uint8.make({1, 1, 1, 2}, {108, 188}));
}

} // namespace
} // namespace native
} // namespace generic
} // namespace impl
