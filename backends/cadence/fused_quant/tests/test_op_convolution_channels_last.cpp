/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/cadence/fused_quant/op_convolution_channels_last.h>
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

using executorch::aten::IntArrayRef;
using executorch::aten::optional;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::testing::TensorFactory;

namespace {

optional<Tensor> none_tensor() {
  return optional<Tensor>();
}

optional<int64_t> none_axis() {
  return optional<int64_t>();
}

} // namespace

class FusedQuantConvolutionChannelsLastTest : public OperatorTest {};

// 1x1 convolution, all quantized (int8 inp, int8 weight, int8 out)
// inp NHWC [1,2,2,1], weight OHWI [1,1,1,1]
TEST_F(FusedQuantConvolutionChannelsLastTest, Conv1x1AllQuantized) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  // inp: NHWC [1, 2, 2, 1]
  Tensor inp = tf_int8.make({1, 2, 2, 1}, {2, 4, 6, 8});
  // weight: OHWI [1, 1, 1, 1]
  Tensor weight = tf_int8.make({1, 1, 1, 1}, {2});

  Tensor inp_scale = tf_float.make({1}, {0.5});
  Tensor inp_zp = tf_long.make({1}, {0});
  Tensor weight_scale = tf_float.make({1}, {0.5});
  Tensor weight_zp = tf_long.make({1}, {0});
  Tensor out_scale = tf_float.make({1}, {0.5});
  Tensor out_zp = tf_long.make({1}, {0});

  // out: NHWC [1, 2, 2, 1]
  Tensor out = tf_int8.zeros({1, 2, 2, 1});

  // dequant inp (scale=0.5, zp=0): {1.0, 2.0, 3.0, 4.0}
  // dequant weight (scale=0.5, zp=0): {1.0}
  // conv 1x1: each output = inp_val * 1.0 = {1.0, 2.0, 3.0, 4.0}
  // requant (scale=0.5, zp=0): round(val/0.5) = {2, 4, 6, 8}

  int64_t stride_arr[] = {1, 1};
  int64_t padding_arr[] = {0, 0};
  int64_t dilation_arr[] = {1, 1};
  IntArrayRef stride(stride_arr, 2);
  IntArrayRef padding(padding_arr, 2);
  IntArrayRef dilation(dilation_arr, 2);

  cadence::fused_quant::native::convolution_channels_last_out(
      context_,
      inp,
      weight,
      none_tensor(),
      // inp qparams
      optional<Tensor>(inp_scale),
      optional<Tensor>(inp_zp),
      ScalarType::Float,
      -128,
      127,
      none_axis(),
      // weight qparams
      optional<Tensor>(weight_scale),
      optional<Tensor>(weight_zp),
      ScalarType::Float,
      -128,
      127,
      none_axis(),
      // bias qparams
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      // out qparams
      optional<Tensor>(out_scale),
      optional<Tensor>(out_zp),
      ScalarType::Char,
      -128,
      127,
      none_axis(),
      // conv params
      stride,
      padding,
      dilation,
      /*groups=*/1,
      out);

  EXPECT_TENSOR_EQ(out, tf_int8.make({1, 2, 2, 1}, {2, 4, 6, 8}));
}

// 3x3 convolution with padding=1, all quantized
// inp NHWC [1,3,3,1], weight OHWI [1,3,3,1], padding=1
TEST_F(FusedQuantConvolutionChannelsLastTest, Conv3x3WithPadding) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  // inp: NHWC [1, 3, 3, 1], all values = 2
  Tensor inp = tf_int8.make({1, 3, 3, 1}, {2, 2, 2, 2, 2, 2, 2, 2, 2});
  // weight: OHWI [1, 3, 3, 1], all values = 2
  Tensor weight = tf_int8.make({1, 3, 3, 1}, {2, 2, 2, 2, 2, 2, 2, 2, 2});

  Tensor inp_scale = tf_float.make({1}, {0.5});
  Tensor inp_zp = tf_long.make({1}, {0});
  Tensor weight_scale = tf_float.make({1}, {0.5});
  Tensor weight_zp = tf_long.make({1}, {0});
  Tensor out_scale = tf_float.make({1}, {1.0});
  Tensor out_zp = tf_long.make({1}, {0});

  // out: NHWC [1, 3, 3, 1]
  Tensor out = tf_int8.zeros({1, 3, 3, 1});

  // dequant inp (scale=0.5, zp=0): all 1.0
  // dequant weight (scale=0.5, zp=0): all 1.0
  // With padding=1, stride=1, dilation=1 on a 3x3 input:
  //   Corner (0,0): 4 contributing pixels -> sum = 4.0
  //   Edge   (0,1): 6 contributing pixels -> sum = 6.0
  //   Center (1,1): 9 contributing pixels -> sum = 9.0
  // Output NHWC [1,3,3,1]:
  //   [4, 6, 4, 6, 9, 6, 4, 6, 4]
  // requant (scale=1.0, zp=0): {4, 6, 4, 6, 9, 6, 4, 6, 4}

  int64_t stride_arr[] = {1, 1};
  int64_t padding_arr[] = {1, 1};
  int64_t dilation_arr[] = {1, 1};
  IntArrayRef stride(stride_arr, 2);
  IntArrayRef padding(padding_arr, 2);
  IntArrayRef dilation(dilation_arr, 2);

  cadence::fused_quant::native::convolution_channels_last_out(
      context_,
      inp,
      weight,
      none_tensor(),
      // inp qparams
      optional<Tensor>(inp_scale),
      optional<Tensor>(inp_zp),
      ScalarType::Float,
      -128,
      127,
      none_axis(),
      // weight qparams
      optional<Tensor>(weight_scale),
      optional<Tensor>(weight_zp),
      ScalarType::Float,
      -128,
      127,
      none_axis(),
      // bias qparams
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      // out qparams
      optional<Tensor>(out_scale),
      optional<Tensor>(out_zp),
      ScalarType::Char,
      -128,
      127,
      none_axis(),
      // conv params
      stride,
      padding,
      dilation,
      /*groups=*/1,
      out);

  EXPECT_TENSOR_EQ(
      out, tf_int8.make({1, 3, 3, 1}, {4, 6, 4, 6, 9, 6, 4, 6, 4}));
}

// Float inputs, quantized int8 output
TEST_F(FusedQuantConvolutionChannelsLastTest, FloatInputsQuantizedOutput) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  // inp: NHWC [1, 2, 2, 1]
  Tensor inp = tf_float.make({1, 2, 2, 1}, {1.0, 2.0, 3.0, 4.0});
  // weight: OHWI [1, 1, 1, 1]
  Tensor weight = tf_float.make({1, 1, 1, 1}, {2.0});

  Tensor out_scale = tf_float.make({1}, {0.5});
  Tensor out_zp = tf_long.make({1}, {0});

  // out: NHWC [1, 2, 2, 1]
  Tensor out = tf_int8.zeros({1, 2, 2, 1});

  // float conv 1x1: {1*2, 2*2, 3*2, 4*2} = {2.0, 4.0, 6.0, 8.0}
  // requant (scale=0.5, zp=0): {4, 8, 12, 16}

  int64_t stride_arr[] = {1, 1};
  int64_t padding_arr[] = {0, 0};
  int64_t dilation_arr[] = {1, 1};
  IntArrayRef stride(stride_arr, 2);
  IntArrayRef padding(padding_arr, 2);
  IntArrayRef dilation(dilation_arr, 2);

  cadence::fused_quant::native::convolution_channels_last_out(
      context_,
      inp,
      weight,
      none_tensor(),
      // inp qparams (float, no quantization)
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      // weight qparams (float, no quantization)
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      // bias qparams
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      // out qparams
      optional<Tensor>(out_scale),
      optional<Tensor>(out_zp),
      ScalarType::Char,
      -128,
      127,
      none_axis(),
      // conv params
      stride,
      padding,
      dilation,
      /*groups=*/1,
      out);

  EXPECT_TENSOR_EQ(out, tf_int8.make({1, 2, 2, 1}, {4, 8, 12, 16}));
}

// Quantized int8 inputs, float output
TEST_F(FusedQuantConvolutionChannelsLastTest, QuantizedInputsFloatOutput) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  // inp: NHWC [1, 2, 2, 1]
  Tensor inp = tf_int8.make({1, 2, 2, 1}, {2, 4, 6, 8});
  // weight: OHWI [1, 1, 1, 1]
  Tensor weight = tf_int8.make({1, 1, 1, 1}, {2});

  Tensor inp_scale = tf_float.make({1}, {0.5});
  Tensor inp_zp = tf_long.make({1}, {0});
  Tensor weight_scale = tf_float.make({1}, {0.5});
  Tensor weight_zp = tf_long.make({1}, {0});

  // out: NHWC [1, 2, 2, 1], float
  Tensor out = tf_float.zeros({1, 2, 2, 1});

  // dequant inp: {1.0, 2.0, 3.0, 4.0}
  // dequant weight: {1.0}
  // conv 1x1: {1.0, 2.0, 3.0, 4.0}
  // no requant (float output)

  int64_t stride_arr[] = {1, 1};
  int64_t padding_arr[] = {0, 0};
  int64_t dilation_arr[] = {1, 1};
  IntArrayRef stride(stride_arr, 2);
  IntArrayRef padding(padding_arr, 2);
  IntArrayRef dilation(dilation_arr, 2);

  cadence::fused_quant::native::convolution_channels_last_out(
      context_,
      inp,
      weight,
      none_tensor(),
      // inp qparams
      optional<Tensor>(inp_scale),
      optional<Tensor>(inp_zp),
      ScalarType::Float,
      -128,
      127,
      none_axis(),
      // weight qparams
      optional<Tensor>(weight_scale),
      optional<Tensor>(weight_zp),
      ScalarType::Float,
      -128,
      127,
      none_axis(),
      // bias qparams
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      // out qparams (float, no quantization)
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      // conv params
      stride,
      padding,
      dilation,
      /*groups=*/1,
      out);

  EXPECT_TENSOR_EQ(out, tf_float.make({1, 2, 2, 1}, {1.0, 2.0, 3.0, 4.0}));
}

// 1x1 conv with bias, all quantized
TEST_F(FusedQuantConvolutionChannelsLastTest, WithBias) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  // inp: NHWC [1, 2, 2, 1]
  Tensor inp = tf_int8.make({1, 2, 2, 1}, {2, 4, 6, 8});
  // weight: OHWI [1, 1, 1, 1]
  Tensor weight = tf_int8.make({1, 1, 1, 1}, {2});
  // bias: [1], float
  Tensor bias = tf_float.make({1}, {0.5});

  Tensor inp_scale = tf_float.make({1}, {0.5});
  Tensor inp_zp = tf_long.make({1}, {0});
  Tensor weight_scale = tf_float.make({1}, {0.5});
  Tensor weight_zp = tf_long.make({1}, {0});
  Tensor out_scale = tf_float.make({1}, {0.5});
  Tensor out_zp = tf_long.make({1}, {0});

  // out: NHWC [1, 2, 2, 1]
  Tensor out = tf_int8.zeros({1, 2, 2, 1});

  // dequant inp: {1.0, 2.0, 3.0, 4.0}
  // dequant weight: {1.0}
  // conv 1x1 + bias(0.5): {1.5, 2.5, 3.5, 4.5}
  // requant (scale=0.5, zp=0): round(1.5/0.5)=3, round(2.5/0.5)=5,
  //   round(3.5/0.5)=7, round(4.5/0.5)=9

  int64_t stride_arr[] = {1, 1};
  int64_t padding_arr[] = {0, 0};
  int64_t dilation_arr[] = {1, 1};
  IntArrayRef stride(stride_arr, 2);
  IntArrayRef padding(padding_arr, 2);
  IntArrayRef dilation(dilation_arr, 2);

  cadence::fused_quant::native::convolution_channels_last_out(
      context_,
      inp,
      weight,
      optional<Tensor>(bias),
      // inp qparams
      optional<Tensor>(inp_scale),
      optional<Tensor>(inp_zp),
      ScalarType::Float,
      -128,
      127,
      none_axis(),
      // weight qparams
      optional<Tensor>(weight_scale),
      optional<Tensor>(weight_zp),
      ScalarType::Float,
      -128,
      127,
      none_axis(),
      // bias qparams (float, no quantization)
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      // out qparams
      optional<Tensor>(out_scale),
      optional<Tensor>(out_zp),
      ScalarType::Char,
      -128,
      127,
      none_axis(),
      // conv params
      stride,
      padding,
      dilation,
      /*groups=*/1,
      out);

  // Note: std::round uses banker's rounding for .5 cases in some
  // implementations, but our quantize uses std::round which rounds
  // half away from zero: round(2.5/0.5) = round(5.0) = 5,
  // round(3.5/0.5) = round(7.0) = 7, etc.
  // Actually val/scale: 1.5/0.5=3.0, 2.5/0.5=5.0, 3.5/0.5=7.0, 4.5/0.5=9.0
  // These are exact integers, so no rounding ambiguity.
  EXPECT_TENSOR_EQ(out, tf_int8.make({1, 2, 2, 1}, {3, 5, 7, 9}));
}

// Grouped convolution: 2 groups, 2 input channels, 2 output channels
// Each group processes 1 input channel and produces 1 output channel.
TEST_F(FusedQuantConvolutionChannelsLastTest, GroupedConvolution) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  // inp: NHWC [1, 2, 2, 2] - 2 input channels
  // Channels interleaved in NHWC:
  //   pixel(0,0): ch0=2, ch1=4
  //   pixel(0,1): ch0=6, ch1=8
  //   pixel(1,0): ch0=10, ch1=12
  //   pixel(1,1): ch0=14, ch1=16
  Tensor inp = tf_int8.make({1, 2, 2, 2}, {2, 4, 6, 8, 10, 12, 14, 16});

  // weight: OHWI [2, 1, 1, 1] (C_out=2, kH=1, kW=1, C_in/groups=1)
  //   filter 0 (group 0): {2}
  //   filter 1 (group 1): {4}
  Tensor weight = tf_int8.make({2, 1, 1, 1}, {2, 4});

  Tensor inp_scale = tf_float.make({1}, {0.5});
  Tensor inp_zp = tf_long.make({1}, {0});
  Tensor weight_scale = tf_float.make({1}, {0.5});
  Tensor weight_zp = tf_long.make({1}, {0});
  Tensor out_scale = tf_float.make({1}, {0.5});
  Tensor out_zp = tf_long.make({1}, {0});

  // out: NHWC [1, 2, 2, 2]
  Tensor out = tf_int8.zeros({1, 2, 2, 2});

  // dequant inp (scale=0.5): {1,2,3,4,5,6,7,8}
  //   pixel(0,0): ch0=1.0, ch1=2.0
  //   pixel(0,1): ch0=3.0, ch1=4.0
  //   pixel(1,0): ch0=5.0, ch1=6.0
  //   pixel(1,1): ch0=7.0, ch1=8.0
  //
  // dequant weight (scale=0.5): filter0={1.0}, filter1={2.0}
  //
  // Group 0 (oc=0): inp ch0 * filter0
  //   pixel(0,0): 1.0*1.0 = 1.0
  //   pixel(0,1): 3.0*1.0 = 3.0
  //   pixel(1,0): 5.0*1.0 = 5.0
  //   pixel(1,1): 7.0*1.0 = 7.0
  //
  // Group 1 (oc=1): inp ch1 * filter1
  //   pixel(0,0): 2.0*2.0 = 4.0
  //   pixel(0,1): 4.0*2.0 = 8.0
  //   pixel(1,0): 6.0*2.0 = 12.0
  //   pixel(1,1): 8.0*2.0 = 16.0
  //
  // NHWC output [1,2,2,2]: interleaved by channel
  //   {1.0,4.0, 3.0,8.0, 5.0,12.0, 7.0,16.0}
  //
  // requant (scale=0.5, zp=0):
  //   {2,8, 6,16, 10,24, 14,32}

  int64_t stride_arr[] = {1, 1};
  int64_t padding_arr[] = {0, 0};
  int64_t dilation_arr[] = {1, 1};
  IntArrayRef stride(stride_arr, 2);
  IntArrayRef padding(padding_arr, 2);
  IntArrayRef dilation(dilation_arr, 2);

  cadence::fused_quant::native::convolution_channels_last_out(
      context_,
      inp,
      weight,
      none_tensor(),
      // inp qparams
      optional<Tensor>(inp_scale),
      optional<Tensor>(inp_zp),
      ScalarType::Float,
      -128,
      127,
      none_axis(),
      // weight qparams
      optional<Tensor>(weight_scale),
      optional<Tensor>(weight_zp),
      ScalarType::Float,
      -128,
      127,
      none_axis(),
      // bias qparams
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      // out qparams
      optional<Tensor>(out_scale),
      optional<Tensor>(out_zp),
      ScalarType::Char,
      -128,
      127,
      none_axis(),
      // conv params
      stride,
      padding,
      dilation,
      /*groups=*/2,
      out);

  EXPECT_TENSOR_EQ(
      out, tf_int8.make({1, 2, 2, 2}, {2, 8, 6, 16, 10, 24, 14, 32}));
}
