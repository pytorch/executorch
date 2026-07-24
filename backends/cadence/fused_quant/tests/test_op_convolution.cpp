/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/cadence/fused_quant/op_convolution.h>
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

class FusedQuantConvolutionTest : public OperatorTest {};

// 1x1 conv, all quantized: int8 inp, int8 weight, no bias, int8 out
TEST_F(FusedQuantConvolutionTest, Conv1x1AllQuantized) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  // inp [1, 1, 2, 2] int8
  Tensor inp = tf_int8.make({1, 1, 2, 2}, {2, 4, 6, 8});
  // weight [1, 1, 1, 1] int8
  Tensor weight = tf_int8.make({1, 1, 1, 1}, {2});

  Tensor inp_scale = tf_float.make({1}, {0.5});
  Tensor inp_zp = tf_long.make({1}, {0});
  Tensor weight_scale = tf_float.make({1}, {0.5});
  Tensor weight_zp = tf_long.make({1}, {0});
  Tensor out_scale = tf_float.make({1}, {0.5});
  Tensor out_zp = tf_long.make({1}, {0});

  // out [1, 1, 2, 2]
  Tensor out = tf_int8.zeros({1, 1, 2, 2});

  int64_t stride_arr[] = {1, 1};
  int64_t padding_arr[] = {0, 0};
  int64_t dilation_arr[] = {1, 1};
  IntArrayRef stride(stride_arr, 2);
  IntArrayRef padding(padding_arr, 2);
  IntArrayRef dilation(dilation_arr, 2);

  // dequant inp: {1, 2, 3, 4}
  // dequant weight: {1}
  // conv (1x1): {1*1, 2*1, 3*1, 4*1} = {1, 2, 3, 4}
  // requant (scale=0.5, zp=0): {round(1/0.5), ...} = {2, 4, 6, 8}
  cadence::fused_quant::native::convolution_out(
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
      // bias qparams (none)
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

  EXPECT_TENSOR_EQ(out, tf_int8.make({1, 1, 2, 2}, {2, 4, 6, 8}));
}

// 3x3 conv with padding=1, all quantized
TEST_F(FusedQuantConvolutionTest, Conv3x3WithPadding) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  // inp [1, 1, 3, 3] int8
  // After dequant (scale=1, zp=0): same values as int8
  Tensor inp = tf_int8.make({1, 1, 3, 3}, {0, 0, 0, 0, 1, 0, 0, 0, 0});
  // weight [1, 1, 3, 3] int8: identity-like (center=1)
  Tensor weight = tf_int8.make({1, 1, 3, 3}, {0, 0, 0, 0, 1, 0, 0, 0, 0});

  Tensor inp_scale = tf_float.make({1}, {1.0});
  Tensor inp_zp = tf_long.make({1}, {0});
  Tensor weight_scale = tf_float.make({1}, {1.0});
  Tensor weight_zp = tf_long.make({1}, {0});
  Tensor out_scale = tf_float.make({1}, {1.0});
  Tensor out_zp = tf_long.make({1}, {0});

  // out [1, 1, 3, 3] (padding=1 preserves spatial dims)
  Tensor out = tf_int8.zeros({1, 1, 3, 3});

  int64_t stride_arr[] = {1, 1};
  int64_t padding_arr[] = {1, 1};
  int64_t dilation_arr[] = {1, 1};
  IntArrayRef stride(stride_arr, 2);
  IntArrayRef padding(padding_arr, 2);
  IntArrayRef dilation(dilation_arr, 2);

  // dequant inp: {0,0,0, 0,1,0, 0,0,0}
  // dequant weight: {0,0,0, 0,1,0, 0,0,0} (identity kernel)
  // conv with padding=1: output equals input (identity convolution)
  // requant (scale=1, zp=0): same values
  cadence::fused_quant::native::convolution_out(
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
      // bias qparams (none)
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
      out, tf_int8.make({1, 1, 3, 3}, {0, 0, 0, 0, 1, 0, 0, 0, 0}));
}

// float inputs, int8 output
TEST_F(FusedQuantConvolutionTest, FloatInputsQuantizedOutput) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  // inp [1, 1, 2, 2] float
  Tensor inp = tf_float.make({1, 1, 2, 2}, {1.0, 2.0, 3.0, 4.0});
  // weight [1, 1, 1, 1] float
  Tensor weight = tf_float.make({1, 1, 1, 1}, {2.0});

  Tensor out_scale = tf_float.make({1}, {1.0});
  Tensor out_zp = tf_long.make({1}, {0});

  Tensor out = tf_int8.zeros({1, 1, 2, 2});

  int64_t stride_arr[] = {1, 1};
  int64_t padding_arr[] = {0, 0};
  int64_t dilation_arr[] = {1, 1};
  IntArrayRef stride(stride_arr, 2);
  IntArrayRef padding(padding_arr, 2);
  IntArrayRef dilation(dilation_arr, 2);

  // conv (1x1, w=2.0): {2.0, 4.0, 6.0, 8.0}
  // requant (scale=1.0, zp=0): {2, 4, 6, 8}
  cadence::fused_quant::native::convolution_out(
      context_,
      inp,
      weight,
      none_tensor(),
      // inp qparams (none, float)
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      // weight qparams (none, float)
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      // bias qparams (none)
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

  EXPECT_TENSOR_EQ(out, tf_int8.make({1, 1, 2, 2}, {2, 4, 6, 8}));
}

// int8 inputs, float output
TEST_F(FusedQuantConvolutionTest, QuantizedInputsFloatOutput) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  // inp [1, 1, 2, 2] int8
  Tensor inp = tf_int8.make({1, 1, 2, 2}, {2, 4, 6, 8});
  // weight [1, 1, 1, 1] int8
  Tensor weight = tf_int8.make({1, 1, 1, 1}, {4});

  Tensor inp_scale = tf_float.make({1}, {0.5});
  Tensor inp_zp = tf_long.make({1}, {0});
  Tensor weight_scale = tf_float.make({1}, {0.5});
  Tensor weight_zp = tf_long.make({1}, {0});

  Tensor out = tf_float.zeros({1, 1, 2, 2});

  int64_t stride_arr[] = {1, 1};
  int64_t padding_arr[] = {0, 0};
  int64_t dilation_arr[] = {1, 1};
  IntArrayRef stride(stride_arr, 2);
  IntArrayRef padding(padding_arr, 2);
  IntArrayRef dilation(dilation_arr, 2);

  // dequant inp: {1, 2, 3, 4}
  // dequant weight: {2}
  // conv (1x1): {2, 4, 6, 8}
  // output is float, no requant
  cadence::fused_quant::native::convolution_out(
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
      // bias qparams (none)
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      // out qparams (none, float)
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

  EXPECT_TENSOR_EQ(out, tf_float.make({1, 1, 2, 2}, {2.0, 4.0, 6.0, 8.0}));
}

// Convolution with bias
TEST_F(FusedQuantConvolutionTest, WithBias) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  // inp [1, 1, 2, 2] int8
  Tensor inp = tf_int8.make({1, 1, 2, 2}, {2, 4, 6, 8});
  // weight [1, 1, 1, 1] int8
  Tensor weight = tf_int8.make({1, 1, 1, 1}, {2});
  // bias [1] float (not quantized)
  Tensor bias = tf_float.make({1}, {10.0});

  Tensor inp_scale = tf_float.make({1}, {0.5});
  Tensor inp_zp = tf_long.make({1}, {0});
  Tensor weight_scale = tf_float.make({1}, {0.5});
  Tensor weight_zp = tf_long.make({1}, {0});

  Tensor out = tf_float.zeros({1, 1, 2, 2});

  int64_t stride_arr[] = {1, 1};
  int64_t padding_arr[] = {0, 0};
  int64_t dilation_arr[] = {1, 1};
  IntArrayRef stride(stride_arr, 2);
  IntArrayRef padding(padding_arr, 2);
  IntArrayRef dilation(dilation_arr, 2);

  // dequant inp: {1, 2, 3, 4}
  // dequant weight: {1}
  // conv (1x1): {1, 2, 3, 4} + bias 10.0 = {11, 12, 13, 14}
  // output is float, no requant
  cadence::fused_quant::native::convolution_out(
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
      // bias qparams (none, bias is float)
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      // out qparams (none, float)
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

  EXPECT_TENSOR_EQ(out, tf_float.make({1, 1, 2, 2}, {11.0, 12.0, 13.0, 14.0}));
}

// Grouped convolution (depthwise: groups == C_in == C_out)
TEST_F(FusedQuantConvolutionTest, GroupedConvolution) {
  TensorFactory<ScalarType::Float> tf_float;

  // inp [1, 2, 2, 2] float, 2 input channels
  Tensor inp = tf_float.make(
      {1, 2, 2, 2},
      {1.0,
       2.0,
       3.0,
       4.0, // channel 0
       5.0,
       6.0,
       7.0,
       8.0}); // channel 1

  // weight [2, 1, 1, 1] float (groups=2, so C_in/groups=1)
  Tensor weight = tf_float.make({2, 1, 1, 1}, {2.0, 3.0});

  Tensor out = tf_float.zeros({1, 2, 2, 2});

  int64_t stride_arr[] = {1, 1};
  int64_t padding_arr[] = {0, 0};
  int64_t dilation_arr[] = {1, 1};
  IntArrayRef stride(stride_arr, 2);
  IntArrayRef padding(padding_arr, 2);
  IntArrayRef dilation(dilation_arr, 2);

  // groups=2, depthwise
  // channel 0: {1,2,3,4} * 2.0 = {2,4,6,8}
  // channel 1: {5,6,7,8} * 3.0 = {15,18,21,24}
  cadence::fused_quant::native::convolution_out(
      context_,
      inp,
      weight,
      none_tensor(),
      // inp qparams (none, float)
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      // weight qparams (none, float)
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      // bias qparams (none)
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      // out qparams (none, float)
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
      /*groups=*/2,
      out);

  EXPECT_TENSOR_EQ(
      out,
      tf_float.make(
          {1, 2, 2, 2}, {2.0, 4.0, 6.0, 8.0, 15.0, 18.0, 21.0, 24.0}));
}
