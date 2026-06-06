/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/cadence/fused_quant/op_hardswish.h>
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

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

class FusedQuantHardswishTest : public OperatorTest {};

// All quantized: int8 → int8 (per-tensor)
TEST_F(FusedQuantHardswishTest, AllQuantizedPerTensor) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  const std::vector<int> sizes{6};

  Tensor inp = tf_int8.make(sizes, {-6, -3, 0, 3, 6, 10});

  Tensor inp_scale = tf_float.make({1}, {1.0});
  Tensor inp_zp = tf_long.make({1}, {0});
  Tensor out_scale = tf_float.make({1}, {1.0});
  Tensor out_zp = tf_long.make({1}, {0});

  Tensor out = tf_int8.zeros(sizes);

  // dequant inp: {-6, -3, 0, 3, 6, 10}
  // hardswish(-6) = -6 * min(max(-3,0),6)/6 = 0
  // hardswish(-3) = -3 * min(max(0,0),6)/6 = 0
  // hardswish(0)  = 0 * min(max(3,0),6)/6 = 0
  // hardswish(3)  = 3 * min(max(6,0),6)/6 = 3
  // hardswish(6)  = 6 * min(max(9,0),6)/6 = 6
  // hardswish(10) = 10 * min(max(13,0),6)/6 = 10
  // requant (scale=1.0, zp=0): {0, 0, 0, 3, 6, 10}
  cadence::fused_quant::native::hardswish_out(
      context_,
      inp,
      optional<Tensor>(inp_scale),
      optional<Tensor>(inp_zp),
      ScalarType::Float,
      -128,
      127,
      none_axis(),
      optional<Tensor>(out_scale),
      optional<Tensor>(out_zp),
      ScalarType::Char,
      -128,
      127,
      none_axis(),
      out);

  EXPECT_TENSOR_EQ(out, tf_int8.make(sizes, {0, 0, 0, 3, 6, 10}));
}

// float → int8
TEST_F(FusedQuantHardswishTest, FloatInputQuantizedOutput) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  const std::vector<int> sizes{6};

  Tensor inp = tf_float.make(sizes, {-6.0, -3.0, 0.0, 3.0, 6.0, 10.0});

  Tensor out_scale = tf_float.make({1}, {1.0});
  Tensor out_zp = tf_long.make({1}, {0});

  Tensor out = tf_int8.zeros(sizes);

  // hardswish: {0, 0, 0, 3, 6, 10}
  // requant (scale=1.0, zp=0): {0, 0, 0, 3, 6, 10}
  cadence::fused_quant::native::hardswish_out(
      context_,
      inp,
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      optional<Tensor>(out_scale),
      optional<Tensor>(out_zp),
      ScalarType::Char,
      -128,
      127,
      none_axis(),
      out);

  EXPECT_TENSOR_EQ(out, tf_int8.make(sizes, {0, 0, 0, 3, 6, 10}));
}

// int8 → float
TEST_F(FusedQuantHardswishTest, QuantizedInputFloatOutput) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  const std::vector<int> sizes{6};

  Tensor inp = tf_int8.make(sizes, {-6, -3, 0, 3, 6, 10});

  Tensor inp_scale = tf_float.make({1}, {1.0});
  Tensor inp_zp = tf_long.make({1}, {0});

  Tensor out = tf_float.zeros(sizes);

  // dequant inp: {-6, -3, 0, 3, 6, 10}
  // hardswish: {0.0, 0.0, 0.0, 3.0, 6.0, 10.0}
  cadence::fused_quant::native::hardswish_out(
      context_,
      inp,
      optional<Tensor>(inp_scale),
      optional<Tensor>(inp_zp),
      ScalarType::Float,
      -128,
      127,
      none_axis(),
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      out);

  EXPECT_TENSOR_EQ(out, tf_float.make(sizes, {0.0, 0.0, 0.0, 3.0, 6.0, 10.0}));
}

// Per-channel dequantization on input, per-tensor output
TEST_F(FusedQuantHardswishTest, PerChannelInput) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  // Shape [2, 3], axis=0 → 2 channels, axis_stride=3
  const std::vector<int> sizes{2, 3};

  Tensor inp = tf_int8.make(sizes, {-6, -3, 0, 3, 6, 10});

  // Per-channel: channel 0 scale=1.0, channel 1 scale=0.5
  Tensor inp_scale = tf_float.make({2}, {1.0, 0.5});
  Tensor inp_zp = tf_long.make({2}, {0, 0});
  Tensor out_scale = tf_float.make({1}, {0.5});
  Tensor out_zp = tf_long.make({1}, {0});

  Tensor out = tf_int8.zeros(sizes);

  // dequant channel 0 (scale=1.0): {-6, -3, 0}
  // dequant channel 1 (scale=0.5): {1.5, 3.0, 5.0}
  // hardswish(-6) = 0, hardswish(-3) = 0, hardswish(0) = 0
  // hardswish(1.5) = 1.5 * min(max(4.5,0),6)/6 = 1.5*4.5/6 = 1.125
  // hardswish(3.0) = 3 * min(max(6,0),6)/6 = 3*6/6 = 3.0
  // hardswish(5.0) = 5 * min(max(8,0),6)/6 = 5*6/6 = 5.0
  // requant (scale=0.5, zp=0): round(0/0.5)=0, 0, 0,
  //   round(1.125/0.5)=round(2.25)=2, round(3.0/0.5)=6, round(5.0/0.5)=10
  cadence::fused_quant::native::hardswish_out(
      context_,
      inp,
      optional<Tensor>(inp_scale),
      optional<Tensor>(inp_zp),
      ScalarType::Float,
      -128,
      127,
      optional<int64_t>(0),
      optional<Tensor>(out_scale),
      optional<Tensor>(out_zp),
      ScalarType::Char,
      -128,
      127,
      none_axis(),
      out);

  EXPECT_TENSOR_EQ(out, tf_int8.make(sizes, {0, 0, 0, 2, 6, 10}));
}

// Per-channel quantization on output
TEST_F(FusedQuantHardswishTest, PerChannelOutput) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  // Shape [2, 3], axis=0 → 2 channels
  const std::vector<int> sizes{2, 3};

  Tensor inp = tf_float.make(sizes, {-6.0, 0.0, 3.0, 6.0, 10.0, 12.0});

  // Per-channel output: channel 0 scale=1.0, channel 1 scale=0.5
  Tensor out_scale = tf_float.make({2}, {1.0, 0.5});
  Tensor out_zp = tf_long.make({2}, {0, 0});

  Tensor out = tf_int8.zeros(sizes);

  // hardswish(-6) = 0, hardswish(0) = 0, hardswish(3) = 3
  // hardswish(6) = 6, hardswish(10) = 10, hardswish(12) = 12
  // requant channel 0 (scale=1.0): round(0/1)=0, round(0/1)=0, round(3/1)=3
  // requant channel 1 (scale=0.5): round(6/0.5)=12, round(10/0.5)=20,
  // round(12/0.5)=24
  cadence::fused_quant::native::hardswish_out(
      context_,
      inp,
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      optional<Tensor>(out_scale),
      optional<Tensor>(out_zp),
      ScalarType::Char,
      -128,
      127,
      optional<int64_t>(0),
      out);

  EXPECT_TENSOR_EQ(out, tf_int8.make(sizes, {0, 0, 3, 12, 20, 24}));
}

// Non-zero zero points
TEST_F(FusedQuantHardswishTest, NonZeroZeroPoint) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  const std::vector<int> sizes{6};

  Tensor inp = tf_int8.make(sizes, {-4, -1, 2, 5, 8, 12});

  // scale=1.0, zp=2 → dequant: (v-2)*1.0
  Tensor inp_scale = tf_float.make({1}, {1.0});
  Tensor inp_zp = tf_long.make({1}, {2});
  // out scale=1.0, zp=1 → requant: round(f/1.0)+1
  Tensor out_scale = tf_float.make({1}, {1.0});
  Tensor out_zp = tf_long.make({1}, {1});

  Tensor out = tf_int8.zeros(sizes);

  // dequant inp: {-6, -3, 0, 3, 6, 10}
  // hardswish: {0, 0, 0, 3, 6, 10}
  // requant (scale=1.0, zp=1): round(0/1)+1=1, 1, 1,
  //   round(3/1)+1=4, round(6/1)+1=7, round(10/1)+1=11
  cadence::fused_quant::native::hardswish_out(
      context_,
      inp,
      optional<Tensor>(inp_scale),
      optional<Tensor>(inp_zp),
      ScalarType::Float,
      -128,
      127,
      none_axis(),
      optional<Tensor>(out_scale),
      optional<Tensor>(out_zp),
      ScalarType::Char,
      -128,
      127,
      none_axis(),
      out);

  EXPECT_TENSOR_EQ(out, tf_int8.make(sizes, {1, 1, 1, 4, 7, 11}));
}

// All values <= -3 should give 0 (negative saturation region)
TEST_F(FusedQuantHardswishTest, NegativeRegion) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  const std::vector<int> sizes{4};

  Tensor inp = tf_float.make(sizes, {-10.0, -6.0, -4.0, -3.0});

  Tensor out_scale = tf_float.make({1}, {1.0});
  Tensor out_zp = tf_long.make({1}, {0});

  Tensor out = tf_int8.zeros(sizes);

  // hardswish(-10) = -10 * min(max(-7,0),6)/6 = 0
  // hardswish(-6)  = -6 * min(max(-3,0),6)/6 = 0
  // hardswish(-4)  = -4 * min(max(-1,0),6)/6 = 0
  // hardswish(-3)  = -3 * min(max(0,0),6)/6 = 0
  // requant (scale=1.0, zp=0): {0, 0, 0, 0}
  cadence::fused_quant::native::hardswish_out(
      context_,
      inp,
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      optional<Tensor>(out_scale),
      optional<Tensor>(out_zp),
      ScalarType::Char,
      -128,
      127,
      none_axis(),
      out);

  EXPECT_TENSOR_EQ(out, tf_int8.make(sizes, {0, 0, 0, 0}));
}

// All values >= 3 should pass through unchanged (linear region)
TEST_F(FusedQuantHardswishTest, LinearRegion) {
  TensorFactory<ScalarType::Float> tf_float;

  const std::vector<int> sizes{4};

  Tensor inp = tf_float.make(sizes, {3.0, 4.0, 6.0, 10.0});

  Tensor out = tf_float.zeros(sizes);

  // hardswish(3)  = 3 * min(max(6,0),6)/6 = 3
  // hardswish(4)  = 4 * min(max(7,0),6)/6 = 4
  // hardswish(6)  = 6 * min(max(9,0),6)/6 = 6
  // hardswish(10) = 10 * min(max(13,0),6)/6 = 10
  cadence::fused_quant::native::hardswish_out(
      context_,
      inp,
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      out);

  EXPECT_TENSOR_EQ(out, tf_float.make(sizes, {3.0, 4.0, 6.0, 10.0}));
}

// Values between -3 and 3 use the piecewise formula
TEST_F(FusedQuantHardswishTest, TransitionRegion) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  const std::vector<int> sizes{5};

  // int8 input with scale=0.5, zp=0 → float {-3.0, -1.5, 0.0, 1.5, 3.0}
  Tensor inp = tf_int8.make(sizes, {-6, -3, 0, 3, 6});

  Tensor inp_scale = tf_float.make({1}, {0.5});
  Tensor inp_zp = tf_long.make({1}, {0});
  Tensor out_scale = tf_float.make({1}, {0.125});
  Tensor out_zp = tf_long.make({1}, {0});

  Tensor out = tf_int8.zeros(sizes);

  // dequant: {-3.0, -1.5, 0.0, 1.5, 3.0}
  // hardswish(-3.0) = -3*min(max(0,0),6)/6 = 0
  // hardswish(-1.5) = -1.5*min(max(1.5,0),6)/6 = -1.5*1.5/6 = -0.375
  // hardswish(0)    = 0*min(max(3,0),6)/6 = 0
  // hardswish(1.5)  = 1.5*min(max(4.5,0),6)/6 = 1.5*4.5/6 = 1.125
  // hardswish(3.0)  = 3*min(max(6,0),6)/6 = 3*6/6 = 3.0
  // requant (scale=0.125, zp=0): round(0/0.125)=0, round(-0.375/0.125)=-3,
  //   round(0/0.125)=0, round(1.125/0.125)=9, round(3.0/0.125)=24
  cadence::fused_quant::native::hardswish_out(
      context_,
      inp,
      optional<Tensor>(inp_scale),
      optional<Tensor>(inp_zp),
      ScalarType::Float,
      -128,
      127,
      none_axis(),
      optional<Tensor>(out_scale),
      optional<Tensor>(out_zp),
      ScalarType::Char,
      -128,
      127,
      none_axis(),
      out);

  EXPECT_TENSOR_EQ(out, tf_int8.make(sizes, {0, -3, 0, 9, 24}));
}
