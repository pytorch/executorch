/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/cadence/fused_quant/op_mul.h>
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

class FusedQuantMulTest : public OperatorTest {};

// All quantized: int8 * int8 -> int8 (per-tensor)
TEST_F(FusedQuantMulTest, AllQuantizedPerTensor) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  const std::vector<int> sizes{4};

  Tensor inp = tf_int8.make(sizes, {2, 4, 6, 8});
  Tensor other = tf_int8.make(sizes, {2, 2, 2, 2});

  Tensor inp_scale = tf_float.make({1}, {0.5});
  Tensor inp_zp = tf_long.make({1}, {0});
  Tensor other_scale = tf_float.make({1}, {0.5});
  Tensor other_zp = tf_long.make({1}, {0});
  Tensor out_scale = tf_float.make({1}, {0.5});
  Tensor out_zp = tf_long.make({1}, {0});

  Tensor out = tf_int8.zeros(sizes);

  // dequant inp: {1.0, 2.0, 3.0, 4.0}
  // dequant other: {1.0, 1.0, 1.0, 1.0}
  // float mul: {1.0, 2.0, 3.0, 4.0}
  // requant (scale=0.5, zp=0): {2, 4, 6, 8}
  cadence::fused_quant::native::mul_out(
      context_,
      inp,
      other,
      optional<Tensor>(inp_scale),
      optional<Tensor>(inp_zp),
      ScalarType::Float,
      -128,
      127,
      none_axis(),
      optional<Tensor>(other_scale),
      optional<Tensor>(other_zp),
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

  EXPECT_TENSOR_EQ(out, tf_int8.make(sizes, {2, 4, 6, 8}));
}

// float * float -> int8
TEST_F(FusedQuantMulTest, FloatInputsQuantizedOutput) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  const std::vector<int> sizes{4};

  Tensor inp = tf_float.make(sizes, {1.0, 2.0, 3.0, 4.0});
  Tensor other = tf_float.make(sizes, {2.0, 2.0, 2.0, 2.0});

  Tensor out_scale = tf_float.make({1}, {0.5});
  Tensor out_zp = tf_long.make({1}, {0});

  Tensor out = tf_int8.zeros(sizes);

  // float mul: {2.0, 4.0, 6.0, 8.0}
  // requant (scale=0.5, zp=0): {4, 8, 12, 16}
  cadence::fused_quant::native::mul_out(
      context_,
      inp,
      other,
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
      optional<Tensor>(out_scale),
      optional<Tensor>(out_zp),
      ScalarType::Char,
      -128,
      127,
      none_axis(),
      out);

  EXPECT_TENSOR_EQ(out, tf_int8.make(sizes, {4, 8, 12, 16}));
}

// int8 * float -> int8
TEST_F(FusedQuantMulTest, QuantizedInpFloatOther) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  const std::vector<int> sizes{4};

  Tensor inp = tf_int8.make(sizes, {2, 4, 6, 8});
  Tensor other = tf_float.make(sizes, {2.0, 2.0, 2.0, 2.0});

  Tensor inp_scale = tf_float.make({1}, {0.5});
  Tensor inp_zp = tf_long.make({1}, {0});
  Tensor out_scale = tf_float.make({1}, {0.5});
  Tensor out_zp = tf_long.make({1}, {0});

  Tensor out = tf_int8.zeros(sizes);

  // dequant inp: {1.0, 2.0, 3.0, 4.0}
  // float mul: {2.0, 4.0, 6.0, 8.0}
  // requant: {4, 8, 12, 16}
  cadence::fused_quant::native::mul_out(
      context_,
      inp,
      other,
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
      optional<Tensor>(out_scale),
      optional<Tensor>(out_zp),
      ScalarType::Char,
      -128,
      127,
      none_axis(),
      out);

  EXPECT_TENSOR_EQ(out, tf_int8.make(sizes, {4, 8, 12, 16}));
}

// float * int8 -> int8
TEST_F(FusedQuantMulTest, FloatInpQuantizedOther) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  const std::vector<int> sizes{4};

  Tensor inp = tf_float.make(sizes, {1.0, 2.0, 3.0, 4.0});
  Tensor other = tf_int8.make(sizes, {2, 2, 2, 2});

  Tensor other_scale = tf_float.make({1}, {0.5});
  Tensor other_zp = tf_long.make({1}, {0});
  Tensor out_scale = tf_float.make({1}, {0.5});
  Tensor out_zp = tf_long.make({1}, {0});

  Tensor out = tf_int8.zeros(sizes);

  // dequant other: {1.0, 1.0, 1.0, 1.0}
  // float mul: {1.0, 2.0, 3.0, 4.0}
  // requant: {2, 4, 6, 8}
  cadence::fused_quant::native::mul_out(
      context_,
      inp,
      other,
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      optional<Tensor>(other_scale),
      optional<Tensor>(other_zp),
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

  EXPECT_TENSOR_EQ(out, tf_int8.make(sizes, {2, 4, 6, 8}));
}

// int8 * int8 -> float
TEST_F(FusedQuantMulTest, QuantizedInputsFloatOutput) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  const std::vector<int> sizes{4};

  Tensor inp = tf_int8.make(sizes, {2, 4, 6, 8});
  Tensor other = tf_int8.make(sizes, {2, 2, 2, 2});

  Tensor inp_scale = tf_float.make({1}, {0.5});
  Tensor inp_zp = tf_long.make({1}, {0});
  Tensor other_scale = tf_float.make({1}, {0.5});
  Tensor other_zp = tf_long.make({1}, {0});

  Tensor out = tf_float.zeros(sizes);

  // dequant inp: {1.0, 2.0, 3.0, 4.0}
  // dequant other: {1.0, 1.0, 1.0, 1.0}
  // float mul: {1.0, 2.0, 3.0, 4.0}
  cadence::fused_quant::native::mul_out(
      context_,
      inp,
      other,
      optional<Tensor>(inp_scale),
      optional<Tensor>(inp_zp),
      ScalarType::Float,
      -128,
      127,
      none_axis(),
      optional<Tensor>(other_scale),
      optional<Tensor>(other_zp),
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

  EXPECT_TENSOR_EQ(out, tf_float.make(sizes, {1.0, 2.0, 3.0, 4.0}));
}

// int8 * float -> float
TEST_F(FusedQuantMulTest, QuantizedInpFloatOtherFloatOutput) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  const std::vector<int> sizes{4};

  Tensor inp = tf_int8.make(sizes, {2, 4, 6, 8});
  Tensor other = tf_float.make(sizes, {2.0, 2.0, 2.0, 2.0});

  Tensor inp_scale = tf_float.make({1}, {0.5});
  Tensor inp_zp = tf_long.make({1}, {0});

  Tensor out = tf_float.zeros(sizes);

  // dequant inp: {1.0, 2.0, 3.0, 4.0}
  // float mul: {2.0, 4.0, 6.0, 8.0}
  cadence::fused_quant::native::mul_out(
      context_,
      inp,
      other,
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
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      out);

  EXPECT_TENSOR_EQ(out, tf_float.make(sizes, {2.0, 4.0, 6.0, 8.0}));
}

// float * int8 -> float
TEST_F(FusedQuantMulTest, FloatInpQuantizedOtherFloatOutput) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  const std::vector<int> sizes{4};

  Tensor inp = tf_float.make(sizes, {1.0, 2.0, 3.0, 4.0});
  Tensor other = tf_int8.make(sizes, {2, 2, 2, 2});

  Tensor other_scale = tf_float.make({1}, {0.5});
  Tensor other_zp = tf_long.make({1}, {0});

  Tensor out = tf_float.zeros(sizes);

  // dequant other: {1.0, 1.0, 1.0, 1.0}
  // float mul: {1.0, 2.0, 3.0, 4.0}
  cadence::fused_quant::native::mul_out(
      context_,
      inp,
      other,
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      optional<Tensor>(other_scale),
      optional<Tensor>(other_zp),
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

  EXPECT_TENSOR_EQ(out, tf_float.make(sizes, {1.0, 2.0, 3.0, 4.0}));
}

// Per-channel dequantization on input, per-tensor output
TEST_F(FusedQuantMulTest, PerChannelInput) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  // Shape [2, 2], axis=0 -> 2 channels, axis_stride=2
  const std::vector<int> sizes{2, 2};

  Tensor inp = tf_int8.make(sizes, {2, 4, 6, 8});
  Tensor other = tf_float.make(sizes, {2.0, 2.0, 2.0, 2.0});

  // Per-channel: channel 0 scale=0.5, channel 1 scale=1.0
  Tensor inp_scale = tf_float.make({2}, {0.5, 1.0});
  Tensor inp_zp = tf_long.make({2}, {0, 0});
  Tensor out_scale = tf_float.make({1}, {0.5});
  Tensor out_zp = tf_long.make({1}, {0});

  Tensor out = tf_int8.zeros(sizes);

  // dequant inp channel 0 (scale=0.5): {1.0, 2.0}
  // dequant inp channel 1 (scale=1.0): {6.0, 8.0}
  // float mul: {1.0*2.0, 2.0*2.0, 6.0*2.0, 8.0*2.0} = {2.0, 4.0, 12.0, 16.0}
  // requant (scale=0.5): {4, 8, 24, 32}
  cadence::fused_quant::native::mul_out(
      context_,
      inp,
      other,
      optional<Tensor>(inp_scale),
      optional<Tensor>(inp_zp),
      ScalarType::Float,
      -128,
      127,
      optional<int64_t>(0),
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

  EXPECT_TENSOR_EQ(out, tf_int8.make(sizes, {4, 8, 24, 32}));
}

// Per-channel quantization on output
TEST_F(FusedQuantMulTest, PerChannelOutput) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  // Shape [2, 2], axis=0 -> 2 channels
  const std::vector<int> sizes{2, 2};

  Tensor inp = tf_float.make(sizes, {2.0, 3.0, 7.0, 9.0});
  Tensor other = tf_float.make(sizes, {1.0, 1.0, 1.0, 1.0});

  // Per-channel output: channel 0 scale=0.5, channel 1 scale=1.0
  Tensor out_scale = tf_float.make({2}, {0.5, 1.0});
  Tensor out_zp = tf_long.make({2}, {0, 0});

  Tensor out = tf_int8.zeros(sizes);

  // float mul: {2.0, 3.0, 7.0, 9.0}
  // requant channel 0 (scale=0.5): round(2.0/0.5)=4, round(3.0/0.5)=6
  // requant channel 1 (scale=1.0): round(7.0/1.0)=7, round(9.0/1.0)=9
  cadence::fused_quant::native::mul_out(
      context_,
      inp,
      other,
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
      optional<Tensor>(out_scale),
      optional<Tensor>(out_zp),
      ScalarType::Char,
      -128,
      127,
      optional<int64_t>(0),
      out);

  EXPECT_TENSOR_EQ(out, tf_int8.make(sizes, {4, 6, 7, 9}));
}

// Non-zero zero_point
TEST_F(FusedQuantMulTest, NonZeroZeroPoint) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  const std::vector<int> sizes{4};

  Tensor inp = tf_int8.make(sizes, {6, 8, 10, 12});
  Tensor other = tf_int8.make(sizes, {4, 4, 4, 4});

  // scale=0.25, zp=2 -> int8 value v maps to (v - 2) * 0.25
  Tensor inp_scale = tf_float.make({1}, {0.25});
  Tensor inp_zp = tf_long.make({1}, {2});
  Tensor other_scale = tf_float.make({1}, {0.25});
  Tensor other_zp = tf_long.make({1}, {2});
  // out: scale=0.5, zp=1 -> float f maps to round(f / 0.5) + 1
  Tensor out_scale = tf_float.make({1}, {0.5});
  Tensor out_zp = tf_long.make({1}, {1});

  Tensor out = tf_int8.zeros(sizes);

  // dequant inp: (6-2)*0.25=1.0, (8-2)*0.25=1.5, (10-2)*0.25=2.0,
  // (12-2)*0.25=2.5 dequant other: (4-2)*0.25=0.5 each float mul: {0.5,
  // 0.75, 1.0, 1.25} requant (scale=0.5, zp=1): round(0.5/0.5)+1=2,
  // round(0.75/0.5)+1=3,
  //   round(1.0/0.5)+1=3, round(1.25/0.5)+1=4
  cadence::fused_quant::native::mul_out(
      context_,
      inp,
      other,
      optional<Tensor>(inp_scale),
      optional<Tensor>(inp_zp),
      ScalarType::Float,
      -128,
      127,
      none_axis(),
      optional<Tensor>(other_scale),
      optional<Tensor>(other_zp),
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

  EXPECT_TENSOR_EQ(out, tf_int8.make(sizes, {2, 3, 3, 4}));
}
