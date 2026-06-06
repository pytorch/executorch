/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/cadence/fused_quant/op_relu.h>
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

class FusedQuantReluTest : public OperatorTest {};

// All quantized: int8 → int8 (per-tensor)
TEST_F(FusedQuantReluTest, AllQuantizedPerTensor) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  const std::vector<int> sizes{4};

  Tensor inp = tf_int8.make(sizes, {-4, -2, 2, 4});

  Tensor inp_scale = tf_float.make({1}, {0.5});
  Tensor inp_zp = tf_long.make({1}, {0});
  Tensor out_scale = tf_float.make({1}, {0.5});
  Tensor out_zp = tf_long.make({1}, {0});

  Tensor out = tf_int8.zeros(sizes);

  // dequant inp: {-2.0, -1.0, 1.0, 2.0}
  // relu: {0.0, 0.0, 1.0, 2.0}
  // requant (scale=0.5, zp=0): {0, 0, 2, 4}
  cadence::fused_quant::native::relu_out(
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

  EXPECT_TENSOR_EQ(out, tf_int8.make(sizes, {0, 0, 2, 4}));
}

// float → int8
TEST_F(FusedQuantReluTest, FloatInputQuantizedOutput) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  const std::vector<int> sizes{4};

  Tensor inp = tf_float.make(sizes, {-2.0, -1.0, 1.0, 2.0});

  Tensor out_scale = tf_float.make({1}, {0.5});
  Tensor out_zp = tf_long.make({1}, {0});

  Tensor out = tf_int8.zeros(sizes);

  // relu: {0.0, 0.0, 1.0, 2.0}
  // requant (scale=0.5, zp=0): {0, 0, 2, 4}
  cadence::fused_quant::native::relu_out(
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

  EXPECT_TENSOR_EQ(out, tf_int8.make(sizes, {0, 0, 2, 4}));
}

// int8 → float
TEST_F(FusedQuantReluTest, QuantizedInputFloatOutput) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  const std::vector<int> sizes{4};

  Tensor inp = tf_int8.make(sizes, {-4, -2, 2, 4});

  Tensor inp_scale = tf_float.make({1}, {0.5});
  Tensor inp_zp = tf_long.make({1}, {0});

  Tensor out = tf_float.zeros(sizes);

  // dequant inp: {-2.0, -1.0, 1.0, 2.0}
  // relu: {0.0, 0.0, 1.0, 2.0}
  cadence::fused_quant::native::relu_out(
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

  EXPECT_TENSOR_EQ(out, tf_float.make(sizes, {0.0, 0.0, 1.0, 2.0}));
}

// Per-channel dequantization on input, per-tensor output
TEST_F(FusedQuantReluTest, PerChannelInput) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  // Shape [2, 2], axis=0 → 2 channels, axis_stride=2
  const std::vector<int> sizes{2, 2};

  Tensor inp = tf_int8.make(sizes, {-4, 2, -3, 6});

  // Per-channel: channel 0 scale=0.5, channel 1 scale=1.0
  Tensor inp_scale = tf_float.make({2}, {0.5, 1.0});
  Tensor inp_zp = tf_long.make({2}, {0, 0});
  Tensor out_scale = tf_float.make({1}, {0.5});
  Tensor out_zp = tf_long.make({1}, {0});

  Tensor out = tf_int8.zeros(sizes);

  // dequant inp channel 0 (scale=0.5): {-2.0, 1.0}
  // dequant inp channel 1 (scale=1.0): {-3.0, 6.0}
  // relu: {0.0, 1.0, 0.0, 6.0}
  // requant (scale=0.5, zp=0): {0, 2, 0, 12}
  cadence::fused_quant::native::relu_out(
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

  EXPECT_TENSOR_EQ(out, tf_int8.make(sizes, {0, 2, 0, 12}));
}

// Per-channel quantization on output
TEST_F(FusedQuantReluTest, PerChannelOutput) {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Long> tf_long;

  // Shape [2, 2], axis=0 → 2 channels
  const std::vector<int> sizes{2, 2};

  Tensor inp = tf_float.make(sizes, {-1.0, 3.0, -2.0, 9.0});

  // Per-channel output: channel 0 scale=0.5, channel 1 scale=1.0
  Tensor out_scale = tf_float.make({2}, {0.5, 1.0});
  Tensor out_zp = tf_long.make({2}, {0, 0});

  Tensor out = tf_int8.zeros(sizes);

  // relu: {0.0, 3.0, 0.0, 9.0}
  // requant channel 0 (scale=0.5): round(0.0/0.5)=0, round(3.0/0.5)=6
  // requant channel 1 (scale=1.0): round(0.0/1.0)=0, round(9.0/1.0)=9
  cadence::fused_quant::native::relu_out(
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

  EXPECT_TENSOR_EQ(out, tf_int8.make(sizes, {0, 6, 0, 9}));
}

// Non-zero zero_point
TEST_F(FusedQuantReluTest, NonZeroZeroPoint) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  const std::vector<int> sizes{4};

  Tensor inp = tf_int8.make(sizes, {-2, 0, 4, 6});

  // scale=0.25, zp=2 → int8 value v maps to (v - 2) * 0.25
  Tensor inp_scale = tf_float.make({1}, {0.25});
  Tensor inp_zp = tf_long.make({1}, {2});
  // out: scale=0.5, zp=1 → float f maps to round(f / 0.5) + 1
  Tensor out_scale = tf_float.make({1}, {0.5});
  Tensor out_zp = tf_long.make({1}, {1});

  Tensor out = tf_int8.zeros(sizes);

  // dequant inp: (-2-2)*0.25=-1.0, (0-2)*0.25=-0.5, (4-2)*0.25=0.5,
  // (6-2)*0.25=1.0 relu: {0.0, 0.0, 0.5, 1.0} requant (scale=0.5, zp=1):
  // round(0.0/0.5)+1=1, round(0.0/0.5)+1=1,
  //   round(0.5/0.5)+1=2, round(1.0/0.5)+1=3
  cadence::fused_quant::native::relu_out(
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

  EXPECT_TENSOR_EQ(out, tf_int8.make(sizes, {1, 1, 2, 3}));
}

// All negative inputs → all zeros after relu
TEST_F(FusedQuantReluTest, AllNegativeInputs) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  const std::vector<int> sizes{4};

  Tensor inp = tf_int8.make(sizes, {-8, -6, -4, -2});

  Tensor inp_scale = tf_float.make({1}, {0.5});
  Tensor inp_zp = tf_long.make({1}, {0});
  Tensor out_scale = tf_float.make({1}, {0.5});
  Tensor out_zp = tf_long.make({1}, {0});

  Tensor out = tf_int8.zeros(sizes);

  // dequant inp: {-4.0, -3.0, -2.0, -1.0}
  // relu: {0.0, 0.0, 0.0, 0.0}
  // requant (scale=0.5, zp=0): {0, 0, 0, 0}
  cadence::fused_quant::native::relu_out(
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

  EXPECT_TENSOR_EQ(out, tf_int8.make(sizes, {0, 0, 0, 0}));
}

// All positive inputs → passes through unchanged
TEST_F(FusedQuantReluTest, AllPositiveInputs) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  const std::vector<int> sizes{4};

  Tensor inp = tf_int8.make(sizes, {2, 4, 6, 8});

  Tensor inp_scale = tf_float.make({1}, {0.5});
  Tensor inp_zp = tf_long.make({1}, {0});
  Tensor out_scale = tf_float.make({1}, {0.5});
  Tensor out_zp = tf_long.make({1}, {0});

  Tensor out = tf_int8.zeros(sizes);

  // dequant inp: {1.0, 2.0, 3.0, 4.0}
  // relu: {1.0, 2.0, 3.0, 4.0} (unchanged, all positive)
  // requant (scale=0.5, zp=0): {2, 4, 6, 8}
  cadence::fused_quant::native::relu_out(
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

  EXPECT_TENSOR_EQ(out, tf_int8.make(sizes, {2, 4, 6, 8}));
}
