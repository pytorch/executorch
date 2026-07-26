/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/cadence/fused_quant/op_linear.h>
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

class FusedQuantLinearTest : public OperatorTest {};

// All quantized, no bias: int8 inp + int8 weight -> int8 out
TEST_F(FusedQuantLinearTest, AllQuantizedNoBias) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  // inp [1,2]: int8 {2,4}, scale=0.5, zp=0 -> float {1.0, 2.0}
  Tensor inp = tf_int8.make({1, 2}, {2, 4});
  Tensor inp_scale = tf_float.make({1}, {0.5});
  Tensor inp_zp = tf_long.make({1}, {0});

  // weight [2,2]: int8 {2,0,0,2}, scale=0.5, zp=0
  //   -> float {{1,0},{0,1}} (identity)
  Tensor weight = tf_int8.make({2, 2}, {2, 0, 0, 2});
  Tensor weight_scale = tf_float.make({1}, {0.5});
  Tensor weight_zp = tf_long.make({1}, {0});

  // out qparams: scale=0.5, zp=0
  Tensor out_scale = tf_float.make({1}, {0.5});
  Tensor out_zp = tf_long.make({1}, {0});

  Tensor out = tf_int8.zeros({1, 2});

  // linear: {1,2} @ identity = {1,2}
  // requant (scale=0.5, zp=0): {round(1/0.5), round(2/0.5)} = {2, 4}
  cadence::fused_quant::native::linear_out(
      context_,
      inp,
      weight,
      none_tensor(), // no bias
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
      // bias qparams (unused, no bias)
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
      out);

  EXPECT_TENSOR_EQ(out, tf_int8.make({1, 2}, {2, 4}));
}

// All quantized with bias: int8 inp + int8 weight + int8 bias -> int8 out
TEST_F(FusedQuantLinearTest, AllQuantizedWithBias) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  // inp [1,2]: int8 {2,4}, scale=0.5, zp=0 -> float {1.0, 2.0}
  Tensor inp = tf_int8.make({1, 2}, {2, 4});
  Tensor inp_scale = tf_float.make({1}, {0.5});
  Tensor inp_zp = tf_long.make({1}, {0});

  // weight [2,2]: int8 {2,0,0,2}, scale=0.5, zp=0
  //   -> float {{1,0},{0,1}} (identity)
  Tensor weight = tf_int8.make({2, 2}, {2, 0, 0, 2});
  Tensor weight_scale = tf_float.make({1}, {0.5});
  Tensor weight_zp = tf_long.make({1}, {0});

  // bias [2]: int8 {2,2}, scale=0.5, zp=0 -> float {1.0, 1.0}
  Tensor bias = tf_int8.make({2}, {2, 2});
  Tensor bias_scale = tf_float.make({1}, {0.5});
  Tensor bias_zp = tf_long.make({1}, {0});

  // out qparams: scale=0.5, zp=0
  Tensor out_scale = tf_float.make({1}, {0.5});
  Tensor out_zp = tf_long.make({1}, {0});

  Tensor out = tf_int8.zeros({1, 2});

  // linear: {1,2} @ identity + {1,1} = {2, 3}
  // requant (scale=0.5, zp=0): {round(2/0.5), round(3/0.5)} = {4, 6}
  cadence::fused_quant::native::linear_out(
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
      // bias qparams
      optional<Tensor>(bias_scale),
      optional<Tensor>(bias_zp),
      ScalarType::Float,
      -128,
      127,
      none_axis(),
      // out qparams
      optional<Tensor>(out_scale),
      optional<Tensor>(out_zp),
      ScalarType::Char,
      -128,
      127,
      none_axis(),
      out);

  EXPECT_TENSOR_EQ(out, tf_int8.make({1, 2}, {4, 6}));
}

// Float inputs -> int8 output
TEST_F(FusedQuantLinearTest, FloatInputsQuantizedOutput) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  // inp [1,2]: float {1.0, 2.0}
  Tensor inp = tf_float.make({1, 2}, {1.0, 2.0});

  // weight [2,2]: float identity {{1,0},{0,1}}
  Tensor weight = tf_float.make({2, 2}, {1.0, 0.0, 0.0, 1.0});

  // out qparams: scale=0.5, zp=0
  Tensor out_scale = tf_float.make({1}, {0.5});
  Tensor out_zp = tf_long.make({1}, {0});

  Tensor out = tf_int8.zeros({1, 2});

  // linear: {1,2} @ identity = {1, 2}
  // requant (scale=0.5, zp=0): {2, 4}
  cadence::fused_quant::native::linear_out(
      context_,
      inp,
      weight,
      none_tensor(), // no bias
      // inp qparams (not quantized)
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      // weight qparams (not quantized)
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      // bias qparams (no bias)
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
      out);

  EXPECT_TENSOR_EQ(out, tf_int8.make({1, 2}, {2, 4}));
}

// int8 inputs -> float output
TEST_F(FusedQuantLinearTest, QuantizedInputsFloatOutput) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  // inp [1,2]: int8 {2,4}, scale=0.5, zp=0 -> float {1.0, 2.0}
  Tensor inp = tf_int8.make({1, 2}, {2, 4});
  Tensor inp_scale = tf_float.make({1}, {0.5});
  Tensor inp_zp = tf_long.make({1}, {0});

  // weight [2,2]: int8 {2,0,0,2}, scale=0.5, zp=0 -> identity
  Tensor weight = tf_int8.make({2, 2}, {2, 0, 0, 2});
  Tensor weight_scale = tf_float.make({1}, {0.5});
  Tensor weight_zp = tf_long.make({1}, {0});

  Tensor out = tf_float.zeros({1, 2});

  // linear: {1,2} @ identity = {1.0, 2.0}
  cadence::fused_quant::native::linear_out(
      context_,
      inp,
      weight,
      none_tensor(), // no bias
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
      // bias qparams (no bias)
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      // out qparams (float, not quantized)
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      out);

  EXPECT_TENSOR_EQ(out, tf_float.make({1, 2}, {1.0, 2.0}));
}

// Per-channel quantized weights (axis=0)
TEST_F(FusedQuantLinearTest, PerChannelWeights) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  // inp [1,2]: float {1.0, 2.0}
  Tensor inp = tf_float.make({1, 2}, {1.0, 2.0});

  // weight [2,2]: int8 {2,4,3,6}, per-channel axis=0
  //   ch0 scale=0.5: {(2-0)*0.5, (4-0)*0.5} = {1.0, 2.0}
  //   ch1 scale=1.0: {(3-0)*1.0, (6-0)*1.0} = {3.0, 6.0}
  Tensor weight = tf_int8.make({2, 2}, {2, 4, 3, 6});
  Tensor weight_scale = tf_float.make({2}, {0.5, 1.0});
  Tensor weight_zp = tf_long.make({2}, {0, 0});

  // out qparams: scale=0.5, zp=0
  Tensor out_scale = tf_float.make({1}, {0.5});
  Tensor out_zp = tf_long.make({1}, {0});

  Tensor out = tf_int8.zeros({1, 2});

  // linear: out[0] = 1*1 + 2*2 = 5, out[1] = 1*3 + 2*6 = 15
  // requant (scale=0.5, zp=0): {round(5/0.5), round(15/0.5)} = {10, 30}
  cadence::fused_quant::native::linear_out(
      context_,
      inp,
      weight,
      none_tensor(), // no bias
      // inp qparams (not quantized)
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      // weight qparams (per-channel, axis=0)
      optional<Tensor>(weight_scale),
      optional<Tensor>(weight_zp),
      ScalarType::Float,
      -128,
      127,
      optional<int64_t>(0),
      // bias qparams (no bias)
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
      out);

  EXPECT_TENSOR_EQ(out, tf_int8.make({1, 2}, {10, 30}));
}

// Batched input: inp [2,2]
TEST_F(FusedQuantLinearTest, BatchedInput) {
  TensorFactory<ScalarType::Float> tf_float;

  // inp [2,2]: float, 2 batch rows
  Tensor inp = tf_float.make({2, 2}, {1.0, 2.0, 3.0, 4.0});

  // weight [2,2]: float identity
  Tensor weight = tf_float.make({2, 2}, {1.0, 0.0, 0.0, 1.0});

  Tensor out = tf_float.zeros({2, 2});

  // linear row0: {1,2} @ identity = {1, 2}
  // linear row1: {3,4} @ identity = {3, 4}
  cadence::fused_quant::native::linear_out(
      context_,
      inp,
      weight,
      none_tensor(), // no bias
      // inp qparams (not quantized)
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      // weight qparams (not quantized)
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      // bias qparams (no bias)
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      // out qparams (not quantized)
      none_tensor(),
      none_tensor(),
      ScalarType::Float,
      0,
      0,
      none_axis(),
      out);

  EXPECT_TENSOR_EQ(out, tf_float.make({2, 2}, {1.0, 2.0, 3.0, 4.0}));
}

// Non-zero zero points
TEST_F(FusedQuantLinearTest, NonZeroZeroPoint) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  // inp [1,2]: int8 {6,8}, scale=0.25, zp=2
  //   dequant: {(6-2)*0.25, (8-2)*0.25} = {1.0, 1.5}
  Tensor inp = tf_int8.make({1, 2}, {6, 8});
  Tensor inp_scale = tf_float.make({1}, {0.25});
  Tensor inp_zp = tf_long.make({1}, {2});

  // weight [2,2]: int8 {6,2,2,6}, scale=0.25, zp=2
  //   dequant: {(6-2)*0.25, (2-2)*0.25, (2-2)*0.25, (6-2)*0.25}
  //          = {1.0, 0.0, 0.0, 1.0} (identity)
  Tensor weight = tf_int8.make({2, 2}, {6, 2, 2, 6});
  Tensor weight_scale = tf_float.make({1}, {0.25});
  Tensor weight_zp = tf_long.make({1}, {2});

  // out: scale=0.5, zp=1
  Tensor out_scale = tf_float.make({1}, {0.5});
  Tensor out_zp = tf_long.make({1}, {1});

  Tensor out = tf_int8.zeros({1, 2});

  // linear: {1.0, 1.5} @ identity = {1.0, 1.5}
  // requant (scale=0.5, zp=1): {round(1.0/0.5)+1, round(1.5/0.5)+1} = {3, 4}
  cadence::fused_quant::native::linear_out(
      context_,
      inp,
      weight,
      none_tensor(), // no bias
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
      // bias qparams (no bias)
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
      out);

  EXPECT_TENSOR_EQ(out, tf_int8.make({1, 2}, {3, 4}));
}
