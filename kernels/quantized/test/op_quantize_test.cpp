/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/quantized/NativeFunctions.h> // Declares the operator
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/test/utils/DeathTest.h>

#include <gtest/gtest.h>
#include <limits>

using namespace ::testing;
using executorch::aten::ArrayRef;
using executorch::aten::Scalar;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using torch::executor::native::quantize_per_channel_out;
using torch::executor::native::quantize_per_tensor_out;
using torch::executor::native::quantize_per_tensor_tensor_args_out;
using torch::executor::testing::TensorFactory;

/// A generic smoke test that works for any dtype that supports ones() and
/// zeros().
template <ScalarType DTYPE>
void test_dtype() {
  TensorFactory<ScalarType::Float> tf;

  Tensor input = tf.full({3, 5}, 4);
  double scale = 0.5;

  int64_t zero_point = 108;
  int64_t quant_min = 0;
  int64_t quant_max = 127;

  TensorFactory<DTYPE> tfo;
  Tensor out = tfo.zeros({3, 5});
  // 4 / 0.5 + 127
  Tensor expected = tfo.full({3, 5}, 116);
  quantize_per_tensor_out(
      input, scale, zero_point, quant_min, quant_max, DTYPE, out);

  EXPECT_TENSOR_EQ(out, expected);
}

template <ScalarType INPUT_DTYPE>
void test_input_dtype() {
  TensorFactory<INPUT_DTYPE> tf_input;

  Tensor input = tf_input.full({3, 5}, 4);
  double scale = 0.5;
  int64_t zero_point = 108;
  int64_t quant_min = 0;
  int64_t quant_max = 127;

  TensorFactory<ScalarType::Char> tfo;
  Tensor out = tfo.zeros({3, 5});
  // 4 / 0.5 + 108 = 116
  Tensor expected = tfo.full({3, 5}, 116);
  quantize_per_tensor_out(
      input, scale, zero_point, quant_min, quant_max, ScalarType::Char, out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpQuantizeOutTest, AllInputDtypesSupported) {
  test_input_dtype<ScalarType::Float>();
  test_input_dtype<ScalarType::Half>();
  test_input_dtype<ScalarType::Double>();
}

TEST(OpQuantizeOutTest, AllDtypesSupported) {
  test_dtype<ScalarType::Byte>();
  test_dtype<ScalarType::Char>();
  test_dtype<ScalarType::Short>();
  test_dtype<ScalarType::Bits16>();
  test_dtype<ScalarType::UInt16>();
  test_dtype<ScalarType::Int>();
}

TEST(OpQuantizeOutTest, DoubleInputTest) {
  TensorFactory<ScalarType::Double> tf_double;

  // Test with a more complex value that might have precision differences
  Tensor input = tf_double.full({2, 3}, 3.14159265359);
  double scale = 0.01;
  int64_t zero_point = -100;
  int64_t quant_min = 0;
  int64_t quant_max = 255;

  TensorFactory<ScalarType::Byte> tfo;
  Tensor out = tfo.zeros({2, 3});
  // 3.14159265359 / 0.01 - 100 = 214.159265359
  Tensor expected = tfo.full({2, 3}, 214);
  quantize_per_tensor_out(
      input, scale, zero_point, quant_min, quant_max, ScalarType::Byte, out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpQuantizeOutTest, HalfInputTest) {
  TensorFactory<ScalarType::Half> tf_half;

  Tensor input = tf_half.full({2, 3}, 2.5);
  double scale = 0.5;
  int64_t zero_point = 10;
  int64_t quant_min = -128;
  int64_t quant_max = 127;

  TensorFactory<ScalarType::Char> tfo;
  Tensor out = tfo.zeros({2, 3});
  // 2.5 / 0.5 + 10 = 15
  Tensor expected = tfo.full({2, 3}, 15);
  quantize_per_tensor_out(
      input, scale, zero_point, quant_min, quant_max, ScalarType::Char, out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpQuantizeOutTest, TensorArgOverload) {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Double> tf_double;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor input = tf_float.full({3, 5}, 4);
  Tensor scale = tf_double.make({1}, {0.5});
  Tensor zero_point = tf_long.make({1}, {127});
  int64_t quant_min = 0;
  int64_t quant_max = 255;

  TensorFactory<ScalarType::Byte> tfo;
  Tensor out = tfo.zeros({3, 5});
  // 4 / 0.5 + 127
  Tensor expected = tfo.full({3, 5}, 135);
  auto context = torch::executor::KernelRuntimeContext();
  quantize_per_tensor_tensor_args_out(
      context,
      input,
      scale,
      zero_point,
      quant_min,
      quant_max,
      ScalarType::Byte,
      out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpQuantizeOutTest, TestOutOfBounds) {
  // Test where 1.0 / epsilon is larger than 8bit integer.

  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Double> tf_double;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor input = tf_float.ones({1, 3, 256, 256});

  Tensor scale = tf_double.make({1}, {0.0011316323652863503});
  Tensor zero_point = tf_long.make({1}, {0});
  int64_t quant_min = -128;
  int64_t quant_max = 127;

  TensorFactory<ScalarType::Char> tfo;
  Tensor out = tfo.zeros({1, 3, 256, 256});

  Tensor expected = tfo.full({1, 3, 256, 256}, 127);

  auto context = torch::executor::KernelRuntimeContext();
  quantize_per_tensor_tensor_args_out(
      context,
      input,
      scale,
      zero_point,
      quant_min,
      quant_max,
      ScalarType::Char,
      out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpQuantizeOutTest, QuantizePerChannel) {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Double> tf_double;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor input = tf_float.full({3, 2}, 4);
  Tensor scale = tf_double.make({2}, {0.5, 1});
  Tensor zero_point = tf_long.make({2}, {127, 63});
  int64_t quant_min = 0;
  int64_t quant_max = 255;

  TensorFactory<ScalarType::Byte> tfo;
  Tensor out = tfo.zeros({3, 2});
  // 4 / 0.5 + 127
  // 4 / 1 + 63
  Tensor expected = tfo.make({3, 2}, {135, 67, 135, 67, 135, 67});
  quantize_per_channel_out(
      input, scale, zero_point, 1, quant_min, quant_max, ScalarType::Byte, out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpQuantizeOutTest, QuantizePerChannelAxis0) {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Double> tf_double;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor input = tf_float.full({3, 2}, 4);
  Tensor scale = tf_double.make({3}, {0.5, 1.0, 2.0});
  Tensor zero_point = tf_long.make({3}, {100, 50, 25});
  int64_t quant_min = 0;
  int64_t quant_max = 255;

  TensorFactory<ScalarType::Byte> tfo;
  Tensor out = tfo.zeros({3, 2});
  // Channel 0: 4 / 0.5 + 100 = 108
  // Channel 1: 4 / 1.0 + 50 = 54
  // Channel 2: 4 / 2.0 + 25 = 27
  Tensor expected = tfo.make({3, 2}, {108, 108, 54, 54, 27, 27});
  quantize_per_channel_out(
      input, scale, zero_point, 0, quant_min, quant_max, ScalarType::Byte, out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpQuantizeOutTest, QuantizePerChannel3D) {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Double> tf_double;
  TensorFactory<ScalarType::Long> tf_long;

  // Test 3D tensor with axis=1 (middle dimension)
  Tensor input = tf_float.full({2, 3, 4}, 6);
  Tensor scale = tf_double.make({3}, {0.5, 1.0, 1.5});
  Tensor zero_point = tf_long.make({3}, {10, 20, 30});
  int64_t quant_min = -128;
  int64_t quant_max = 127;

  TensorFactory<ScalarType::Char> tfo;
  Tensor out = tfo.zeros({2, 3, 4});
  // Channel 0: 6 / 0.5 + 10 = 22
  // Channel 1: 6 / 1.0 + 20 = 26
  // Channel 2: 6 / 1.5 + 30 = 34
  Tensor expected = tfo.make(
      {2, 3, 4},
      {
          22, 22, 22, 22, // First batch, channel 0
          26, 26, 26, 26, // First batch, channel 1
          34, 34, 34, 34, // First batch, channel 2
          22, 22, 22, 22, // Second batch, channel 0
          26, 26, 26, 26, // Second batch, channel 1
          34, 34, 34, 34 // Second batch, channel 2
      });
  quantize_per_channel_out(
      input, scale, zero_point, 1, quant_min, quant_max, ScalarType::Char, out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpQuantizeOutTest, QuantizePerChannel4D) {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Double> tf_double;
  TensorFactory<ScalarType::Long> tf_long;

  // Test 4D tensor with axis=2 (typical conv weight layout: N,C,H,W)
  Tensor input = tf_float.full({2, 2, 3, 2}, 8);
  Tensor scale = tf_double.make({3}, {0.25, 0.5, 1.0});
  Tensor zero_point = tf_long.make({3}, {0, 10, 20});
  int64_t quant_min = -128;
  int64_t quant_max = 127;

  TensorFactory<ScalarType::Char> tfo;
  Tensor out = tfo.zeros({2, 2, 3, 2});
  // Channel 0: 8 / 0.25 + 0 = 32
  // Channel 1: 8 / 0.5 + 10 = 26
  // Channel 2: 8 / 1.0 + 20 = 28
  std::vector<int8_t> expected_data;
  for (int n = 0; n < 2; n++) {
    for (int c = 0; c < 2; c++) {
      for (int h = 0; h < 3; h++) {
        for (int w = 0; w < 2; w++) {
          int8_t val = (h == 0) ? 32 : (h == 1) ? 26 : 28;
          expected_data.push_back(val);
        }
      }
    }
  }
  Tensor expected = tfo.make({2, 2, 3, 2}, expected_data);
  quantize_per_channel_out(
      input, scale, zero_point, 2, quant_min, quant_max, ScalarType::Char, out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpQuantizeOutTest, QuantizePerChannelNegativeAxis) {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Double> tf_double;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor input = tf_float.full({2, 3}, 5);
  Tensor scale = tf_double.make({3}, {0.5, 1.0, 2.0});
  Tensor zero_point = tf_long.make({3}, {0, 10, 20});
  int64_t quant_min = 0;
  int64_t quant_max = 255;

  TensorFactory<ScalarType::Byte> tfo;
  Tensor out = tfo.zeros({2, 3});
  // Using axis=-1 should be equivalent to axis=1 for 2D tensor
  // Channel 0: 5 / 0.5 + 0 = 10
  // Channel 1: 5 / 1.0 + 10 = 15
  // Channel 2: 5 / 2.0 + 20 = 22 (rounded from 22.5)
  Tensor expected = tfo.make({2, 3}, {10, 15, 22, 10, 15, 22});
  quantize_per_channel_out(
      input,
      scale,
      zero_point,
      -1,
      quant_min,
      quant_max,
      ScalarType::Byte,
      out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpQuantizeOutTest, QuantizePerChannelSingleChannel) {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Double> tf_double;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor input = tf_float.full({3, 1, 4}, 7);
  Tensor scale = tf_double.make({1}, {0.5});
  Tensor zero_point = tf_long.make({1}, {128});
  int64_t quant_min = 0;
  int64_t quant_max = 255;

  TensorFactory<ScalarType::Byte> tfo;
  Tensor out = tfo.zeros({3, 1, 4});
  // Single channel: 7 / 0.5 + 128 = 142
  Tensor expected = tfo.full({3, 1, 4}, 142);
  quantize_per_channel_out(
      input, scale, zero_point, 1, quant_min, quant_max, ScalarType::Byte, out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpQuantizeOutTest, QuantizePerChannelDifferentInputTypes) {
  TensorFactory<ScalarType::Double> tf_double_input;
  TensorFactory<ScalarType::Double> tf_double;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor input = tf_double_input.full({2, 2}, 3.14159);
  Tensor scale = tf_double.make({2}, {0.01, 0.02});
  Tensor zero_point = tf_long.make({2}, {0, 100});
  int64_t quant_min = -128;
  int64_t quant_max = 127;

  TensorFactory<ScalarType::Char> tfo;
  Tensor out = tfo.zeros({2, 2});
  // Channel 0: 3.14159 / 0.01 + 0 = 314 -> clamped to 127
  // Channel 1: 3.14159 / 0.02 + 100 = 257 -> clamped to 127
  Tensor expected = tfo.make({2, 2}, {127, 127, 127, 127});
  quantize_per_channel_out(
      input, scale, zero_point, 1, quant_min, quant_max, ScalarType::Char, out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpQuantizeOutTest, QuantizePerChannelDifferentOutputTypes) {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Double> tf_double;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor input = tf_float.full({2, 2}, 10);
  Tensor scale = tf_double.make({2}, {1.0, 2.0});
  Tensor zero_point = tf_long.make({2}, {1000, 2000});
  int64_t quant_min = -32768;
  int64_t quant_max = 32767;

  // Test with 16-bit output
  TensorFactory<ScalarType::Short> tfo;
  Tensor out = tfo.zeros({2, 2});
  // Channel 0: 10 / 1.0 + 1000 = 1010
  // Channel 1: 10 / 2.0 + 2000 = 2005
  Tensor expected = tfo.make({2, 2}, {1010, 2005, 1010, 2005});
  quantize_per_channel_out(
      input,
      scale,
      zero_point,
      1,
      quant_min,
      quant_max,
      ScalarType::Short,
      out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpQuantizeOutTest, QuantizePerChannelMixedValues) {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Double> tf_double;
  TensorFactory<ScalarType::Long> tf_long;

  // Test with different input values per position
  Tensor input = tf_float.make({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  Tensor scale = tf_double.make({3}, {0.5, 1.0, 1.5});
  Tensor zero_point = tf_long.make({3}, {10, 20, 30});
  int64_t quant_min = 0;
  int64_t quant_max = 255;

  TensorFactory<ScalarType::Byte> tfo;
  Tensor out = tfo.zeros({2, 3});
  // Row 0: [1.0/0.5+10, 2.0/1.0+20, 3.0/1.5+30] = [12, 22, 32]
  // Row 1: [4.0/0.5+10, 5.0/1.0+20, 6.0/1.5+30] = [18, 25, 34]
  Tensor expected = tfo.make({2, 3}, {12, 22, 32, 18, 25, 34});
  quantize_per_channel_out(
      input, scale, zero_point, 1, quant_min, quant_max, ScalarType::Byte, out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpQuantizeOutTest, QuantizePerChannelClampingBehavior) {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Double> tf_double;
  TensorFactory<ScalarType::Long> tf_long;

  // Test values that will exceed quant_min/quant_max bounds
  Tensor input = tf_float.make({1, 3}, {-100.0, 0.0, 100.0});
  Tensor scale = tf_double.make({3}, {1.0, 1.0, 1.0});
  Tensor zero_point = tf_long.make({3}, {0, 0, 0});
  int64_t quant_min = -10;
  int64_t quant_max = 10;

  TensorFactory<ScalarType::Char> tfo;
  Tensor out = tfo.zeros({1, 3});
  // Values: [-100, 0, 100] should be clamped to [-10, 0, 10]
  Tensor expected = tfo.make({1, 3}, {-10, 0, 10});
  quantize_per_channel_out(
      input, scale, zero_point, 1, quant_min, quant_max, ScalarType::Char, out);

  EXPECT_TENSOR_EQ(out, expected);
}
