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

TEST(OpQuantizeOutTest, LargePerChannelClampingSIMDPath) {
  // Test quant_min/quant_max clamping with large tensor to exercise SIMD path
  // Shape: [3, 80] with axis=0 (3 channels, 80 elements each)
  // 80 elements = 10 SIMD iterations (8 elements each)
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Double> tf_double;
  TensorFactory<ScalarType::Long> tf_long;

  const int num_channels = 3;
  const int block_size = 80;
  std::vector<float> input_data(num_channels * block_size);

  // Create input data with values that exceed quant_min/quant_max
  for (int ch = 0; ch < num_channels; ch++) {
    for (int i = 0; i < block_size; i++) {
      // Generate values from -150 to 150 to test clamping
      input_data[ch * block_size + i] =
          static_cast<float>((i % 40) - 20) * 5.0f * (ch + 1);
    }
  }
  Tensor input = tf_float.make({num_channels, block_size}, input_data);

  // Use uniform scale and zero_point for all channels
  Tensor scale = tf_double.make({num_channels}, {1.0, 1.0, 1.0});
  Tensor zero_point = tf_long.make({num_channels}, {0, 0, 0});

  // Set narrow quant_min/quant_max to force clamping
  int64_t quant_min = -20;
  int64_t quant_max = 20;

  TensorFactory<ScalarType::Char> tfo;
  Tensor out = tfo.zeros({num_channels, block_size});

  // Compute expected values with clamping
  std::vector<int8_t> expected_data(num_channels * block_size);
  for (int ch = 0; ch < num_channels; ch++) {
    double ch_scale = scale.const_data_ptr<double>()[ch];
    int64_t ch_zero_point = zero_point.const_data_ptr<int64_t>()[ch];

    for (int i = 0; i < block_size; i++) {
      int idx = ch * block_size + i;
      // Use double precision to avoid overflow
      double val = static_cast<double>(input_data[idx]) / ch_scale;
      // Clamp before converting to int to avoid overflow
      val = std::max(-1000.0, std::min(1000.0, val));
      int32_t qval = static_cast<int32_t>(std::nearbyint(val)) +
          static_cast<int32_t>(ch_zero_point);
      // Apply quant_min/quant_max clamping
      qval = std::max(
          static_cast<int32_t>(quant_min),
          std::min(static_cast<int32_t>(quant_max), qval));
      expected_data[idx] = static_cast<int8_t>(qval);
    }
  }
  Tensor expected = tfo.make({num_channels, block_size}, expected_data);

  quantize_per_channel_out(
      input, scale, zero_point, 0, quant_min, quant_max, ScalarType::Char, out);

  EXPECT_TENSOR_EQ(out, expected);
}

// Large tensor tests to ensure ARM NEON SIMD path is exercised

TEST(OpQuantizeOutTest, LargeTensorUInt8SIMDPath) {
  // Test with 64 elements to fully exercise SIMD path (8 elements per
  // iteration)
  TensorFactory<ScalarType::Float> tf_float;

  // Create input with known values for verification
  std::vector<float> input_data(64);
  for (size_t i = 0; i < 64; i++) {
    input_data[i] = static_cast<float>(i) * 0.5f; // 0.0, 0.5, 1.0, 1.5, ...
  }
  Tensor input = tf_float.make({64}, input_data);

  double scale = 0.1;
  int64_t zero_point = 10;
  int64_t quant_min = 0;
  int64_t quant_max = 255;

  TensorFactory<ScalarType::Byte> tfo;
  Tensor out = tfo.zeros({64});

  // Compute expected values: round(value / scale) + zero_point
  std::vector<uint8_t> expected_data(64);
  for (size_t i = 0; i < 64; i++) {
    float val = input_data[i] / static_cast<float>(scale);
    int32_t qval = static_cast<int32_t>(std::nearbyint(val)) + zero_point;
    qval = std::min(255, std::max(0, qval));
    expected_data[i] = static_cast<uint8_t>(qval);
  }
  Tensor expected = tfo.make({64}, expected_data);

  quantize_per_tensor_out(
      input, scale, zero_point, quant_min, quant_max, ScalarType::Byte, out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpQuantizeOutTest, LargeTensorInt8SIMDPath) {
  // Test with 72 elements (9 SIMD iterations of 8) to test both vectorized and
  // scalar paths
  TensorFactory<ScalarType::Float> tf_float;

  std::vector<float> input_data(72);
  for (size_t i = 0; i < 72; i++) {
    // Mix of positive and negative values
    input_data[i] = static_cast<float>(static_cast<int>(i) - 36) * 0.25f;
  }
  Tensor input = tf_float.make({72}, input_data);

  double scale = 0.2;
  int64_t zero_point = 0;
  int64_t quant_min = -128;
  int64_t quant_max = 127;

  TensorFactory<ScalarType::Char> tfo;
  Tensor out = tfo.zeros({72});

  // Compute expected values
  std::vector<int8_t> expected_data(72);
  for (size_t i = 0; i < 72; i++) {
    float val = input_data[i] / static_cast<float>(scale);
    int32_t qval = static_cast<int32_t>(std::nearbyint(val)) + zero_point;
    qval = std::min(127, std::max(-128, qval));
    expected_data[i] = static_cast<int8_t>(qval);
  }
  Tensor expected = tfo.make({72}, expected_data);

  quantize_per_tensor_out(
      input, scale, zero_point, quant_min, quant_max, ScalarType::Char, out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpQuantizeOutTest, LargeTensorWithRemainderUInt8) {
  // Test with 100 elements (12 SIMD iterations + 4 remainder) to test remainder
  // handling
  TensorFactory<ScalarType::Float> tf_float;

  std::vector<float> input_data(100);
  for (size_t i = 0; i < 100; i++) {
    input_data[i] = static_cast<float>(i % 50) * 0.3f;
  }
  Tensor input = tf_float.make({100}, input_data);

  double scale = 0.15;
  int64_t zero_point = 128;
  int64_t quant_min = 0;
  int64_t quant_max = 255;

  TensorFactory<ScalarType::Byte> tfo;
  Tensor out = tfo.zeros({100});

  std::vector<uint8_t> expected_data(100);
  for (size_t i = 0; i < 100; i++) {
    float val = input_data[i] / static_cast<float>(scale);
    int32_t qval = static_cast<int32_t>(std::nearbyint(val)) + zero_point;
    qval = std::min(255, std::max(0, qval));
    expected_data[i] = static_cast<uint8_t>(qval);
  }
  Tensor expected = tfo.make({100}, expected_data);

  quantize_per_tensor_out(
      input, scale, zero_point, quant_min, quant_max, ScalarType::Byte, out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpQuantizeOutTest, LargeTensorWithRemainderInt8) {
  // Test with 99 elements (12 SIMD iterations + 3 remainder)
  TensorFactory<ScalarType::Float> tf_float;

  std::vector<float> input_data(99);
  for (size_t i = 0; i < 99; i++) {
    input_data[i] = std::sin(static_cast<float>(i) * 0.1f) * 10.0f;
  }
  Tensor input = tf_float.make({99}, input_data);

  double scale = 0.1;
  int64_t zero_point = 5;
  int64_t quant_min = -128;
  int64_t quant_max = 127;

  TensorFactory<ScalarType::Char> tfo;
  Tensor out = tfo.zeros({99});

  std::vector<int8_t> expected_data(99);
  for (size_t i = 0; i < 99; i++) {
    float val = input_data[i] / static_cast<float>(scale);
    int32_t qval = static_cast<int32_t>(std::nearbyint(val)) + zero_point;
    qval = std::min(127, std::max(-128, qval));
    expected_data[i] = static_cast<int8_t>(qval);
  }
  Tensor expected = tfo.make({99}, expected_data);

  quantize_per_tensor_out(
      input, scale, zero_point, quant_min, quant_max, ScalarType::Char, out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpQuantizeOutTest, VeryLargeTensor2DUInt8) {
  // Test with realistic 2D tensor size that would be used in neural networks
  // 256x256 = 65536 elements (8192 SIMD iterations)
  TensorFactory<ScalarType::Float> tf_float;

  std::vector<float> input_data(256 * 256);
  for (size_t i = 0; i < 256 * 256; i++) {
    // Generate diverse values in a safe range
    input_data[i] =
        static_cast<float>((static_cast<int>(i % 256) - 128)) * 0.05f;
  }
  Tensor input = tf_float.make({256, 256}, input_data);

  double scale = 0.05;
  int64_t zero_point = 128;
  int64_t quant_min = 0;
  int64_t quant_max = 255;

  TensorFactory<ScalarType::Byte> tfo;
  Tensor out = tfo.zeros({256, 256});

  // Compute expected values with proper overflow handling
  std::vector<uint8_t> expected_data(256 * 256);
  for (size_t i = 0; i < 256 * 256; i++) {
    // Use double precision to avoid overflow
    double val = static_cast<double>(input_data[i]) / scale;
    // Clamp before converting to int to avoid overflow
    val = std::max(-1000.0, std::min(1000.0, val));
    int32_t qval = static_cast<int32_t>(std::nearbyint(val)) +
        static_cast<int32_t>(zero_point);
    qval = std::min(255, std::max(0, qval));
    expected_data[i] = static_cast<uint8_t>(qval);
  }
  Tensor expected = tfo.make({256, 256}, expected_data);

  quantize_per_tensor_out(
      input, scale, zero_point, quant_min, quant_max, ScalarType::Byte, out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpQuantizeOutTest, VeryLargeTensor3DInt8) {
  // Test with 3D tensor (batch_size=2, height=64, width=128) = 16384 elements
  TensorFactory<ScalarType::Float> tf_float;

  const size_t total_elements = 2 * 64 * 128;
  std::vector<float> input_data(total_elements);
  for (size_t i = 0; i < total_elements; i++) {
    input_data[i] = std::cos(static_cast<float>(i) * 0.01f) * 8.0f;
  }
  Tensor input = tf_float.make({2, 64, 128}, input_data);

  double scale = 0.0625; // 1/16
  int64_t zero_point = -10;
  int64_t quant_min = -128;
  int64_t quant_max = 127;

  TensorFactory<ScalarType::Char> tfo;
  Tensor out = tfo.zeros({2, 64, 128});

  std::vector<int8_t> expected_data(total_elements);
  for (size_t i = 0; i < total_elements; i++) {
    float val = input_data[i] / static_cast<float>(scale);
    int32_t qval = static_cast<int32_t>(std::nearbyint(val)) + zero_point;
    qval = std::min(127, std::max(-128, qval));
    expected_data[i] = static_cast<int8_t>(qval);
  }
  Tensor expected = tfo.make({2, 64, 128}, expected_data);

  quantize_per_tensor_out(
      input, scale, zero_point, quant_min, quant_max, ScalarType::Char, out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpQuantizeOutTest, EdgeCaseSizesSIMD) {
  // Test specific sizes around SIMD boundaries
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Byte> tfo;

  double scale = 0.1;
  int64_t zero_point = 100;
  int64_t quant_min = 0;
  int64_t quant_max = 255;

  // Test sizes: 7 (just before SIMD), 8 (exactly 1 SIMD), 9 (1 SIMD + 1), 15,
  // 16, 17
  std::vector<size_t> test_sizes = {
      7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33};

  for (size_t size : test_sizes) {
    std::vector<float> input_data(size);
    std::vector<uint8_t> expected_data(size);

    for (size_t i = 0; i < size; i++) {
      input_data[i] = static_cast<float>(i) * 0.3f;
      float val = input_data[i] / static_cast<float>(scale);
      int32_t qval = static_cast<int32_t>(std::nearbyint(val)) + zero_point;
      qval = std::min(255, std::max(0, qval));
      expected_data[i] = static_cast<uint8_t>(qval);
    }

    Tensor input = tf_float.make({static_cast<int>(size)}, input_data);
    Tensor out = tfo.zeros({static_cast<int>(size)});
    Tensor expected = tfo.make({static_cast<int>(size)}, expected_data);

    quantize_per_tensor_out(
        input, scale, zero_point, quant_min, quant_max, ScalarType::Byte, out);

    EXPECT_TENSOR_EQ(out, expected);
  }
}

// Large tensor tests for per-channel quantization to ensure SIMD path is
// exercised

TEST(OpQuantizeOutTest, LargePerChannelUInt8SIMDPath) {
  // Test per-channel quantization with large blocks (64 elements per channel)
  // Shape: [4, 64] with axis=1 (4 channels, 64 elements each)
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Double> tf_double;
  TensorFactory<ScalarType::Long> tf_long;

  const int num_channels = 4;
  const int block_size = 64;
  std::vector<float> input_data(num_channels * block_size);

  // Create varying input data for each channel
  for (int ch = 0; ch < num_channels; ch++) {
    for (int i = 0; i < block_size; i++) {
      input_data[ch * block_size + i] = static_cast<float>((ch + 1) * i) * 0.1f;
    }
  }
  Tensor input = tf_float.make({num_channels, block_size}, input_data);

  // Different scale and zero_point for each channel
  Tensor scale = tf_double.make({num_channels}, {0.1, 0.2, 0.15, 0.25});
  Tensor zero_point = tf_long.make({num_channels}, {10, 20, 15, 25});

  int64_t quant_min = 0;
  int64_t quant_max = 255;

  TensorFactory<ScalarType::Byte> tfo;
  Tensor out = tfo.zeros({num_channels, block_size});

  // Compute expected values
  std::vector<uint8_t> expected_data(num_channels * block_size);
  for (int ch = 0; ch < num_channels; ch++) {
    double ch_scale = scale.const_data_ptr<double>()[ch];
    int64_t ch_zero_point = zero_point.const_data_ptr<int64_t>()[ch];

    for (int i = 0; i < block_size; i++) {
      int idx = ch * block_size + i;
      float val = input_data[idx] / static_cast<float>(ch_scale);
      int32_t qval = static_cast<int32_t>(std::nearbyint(val)) +
          static_cast<int32_t>(ch_zero_point);
      qval = std::min(255, std::max(0, qval));
      expected_data[idx] = static_cast<uint8_t>(qval);
    }
  }
  Tensor expected = tfo.make({num_channels, block_size}, expected_data);

  quantize_per_channel_out(
      input, scale, zero_point, 0, quant_min, quant_max, ScalarType::Byte, out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpQuantizeOutTest, LargePerChannelInt8SIMDPath) {
  // Test per-channel quantization with int8 and large blocks
  // Shape: [3, 100] with axis=1 (3 channels, 100 elements each)
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Double> tf_double;
  TensorFactory<ScalarType::Long> tf_long;

  const int num_channels = 3;
  const int block_size = 100; // 12 SIMD iterations + 4 remainder
  std::vector<float> input_data(num_channels * block_size);

  // Create varying input data with negative values
  for (int ch = 0; ch < num_channels; ch++) {
    for (int i = 0; i < block_size; i++) {
      input_data[ch * block_size + i] =
          static_cast<float>(i - 50) * 0.2f * (ch + 1);
    }
  }
  Tensor input = tf_float.make({num_channels, block_size}, input_data);

  Tensor scale = tf_double.make({num_channels}, {0.1, 0.15, 0.2});
  Tensor zero_point = tf_long.make({num_channels}, {0, -5, 5});

  int64_t quant_min = -128;
  int64_t quant_max = 127;

  TensorFactory<ScalarType::Char> tfo;
  Tensor out = tfo.zeros({num_channels, block_size});

  // Compute expected values
  std::vector<int8_t> expected_data(num_channels * block_size);
  for (int ch = 0; ch < num_channels; ch++) {
    double ch_scale = scale.const_data_ptr<double>()[ch];
    int64_t ch_zero_point = zero_point.const_data_ptr<int64_t>()[ch];

    for (int i = 0; i < block_size; i++) {
      int idx = ch * block_size + i;
      float val = input_data[idx] / static_cast<float>(ch_scale);
      int32_t qval = static_cast<int32_t>(std::nearbyint(val)) +
          static_cast<int32_t>(ch_zero_point);
      qval = std::min(127, std::max(-128, qval));
      expected_data[idx] = static_cast<int8_t>(qval);
    }
  }
  Tensor expected = tfo.make({num_channels, block_size}, expected_data);

  quantize_per_channel_out(
      input, scale, zero_point, 0, quant_min, quant_max, ScalarType::Char, out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpQuantizeOutTest, VeryLargePerChannel2DUInt8) {
  // Test realistic neural network weight tensor
  // Shape: [128, 256] with axis=0 (128 channels, 256 elements each)
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Double> tf_double;
  TensorFactory<ScalarType::Long> tf_long;

  const int num_channels = 128;
  const int block_size = 256;
  const int total_elements = num_channels * block_size;

  std::vector<float> input_data(total_elements);
  for (int i = 0; i < total_elements; i++) {
    input_data[i] = std::sin(static_cast<float>(i) * 0.01f) * 5.0f;
  }
  Tensor input = tf_float.make({num_channels, block_size}, input_data);

  // Create varying scales and zero_points for each channel
  std::vector<double> scales(num_channels);
  std::vector<int64_t> zero_points(num_channels);
  for (int ch = 0; ch < num_channels; ch++) {
    scales[ch] = 0.02 + (ch % 10) * 0.001; // Varying scales
    zero_points[ch] = 128 + (ch % 5); // Varying zero_points
  }
  Tensor scale = tf_double.make({num_channels}, scales);
  Tensor zero_point = tf_long.make({num_channels}, zero_points);

  int64_t quant_min = 0;
  int64_t quant_max = 255;

  TensorFactory<ScalarType::Byte> tfo;
  Tensor out = tfo.zeros({num_channels, block_size});

  // Compute expected values
  std::vector<uint8_t> expected_data(total_elements);
  for (int ch = 0; ch < num_channels; ch++) {
    float inv_scale = 1.0f / static_cast<float>(scales[ch]);
    int64_t ch_zero_point = zero_points[ch];

    for (int i = 0; i < block_size; i++) {
      int idx = ch * block_size + i;
      float val = input_data[idx] * inv_scale;
      // Clamp before converting to avoid overflow
      val = std::max(-1000.0f, std::min(1000.0f, val));
      int32_t qval = static_cast<int32_t>(std::nearbyint(val)) +
          static_cast<int32_t>(ch_zero_point);

      qval = std::min(255, std::max(0, qval));
      expected_data[idx] = static_cast<uint8_t>(qval);
    }
  }
  Tensor expected = tfo.make({num_channels, block_size}, expected_data);

  quantize_per_channel_out(
      input, scale, zero_point, 0, quant_min, quant_max, ScalarType::Byte, out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpQuantizeOutTest, PerChannelAxis1LargeBlocks) {
  // Test per-channel quantization with axis=1 and large contiguous blocks
  // Shape: [2, 3, 64] with axis=1 (2 batches, 3 channels, 64 elements each)
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Double> tf_double;
  TensorFactory<ScalarType::Long> tf_long;

  const int batch_size = 2;
  const int num_channels = 3;
  const int block_size = 64;
  const int total_elements = batch_size * num_channels * block_size;

  std::vector<float> input_data(total_elements);
  for (int i = 0; i < total_elements; i++) {
    input_data[i] = static_cast<float>(i % 100) * 0.1f;
  }
  Tensor input =
      tf_float.make({batch_size, num_channels, block_size}, input_data);

  Tensor scale = tf_double.make({num_channels}, {0.05, 0.1, 0.15});
  Tensor zero_point = tf_long.make({num_channels}, {100, 110, 120});

  int64_t quant_min = 0;
  int64_t quant_max = 255;

  TensorFactory<ScalarType::Byte> tfo;
  Tensor out = tfo.zeros({batch_size, num_channels, block_size});

  // Compute expected values
  std::vector<uint8_t> expected_data(total_elements);
  for (int b = 0; b < batch_size; b++) {
    for (int ch = 0; ch < num_channels; ch++) {
      double ch_scale = scale.const_data_ptr<double>()[ch];
      int64_t ch_zero_point = zero_point.const_data_ptr<int64_t>()[ch];

      for (int i = 0; i < block_size; i++) {
        int idx = (b * num_channels + ch) * block_size + i;
        float val = input_data[idx] / static_cast<float>(ch_scale);
        int32_t qval = static_cast<int32_t>(std::nearbyint(val)) +
            static_cast<int32_t>(ch_zero_point);
        qval = std::min(255, std::max(0, qval));
        expected_data[idx] = static_cast<uint8_t>(qval);
      }
    }
  }
  Tensor expected =
      tfo.make({batch_size, num_channels, block_size}, expected_data);

  quantize_per_channel_out(
      input, scale, zero_point, 1, quant_min, quant_max, ScalarType::Byte, out);

  EXPECT_TENSOR_EQ(out, expected);
}
