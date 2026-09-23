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
#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/DeathTest.h>

#include <gtest/gtest.h>
#include <limits>

using namespace ::testing;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using std::optional;
using torch::executor::native::dequantize_per_channel_out;
using torch::executor::native::dequantize_per_tensor_out;
using torch::executor::native::dequantize_per_tensor_tensor_args_out;
using torch::executor::testing::TensorFactory;

/// A generic smoke test that works for any dtype that supports ones() and
/// zeros().
template <ScalarType DTYPE>
void test_dtype() {
  TensorFactory<DTYPE> tf;

  Tensor input = tf.full({3, 5}, 100);
  double scale = 0.5;
  int64_t zero_point = 30;
  int64_t quant_min = 0;
  int64_t quant_max = 255;

  TensorFactory<ScalarType::Float> tfo;
  Tensor out = tfo.zeros({3, 5});
  // (100 - 30) * 0.5
  Tensor expected = tfo.full({3, 5}, 35);
  dequantize_per_tensor_out(
      input,
      scale,
      zero_point,
      quant_min,
      quant_max,
      DTYPE,
      optional<ScalarType>(),
      out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpDequantizeOutTest, AllDtypesSupported) {
  et_pal_init();
  test_dtype<ScalarType::Byte>();
  test_dtype<ScalarType::Char>();
  test_dtype<ScalarType::Short>();
  test_dtype<ScalarType::Bits16>();
  test_dtype<ScalarType::UInt16>();
  test_dtype<ScalarType::Int>();
}

/// Test all supported output dtypes for dequantization
template <ScalarType OUT_DTYPE>
void test_output_dtype() {
  TensorFactory<ScalarType::Byte> tf;

  Tensor input = tf.full({3, 5}, 100);
  double scale = 0.5;
  int64_t zero_point = 30;
  int64_t quant_min = 0;
  int64_t quant_max = 255;

  TensorFactory<OUT_DTYPE> tfo;
  Tensor out = tfo.zeros({3, 5});
  // (100 - 30) * 0.5 = 35
  Tensor expected = tfo.full({3, 5}, 35);
  dequantize_per_tensor_out(
      input,
      scale,
      zero_point,
      quant_min,
      quant_max,
      ScalarType::Byte,
      optional<ScalarType>(OUT_DTYPE),
      out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpDequantizeOutTest, AllOutputDtypesSupported) {
  et_pal_init();
  test_output_dtype<ScalarType::Float>();
  test_output_dtype<ScalarType::Double>();
  test_output_dtype<ScalarType::Half>();
}

TEST(OpDequantizeOutTest, HalfOutput) {
  et_pal_init();
  TensorFactory<ScalarType::Byte> tf;

  Tensor input = tf.full({3, 5}, 10);
  double scale = 0.5;
  int64_t zero_point = 100000;
  int64_t quant_min = 0;
  int64_t quant_max = 255;

  TensorFactory<ScalarType::Half> tfo;
  Tensor out = tfo.zeros({3, 5});
  // (10 - 100000) * 0.5 = -49995
  dequantize_per_tensor_out(
      input,
      scale,
      zero_point,
      quant_min,
      quant_max,
      ScalarType::Byte,
      optional<ScalarType>(ScalarType::Half),
      out);

  // The expected result should be (10 - 100000) * 0.5 = -49995
  Tensor expected = tfo.full({3, 5}, -49995);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpDequantizeOutTest, DoubleOutput) {
  et_pal_init();
  TensorFactory<ScalarType::Byte> tf;

  Tensor input = tf.full({3, 5}, 10);
  double scale = 0.5;
  int64_t zero_point = 100000;
  int64_t quant_min = 0;
  int64_t quant_max = 255;

  TensorFactory<ScalarType::Double> tfo;
  Tensor out = tfo.zeros({3, 5});
  dequantize_per_tensor_out(
      input,
      scale,
      zero_point,
      quant_min,
      quant_max,
      ScalarType::Byte,
      optional<ScalarType>(ScalarType::Double),
      out);

  // The expected result should be (10 - 100000) * 0.5 = -49995
  Tensor expected = tfo.full({3, 5}, -49995);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpDequantizeOutTest, NonWholeNumbers) {
  et_pal_init();
  TensorFactory<ScalarType::Byte> tf;

  Tensor input = tf.full({3, 5}, 100);
  double scale = 0.45;
  int64_t zero_point = 30;
  int64_t quant_min = 0;
  int64_t quant_max = 255;

  TensorFactory<ScalarType::Float> tfo;
  Tensor out = tfo.zeros({3, 5});
  // (100 - 30) * 0.5
  Tensor expected = tfo.full({3, 5}, 31.5);
  dequantize_per_tensor_out(
      input,
      scale,
      zero_point,
      quant_min,
      quant_max,
      ScalarType::Byte,
      optional<ScalarType>(),
      out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpDequantizeOutTest, TensorArgOverload) {
  et_pal_init();
  TensorFactory<ScalarType::Byte> tf_byte;
  TensorFactory<ScalarType::Double> tf_double;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor input = tf_byte.full({3, 5}, 100);
  Tensor scale = tf_double.make({1}, {0.45});
  Tensor zero_point = tf_long.make({1}, {30});
  int64_t quant_min = 0;
  int64_t quant_max = 255;

  TensorFactory<ScalarType::Float> tfo;
  Tensor out = tfo.zeros({3, 5});
  // (100 - 30) * 0.5
  Tensor expected = tfo.full({3, 5}, 31.5);
  dequantize_per_tensor_tensor_args_out(
      input,
      scale,
      zero_point,
      quant_min,
      quant_max,
      ScalarType::Byte,
      optional<ScalarType>(),
      out);

  EXPECT_TENSOR_EQ(out, expected);
}

template <ScalarType DTYPE>
void test_per_channel_dtype() {
  TensorFactory<DTYPE> tf;
  TensorFactory<ScalarType::Double> tf_double;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor input = tf.full({3, 2}, 100);
  Tensor scale = tf_double.make({2}, {0.5, 1});
  Tensor zero_point = tf_long.make({2}, {30, 60});
  int64_t quant_min = 0;
  int64_t quant_max = 255;

  TensorFactory<ScalarType::Float> tfo;
  Tensor out = tfo.zeros({3, 2});
  // (100 - 30) * 0.5
  // (100 - 60) * 1
  Tensor expected = tfo.make({3, 2}, {35, 40, 35, 40, 35, 40});
  dequantize_per_channel_out(
      input,
      scale,
      zero_point,
      /*axis=*/1,
      quant_min,
      quant_max,
      DTYPE,
      optional<ScalarType>(),
      out);

  EXPECT_TENSOR_EQ(out, expected);

  // Test with a different axis
  out = tfo.zeros({3, 2});
  scale = tf_double.make({3}, {0.5, 0.75, 1});
  zero_point = tf_long.make({3}, {30, 50, 60});
  // (100 - 30) * 0.5
  // (100 - 50) * 0.75
  // (100 - 60) * 1
  expected = tfo.make({3, 2}, {35, 35, 37.5, 37.5, 40, 40});
  dequantize_per_channel_out(
      input,
      scale,
      zero_point,
      /*axis=*/0,
      quant_min,
      quant_max,
      DTYPE,
      optional<ScalarType>(),
      out);

  EXPECT_TENSOR_EQ(out, expected);

  // Test with a different axis
  out = tfo.zeros({3});
  input = tf.make({3}, {100, 100, 100});
  scale = tf_double.make({3}, {0.5, 0.75, 1});
  zero_point = tf_long.make({3}, {30, 50, 60});
  // (100 - 30) * 0.5
  // (100 - 50) * 0.75
  // (100 - 60) * 1
  expected = tfo.make({3}, {35, 37.5, 40});
  dequantize_per_channel_out(
      input,
      scale,
      zero_point,
      /*axis=*/0,
      quant_min,
      quant_max,
      DTYPE,
      optional<ScalarType>(),
      out);
  EXPECT_TENSOR_EQ(out, expected);

  // Test with a different axis
  input = tf.full({3, 19}, 100);
  out = tfo.zeros({3, 19});
  scale = tf_double.make({3}, {0.5, 0.75, 1});
  zero_point = tf_long.make({3}, {30, 50, 60});
  // (100 - 30) * 0.5
  // (100 - 50) * 0.75
  // (100 - 60) * 1
  expected = tfo.make(
      {3, 19},
      {35,   35,   35,   35,   35,   35,   35,   35,   35,   35,   35,   35,
       35,   35,   35,   35,   35,   35,   35,   37.5, 37.5, 37.5, 37.5, 37.5,
       37.5, 37.5, 37.5, 37.5, 37.5, 37.5, 37.5, 37.5, 37.5, 37.5, 37.5, 37.5,
       37.5, 37.5, 40,   40,   40,   40,   40,   40,   40,   40,   40,   40,
       40,   40,   40,   40,   40,   40,   40,   40,   40});
  dequantize_per_channel_out(
      input,
      scale,
      zero_point,
      /*axis=*/0,
      quant_min,
      quant_max,
      DTYPE,
      optional<ScalarType>(),
      out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpDequantizeOutTest, DequantizePerChannel) {
  et_pal_init();
  test_per_channel_dtype<ScalarType::Byte>();
  test_per_channel_dtype<ScalarType::Char>();
}

// Per-channel dequantize on a channels-last input. Each element's expected
// value depends only on its channel, never on where it sits in memory, so this
// only passes if the kernel honors the tensor's dim order. Before the
// reduce_util.h carry-over fix this wrote past the end of the output. See
// issue #16429.
TEST(OpDequantizeOutTest, DequantizePerChannelChannelsLast) {
  et_pal_init();
  TensorFactory<ScalarType::Char> tf;
  TensorFactory<ScalarType::Double> tf_double;
  TensorFactory<ScalarType::Long> tf_long;
  TensorFactory<ScalarType::Float> tfo;

  std::vector<int8_t> logical = {
      -20, -59,  -22, -40, 127, -108, -57, 117, 24,  -103, 48,  -110,
      80,  15,   -10, -75, -77, -46,  -12, -66, 35,  -87,  -50, -80,
      12,  -127, 107, 91,  115, -54,  -6,  -6,  -41, 46,   42,  -83};
  const double s0 = 0.0016989057185128331;
  const double s1 = 0.001776964869350195;

  Tensor input = tf.channels_last_like(tf.make({2, 2, 3, 3}, logical));
  Tensor scale = tf_double.make({2}, {s0, s1});
  Tensor zero_point = tf_long.make({2}, {0, 0});

  std::vector<float> expected_logical(36);
  for (int n = 0; n < 2; ++n) {
    for (int c = 0; c < 2; ++c) {
      for (int hw = 0; hw < 9; ++hw) {
        size_t i = n * 18 + c * 9 + hw;
        expected_logical[i] = static_cast<float>(logical[i]) *
            static_cast<float>(c == 0 ? s0 : s1);
      }
    }
  }
  Tensor expected =
      tfo.channels_last_like(tfo.make({2, 2, 3, 3}, expected_logical));

  Tensor out = tfo.zeros_channels_last({2, 2, 3, 3});
  dequantize_per_channel_out(
      input,
      scale,
      zero_point,
      /*axis=*/1,
      /*quant_min=*/-128,
      /*quant_max=*/127,
      ScalarType::Char,
      optional<ScalarType>(),
      out);

  EXPECT_TENSOR_CLOSE(out, expected);
}

// The schema allows an Int zero_point tensor as well as a Long one. Reading an
// Int tensor as int64 reinterprets pairs of channels as a single value and runs
// off the end of the buffer, which silently corrupted every channel.
template <ScalarType DTYPE>
void test_per_channel_int_zero_point() {
  TensorFactory<DTYPE> tf;
  TensorFactory<ScalarType::Double> tf_double;
  TensorFactory<ScalarType::Int> tf_int;
  TensorFactory<ScalarType::Float> tfo;

  Tensor scale = tf_double.make({4}, {0.5, 0.75, 1, 2});
  Tensor zero_point = tf_int.make({4}, {30, 50, 60, 90});
  int64_t quant_min = 0;
  int64_t quant_max = 127;

  // Multi-dimensional input, channel axis 0.
  Tensor input = tf.full({4, 2}, 100);
  Tensor out = tfo.zeros({4, 2});
  Tensor expected = tfo.make({4, 2}, {35, 35, 37.5, 37.5, 40, 40, 20, 20});
  dequantize_per_channel_out(
      input,
      scale,
      zero_point,
      /*axis=*/0,
      quant_min,
      quant_max,
      DTYPE,
      optional<ScalarType>(),
      out);
  EXPECT_TENSOR_EQ(out, expected);

  // Single-dimensional input takes a separate branch in the kernel.
  input = tf.make({4}, {100, 100, 100, 100});
  out = tfo.zeros({4});
  expected = tfo.make({4}, {35, 37.5, 40, 20});
  dequantize_per_channel_out(
      input,
      scale,
      zero_point,
      /*axis=*/0,
      quant_min,
      quant_max,
      DTYPE,
      optional<ScalarType>(),
      out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpDequantizeOutTest, DequantizePerChannelIntZeroPoint) {
  et_pal_init();
  test_per_channel_int_zero_point<ScalarType::Byte>();
  test_per_channel_int_zero_point<ScalarType::Char>();
  test_per_channel_int_zero_point<ScalarType::Short>();
  test_per_channel_int_zero_point<ScalarType::Int>();
}
