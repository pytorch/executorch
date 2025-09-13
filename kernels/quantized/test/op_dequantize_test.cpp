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
using executorch::aten::ArrayRef;
using executorch::aten::Scalar;
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
      std::optional<ScalarType>(),
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
      std::optional<ScalarType>(OUT_DTYPE),
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
      std::optional<ScalarType>(ScalarType::Half),
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
      std::optional<ScalarType>(ScalarType::Double),
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
      std::optional<ScalarType>(),
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
      std::optional<ScalarType>(),
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
      std::optional<ScalarType>(),
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
      std::optional<ScalarType>(),
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
      std::optional<ScalarType>(),
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
      std::optional<ScalarType>(),
      out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpDequantizeOutTest, DequantizePerChannel) {
  et_pal_init();
  test_per_channel_dtype<ScalarType::Byte>();
  test_per_channel_dtype<ScalarType::Char>();
}
