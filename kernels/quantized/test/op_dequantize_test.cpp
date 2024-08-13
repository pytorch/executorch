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
using exec_aten::ArrayRef;
using exec_aten::optional;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
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
  test_dtype<ScalarType::Byte>();
}

TEST(OpDequantizeOutTest, NonWholeNumbers) {
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

TEST(OpDequantizeOutTest, DequantizePerChannel) {
  TensorFactory<ScalarType::Byte> tf_byte;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor input = tf_byte.full({3, 2}, 100);
  Tensor scale = tf_float.make({2}, {0.5, 1});
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
      ScalarType::Byte,
      optional<ScalarType>(),
      out);

  EXPECT_TENSOR_EQ(out, expected);

  // Test with a different axis
  out = tfo.zeros({3, 2});
  scale = tf_float.make({3}, {0.5, 0.75, 1});
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
      ScalarType::Byte,
      optional<ScalarType>(),
      out);

  EXPECT_TENSOR_EQ(out, expected);

  // Test with a different axis
  out = tfo.zeros({3});
  input = tf_byte.make({3}, {100, 100, 100});
  scale = tf_float.make({3}, {0.5, 0.75, 1});
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
      ScalarType::Byte,
      optional<ScalarType>(),
      out);
  EXPECT_TENSOR_EQ(out, expected);
}
