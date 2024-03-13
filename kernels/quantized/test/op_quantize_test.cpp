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
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
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

  int64_t zero_point = 127;
  int64_t quant_min = 0;
  int64_t quant_max = 255;

  TensorFactory<DTYPE> tfo;
  Tensor out = tfo.zeros({3, 5});
  // 4 / 0.5 + 127
  Tensor expected = tfo.full({3, 5}, 135);
  quantize_per_tensor_out(
      input, scale, zero_point, quant_min, quant_max, DTYPE, out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpQuantizeOutTest, AllDtypesSupported) {
  test_dtype<ScalarType::Byte>();
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
