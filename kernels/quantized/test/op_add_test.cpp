/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/NativeFunctions.h> // Declares the aten operator
#include <executorch/kernels/quantized/NativeFunctions.h> // Declares the quantized operator
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/DeathTest.h>

#include <gtest/gtest.h>
#include <limits>

using namespace ::testing;
using exec_aten::ArrayRef;
using exec_aten::optional;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using executorch::runtime::KernelRuntimeContext;
using torch::executor::native::add_out;
using torch::executor::native::dequantize_per_tensor_out;
using torch::executor::native::quantize_per_tensor_out;
using torch::executor::native::quantized_add_out;

using torch::executor::testing::TensorFactory;

/// A generic smoke test that works for any dtype that supports ones() and
/// zeros().
template <exec_aten::ScalarType DTYPE>
void test_dtype() {
  TensorFactory<ScalarType::Float> tf;

  Tensor input1 = tf.full({3, 5}, 3.5);
  Tensor input2 = tf.full({3, 5}, 3.5);
  double scale = 0.5;

  int64_t zero_point = 1;
  int64_t quant_min = 0;
  int64_t quant_max = 255;

  TensorFactory<DTYPE> tfo;
  Tensor qinput1 = tfo.zeros({3, 5});
  Tensor qinput2 = tfo.zeros({3, 5});
  Tensor qoutput = tfo.zeros({3, 5});
  // 3.5 / 0.5 + 1 = 8
  quantize_per_tensor_out(
      input1,
      scale,
      zero_point,
      quant_min,
      quant_max,
      ScalarType::Byte,
      qinput1);

  quantize_per_tensor_out(
      input2,
      scale,
      zero_point,
      quant_min,
      quant_max,
      ScalarType::Byte,
      qinput2);

  quantized_add_out(
      qinput1,
      scale,
      zero_point,
      quant_min,
      quant_max,
      qinput2,
      scale,
      zero_point,
      quant_min,
      quant_max,
      scale,
      zero_point,
      quant_min,
      quant_max,
      qoutput);

  // can lossessly dq here so retrive the full 3.5 in operation
  // (3.5 + 3.5) / 0.5 + 1 = 15
  Tensor expected = tfo.full({3, 5}, 15.0);

  EXPECT_TENSOR_EQ(qoutput, expected);
}

TEST(OpQuantizeAddTest, AllDtypesSupported) {
  test_dtype<ScalarType::Byte>();
}

TEST(OpQuantizeAddTest, DifferentQParams) {
  TensorFactory<ScalarType::Float> tf;

  Tensor input1 = tf.full({3, 5}, 3.5);
  Tensor input2 = tf.full({3, 5}, 3.5);
  double a_scale = 0.5;
  int64_t a_zero_point = 1;

  double b_scale = 0.25;
  int64_t b_zero_point = 2;

  double out_scale = 0.1;
  int64_t out_zero_point = 5;

  int64_t quant_min = 0;
  int64_t quant_max = 255;

  TensorFactory<ScalarType::Byte> tfo;
  Tensor qinput1 = tfo.zeros({3, 5});
  Tensor qinput2 = tfo.zeros({3, 5});
  Tensor qoutput = tfo.zeros({3, 5});
  // 3.5 / 0.5 + 1 = 8
  quantize_per_tensor_out(
      input1,
      a_scale,
      a_zero_point,
      quant_min,
      quant_max,
      ScalarType::Byte,
      qinput1);

  // 3.5 / 0.25 + 2 = 16
  quantize_per_tensor_out(
      input2,
      b_scale,
      b_zero_point,
      quant_min,
      quant_max,
      ScalarType::Byte,
      qinput2);

  quantized_add_out(
      qinput1,
      a_scale,
      a_zero_point,
      quant_min,
      quant_max,
      qinput2,
      b_scale,
      b_zero_point,
      quant_min,
      quant_max,
      out_scale,
      out_zero_point,
      quant_min,
      quant_max,
      qoutput);

  // can lossessly dq here so retrive the full 3.5 in operation
  // (3.5 + 3.5) / 0.1 + 5 = 75
  Tensor expected = tfo.full({3, 5}, 75.0);

  EXPECT_TENSOR_EQ(qoutput, expected);
}

// Q -> DQ -> FP ADD -> Q -> DQ should be == to Q -> QADD -> DQ
TEST(OpQuantizeAddTest, ConsitencyWithReferencePattern) {
  TensorFactory<ScalarType::Float> tf;

  Tensor input1 = tf.full({3, 5}, 3.5);
  Tensor input2 = tf.full({3, 5}, 3.5);
  Tensor dq_input1 = tf.zeros({3, 5});
  Tensor dq_input2 = tf.zeros({3, 5});
  Tensor reference_op_output = tf.zeros({3, 5});
  Tensor reference_pattern_output = tf.zeros({3, 5});
  Tensor fp_output = tf.zeros({3, 5});

  double a_scale = 0.5;
  int64_t a_zero_point = 1;

  double b_scale = 0.25;
  int64_t b_zero_point = 2;

  double out_scale = 0.1;
  int64_t out_zero_point = 5;

  int64_t quant_min = 0;
  int64_t quant_max = 255;

  TensorFactory<ScalarType::Byte> tfo;
  Tensor qinput1 = tfo.zeros({3, 5});
  Tensor qinput2 = tfo.zeros({3, 5});
  Tensor qoutput = tfo.zeros({3, 5});

  optional<ScalarType> out_dtype = optional<ScalarType>();

  KernelRuntimeContext context{};
  // q -> qadd -> dq
  // 3.5 / 0.5 + 1 = 8
  quantize_per_tensor_out(
      input1,
      a_scale,
      a_zero_point,
      quant_min,
      quant_max,
      ScalarType::Byte,
      qinput1);

  // 3.5 / 0.25 + 2 = 16
  quantize_per_tensor_out(
      input2,
      b_scale,
      b_zero_point,
      quant_min,
      quant_max,
      ScalarType::Byte,
      qinput2);

  quantized_add_out(
      qinput1,
      a_scale,
      a_zero_point,
      quant_min,
      quant_max,
      qinput2,
      b_scale,
      b_zero_point,
      quant_min,
      quant_max,
      out_scale,
      out_zero_point,
      quant_min,
      quant_max,
      qoutput);
  dequantize_per_tensor_out(
      qoutput,
      out_scale,
      out_zero_point,
      quant_min,
      quant_max,
      ScalarType::Byte,
      out_dtype,
      reference_op_output);

  // now get results for q -> dq -> fp add -> q -> dq
  dequantize_per_tensor_out(
      qinput1,
      a_scale,
      a_zero_point,
      quant_min,
      quant_max,
      ScalarType::Byte,
      out_dtype,
      dq_input1);

  dequantize_per_tensor_out(
      qinput2,
      b_scale,
      b_zero_point,
      quant_min,
      quant_max,
      ScalarType::Byte,
      out_dtype,
      dq_input2);

  add_out(context, dq_input1, dq_input2, 1.0, fp_output);
  // reuse 'qoutput' tensor as an intermediate
  quantize_per_tensor_out(
      fp_output,
      out_scale,
      out_zero_point,
      quant_min,
      quant_max,
      ScalarType::Byte,
      qoutput);

  dequantize_per_tensor_out(
      qoutput,
      out_scale,
      out_zero_point,
      quant_min,
      quant_max,
      ScalarType::Byte,
      out_dtype,
      reference_pattern_output);

  Tensor expected = tf.full({3, 5}, 7.0);

  // Pattern and op results should both be equal to expected and each other,
  // check all cases explicitly instead of relying on transitivity
  EXPECT_TENSOR_EQ(reference_op_output, expected);
  EXPECT_TENSOR_EQ(reference_pattern_output, expected);
  EXPECT_TENSOR_EQ(reference_op_output, reference_pattern_output);
}

TEST(OpQuantizeAddTest, InvalidMinMaxDies) {
  TensorFactory<ScalarType::Float> tf;

  Tensor input1 = tf.full({3, 5}, 3.5);
  Tensor input2 = tf.full({3, 5}, 3.5);
  double scale = 0.5;
  int64_t zero_point = 1;

  int64_t quant_min = 0;
  int64_t quant_max = 255;
  int64_t out_quant_min = -1;
  int64_t out_quant_max = 256;

  TensorFactory<ScalarType::Byte> tfo;
  Tensor qinput1 = tfo.zeros({3, 5});
  Tensor qinput2 = tfo.zeros({3, 5});
  Tensor qoutput = tfo.zeros({3, 5});
  // 3.5 / 0.5 + 1 = 8
  quantize_per_tensor_out(
      input1,
      scale,
      zero_point,
      quant_min,
      quant_max,
      ScalarType::Byte,
      qinput1);

  // 3.5 / 0.25 + 2 = 16
  quantize_per_tensor_out(
      input2,
      scale,
      zero_point,
      quant_min,
      quant_max,
      ScalarType::Byte,
      qinput2);

  ET_EXPECT_DEATH(
      quantized_add_out(
          qinput1,
          scale,
          zero_point,
          quant_min,
          quant_max,
          qinput2,
          scale,
          zero_point,
          quant_min,
          quant_max,
          scale,
          zero_point,
          out_quant_min,
          out_quant_max,
          qoutput),
      "");
}

TEST(OpQuantizeAddTest, TopOfRangeTest) {
  TensorFactory<ScalarType::Float> tf;

  Tensor input1 = tf.full({3, 5}, 255);
  Tensor input2 = tf.full({3, 5}, 255);
  double a_scale = 1;
  int64_t a_zero_point = 0;

  double b_scale = 1;
  int64_t b_zero_point = 0;

  double out_scale = 1;
  int64_t out_zero_point = 0;

  int64_t quant_min = 0;
  int64_t quant_max = 255;

  TensorFactory<ScalarType::Byte> tfo;
  Tensor qinput1 = tfo.zeros({3, 5});
  Tensor qinput2 = tfo.zeros({3, 5});
  Tensor qoutput = tfo.zeros({3, 5});

  quantize_per_tensor_out(
      input1,
      a_scale,
      a_zero_point,
      quant_min,
      quant_max,
      ScalarType::Byte,
      qinput1);

  // 3.5 / 0.25 + 2 = 16
  quantize_per_tensor_out(
      input2,
      b_scale,
      b_zero_point,
      quant_min,
      quant_max,
      ScalarType::Byte,
      qinput2);

  quantized_add_out(
      qinput1,
      a_scale,
      a_zero_point,
      quant_min,
      quant_max,
      qinput2,
      b_scale,
      b_zero_point,
      quant_min,
      quant_max,
      out_scale,
      out_zero_point,
      quant_min,
      quant_max,
      qoutput);

  Tensor expected = tfo.full({3, 5}, 255);

  EXPECT_TENSOR_EQ(qoutput, expected);
}
