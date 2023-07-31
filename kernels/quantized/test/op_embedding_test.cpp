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
using exec_aten::RuntimeContext;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::native::dequantize_per_tensor_out;
using torch::executor::native::embedding_out;
using torch::executor::native::quantize_per_tensor_out;
using torch::executor::native::quantized_embedding_byte_out;

using torch::executor::testing::TensorFactory;

/// A generic smoke test that works for any dtype that supports ones() and
/// zeros().
template <exec_aten::ScalarType DTYPE>
void test_dtype() {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tf_l;

  float scale = 0.5;
  float zero_point = 1;
  int64_t quant_min = 0;
  int64_t quant_max = 255;

  // clang-format off
  Tensor weight = tf.make({3, 2}, {3.5, 2.0,
                                   4, 1,
                                   5.5, 13.2});
  // clang-format on
  // TODO make these different per dimension once per channel quant ops
  // available
  Tensor weight_scales = tf.full({3}, scale);
  Tensor weight_zero_points = tf.full({3}, zero_point);

  Tensor indices = tf_l.make({2}, {0, 2});

  Tensor out = tf.zeros({2, 2});

  TensorFactory<DTYPE> tfo;
  Tensor qweight = tfo.zeros({3, 2});

  // 3.5 / 0.5 + 1 = 8
  // 2 / 0.5 + 1 = 5
  // 4 / 0.5 + 1 = 9
  // 1 / 0.5 + 1 = 3
  // 5.5 / 0.5 + 1 = 12
  // 13.2 / 0.5 + 1 = 27
  quantize_per_tensor_out(
      weight, scale, (float)zero_point, quant_min, quant_max, DTYPE, qweight);

  quantized_embedding_byte_out(
      qweight,
      weight_scales,
      weight_zero_points,
      quant_min,
      quant_max,
      indices,
      out);

  // (8 - 1) * 0.5 = 3.5
  // (5 - 1) * 0.5 = 2.0
  // (12 - 1) * 0.5 = 5.5
  // (27 - 1) * 0.5 = 13
  // clang-format off
  Tensor expected = tf.make({2, 2}, {3.5, 2,
                                      5.5, 13});
  // clang-format on

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpQuantizedEmbeddingTest, AllDtypesSupported) {
  test_dtype<ScalarType::Byte>();
}

// Q -> DQ -> FP Embedding should be == to Q -> QEmbedding Bytes
TEST(OpQuantizedEmbeddingTest, ConsitencyWithReferencePattern) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Int> tf_i;
  TensorFactory<ScalarType::Long> tf_l;

  float scale = 0.5;
  float zero_point = 1;
  int64_t quant_min = 0;
  int64_t quant_max = 255;

  // Do Q -> QEmbedding Bytes
  Tensor weight = tf.make({3, 1}, {3.5, 5.5, 1.0});
  // TODO make these different per dimension once per channel quant ops
  // available
  Tensor weight_scales = tf.full({3}, scale);
  Tensor weight_zero_points = tf.full({3}, zero_point);

  Tensor indices = tf_l.make({2}, {0, 2});

  Tensor out = tf.zeros({2, 1});
  Tensor fp_out = tf.zeros({2, 1});

  TensorFactory<ScalarType::Byte> tfo;
  Tensor qweight = tfo.zeros({3, 1});
  RuntimeContext context{};
  // 3.5 / 0.5 + 1 = 8
  // 5.5 / 0.5 + 1 = 12
  // 1 / 0.5 + 1 = 3
  quantize_per_tensor_out(
      weight,
      scale,
      (int64_t)zero_point,
      quant_min,
      quant_max,
      ScalarType::Byte,
      qweight);

  quantized_embedding_byte_out(
      qweight,
      weight_scales,
      weight_zero_points,
      quant_min,
      quant_max,
      indices,
      out);

  // Do Q DQ embedding
  dequantize_per_tensor_out(
      qweight,
      scale,
      (int64_t)zero_point,
      quant_min,
      quant_max,
      ScalarType::Byte,
      weight);

  embedding_out(
      context,
      weight,
      indices,
      /*padding_idx=*/0,
      /*scale_grad_by_freq=*/false,
      /*sparse=*/false,
      fp_out);

  // can lossessly dq here so retrive the full information
  // (8 - 1) * 0.5 = 3.5
  // (3 - 1) * 0.5 = 1
  Tensor expected = tf.make({2, 1}, {3.5, 1});
  EXPECT_TENSOR_EQ(out, fp_out);
  EXPECT_TENSOR_EQ(out, expected);
}
