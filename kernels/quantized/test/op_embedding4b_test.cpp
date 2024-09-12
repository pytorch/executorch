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
using exec_aten::ScalarType;
using exec_aten::Tensor;
using executorch::runtime::KernelRuntimeContext;
using torch::executor::native::quantized_embedding_4bit_out;

using torch::executor::testing::TensorFactory;

TEST(OpQuantizedEmbedding4bTest, TestGroupWiseQuantizedEmbedding) {
  et_pal_init();
  TensorFactory<ScalarType::Byte> tfb;
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tfl;

  int64_t quant_min = -8;
  int64_t quant_max = 7;

  Tensor weight_scales = tf.make({3}, {0.5, 1.0, 1.5});
  Tensor weight_zero_points = tf.make({3}, {1, -5, 0});

  // -3,  1,  6, 7,
  //  2, -5, -4, 0,
  // -8,  3, -1, 6,

  Tensor qweight = tfb.make({3, 2}, {89, 239, 163, 72, 11, 126});

  Tensor indices = tfl.make({3}, {0, 2, 1});

  Tensor out = tf.zeros({3, 4});
  Tensor expected = tf.make(
      {3, 4}, {-2.0, 0.0, 2.5, 3.0, -12.0, 4.5, -1.5, 9.0, 7.0, 0.0, 1.0, 5.0});

  quantized_embedding_4bit_out(
      qweight,
      weight_scales,
      weight_zero_points,
      quant_min,
      quant_max,
      indices,
      out);

  EXPECT_TENSOR_EQ(out, expected);

  out = tf.zeros({3, 4});
  auto context = KernelRuntimeContext();
  torch::executor::native::quantized_embedding_4bit_out(
      context,
      qweight,
      weight_scales,
      weight_zero_points,
      quant_min,
      quant_max,
      indices,
      out);

  EXPECT_TENSOR_EQ(out, expected);

  // Groupwise quantization. groupsize = 2
  weight_scales = tf.make({3, 2}, {0.5, 1.0, 1.5, 2.0, 2.5, 3.0});
  weight_zero_points = tf.make({3, 2}, {1, -5, 0, 2, -3, -1});
  /*
  fp_weight = [-2.0,  0.0,  11.0, 12.0,
                3.0, -7.5, -12.0, -4.0,
              -12.5, 15.0,   0.0, 21.0]
  */

  out = tf.zeros({3, 4});
  expected = tf.make(
      {3, 4},
      {-2.0, 0.0, 11.0, 12.0, -12.5, 15.0, 0.0, 21.0, 3.0, -7.5, -12.0, -4.0});

  quantized_embedding_4bit_out(
      qweight,
      weight_scales,
      weight_zero_points,
      quant_min,
      quant_max,
      indices,
      out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpQuantizedEmbedding4bTest, TestGroupWiseQuantizedEmbeddingDeath1) {
  et_pal_init();
  TensorFactory<ScalarType::Byte> tfb;
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tfl;

  int64_t quant_min = -8;
  int64_t quant_max = 7;

  Tensor weight_scales = tf.make({4}, {0.5, 1.0, 1.5, 3.3});
  Tensor weight_zero_points = tf.make({4}, {1, 5, 7, 5});
  Tensor qweight = tfb.make({3, 2}, {89, 239, 163, 72, 11, 126});
  Tensor indices = tfl.make({3}, {0, 2, 1});

  Tensor out = tf.zeros({3, 4});
  ET_EXPECT_DEATH(
      quantized_embedding_4bit_out(
          qweight,
          weight_scales,
          weight_zero_points,
          quant_min,
          quant_max,
          indices,
          out),
      "");
}

TEST(OpQuantizedEmbedding4bTest, TestGroupWiseQuantizedEmbeddingDeath2) {
  et_pal_init();
  TensorFactory<ScalarType::Byte> tfb;
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tfl;

  int64_t quant_min = -8;
  int64_t quant_max = 7;

  Tensor weight_scales = tf.make({2}, {0.5, 1.0});
  Tensor weight_zero_points = tf.make({2}, {1, 5});
  Tensor qweight = tfb.make({3, 2}, {89, 239, 163, 72, 11, 126});
  Tensor indices = tfl.make({3}, {0, 2, 1});

  Tensor out = tf.zeros({3, 4});
  ET_EXPECT_DEATH(
      quantized_embedding_4bit_out(
          qweight,
          weight_scales,
          weight_zero_points,
          quant_min,
          quant_max,
          indices,
          out),
      "");
}
