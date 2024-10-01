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
using torch::executor::native::quantized_embedding_2bit_out;

using torch::executor::testing::TensorFactory;

TEST(OpQuantizedEmbedding2bTest, TestGroupWiseQuantizedEmbedding) {
  et_pal_init();
  TensorFactory<ScalarType::Byte> tfb;
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tfl;

  int64_t quant_min = -2;
  int64_t quant_max = 1;

  Tensor weight_scales = tf.make({3}, {0.5, 1.0, 1.5});
  Tensor weight_zero_points = tf.make({3}, {1, -2, 0});

  // -2,  1,  0, 1, -> 0, 3, 2, 3 -> 00 11 10 11 -> 59
  //  0, -1, -2, 0, -> 2, 1, 0, 2 -> 10 01 00 10 -> 146
  // -2,  -1, 0, 1, -> 0, 1, 2, 3 -> 00 01 10 11 -> 27

  Tensor qweight = tfb.make({3, 1}, {59, 146, 27});

  Tensor indices = tfl.make({3}, {0, 2, 1});

  Tensor out = tf.zeros({3, 4});
  Tensor expected = tf.make( 
      {3, 4}, {-1.5, 0.0, -0.5, 0.0, -3.0, -1.5, 0.0, 1.5, -2.0, -3.0, -4.0, -2.0});

  quantized_embedding_2bit_out(
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
  weight_zero_points = tf.make({3, 2}, {1, -2, 0, 1, -2, -1});

  // -2,  1,  0, 1, -> 0, 3, 2, 3 -> 00 11 10 11 -> 59
  //  0, -1, -2, 0, -> 2, 1, 0, 2 -> 10 01 00 10 -> 146
  // -2,  -1, 0, 1, -> 0, 1, 2, 3 -> 00 01 10 11 -> 27

  Tensor qweight = tfb.make({3, 1}, {59, 146, 27});

  Tensor indices = tfl.make({3}, {0, 2, 1});

  Tensor out = tf.zeros({3, 4});
  Tensor expected = tf.make( 
      {3, 4}, {-1.5, 0.0, -2.0, -1.0, 0.0, 2.5, 3.0, 6.0, 0.0, -1.5, -6.0, -2.0});

  quantized_embedding_2bit_out(
      qweight,
      weight_scales,
      weight_zero_points,
      quant_min,
      quant_max,
      indices,
      out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpQuantizedEmbedding2bTest, TestGroupWiseQuantizedEmbeddingDeath1) {
  et_pal_init();
  TensorFactory<ScalarType::Byte> tfb;
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tfl;

  int64_t quant_min = -2;
  int64_t quant_max = 1;

  Tensor weight_scales = tf.make({4}, {0.5, 1.0, 1.5, 3.3});
  Tensor weight_zero_points = tf.make({4}, {1, -2, 1, 0});
  Tensor qweight = tfb.make({3, 1}, {59, 146, 27});
  Tensor indices = tfl.make({3}, {0, 2, 1});
  Tensor out = tf.zeros({3, 4});

  ET_EXPECT_DEATH(
      quantized_embedding_2bit_out(
          qweight,
          weight_scales,
          weight_zero_points,
          quant_min,
          quant_max,
          indices,
          out),
      "");
}

TEST(OpQuantizedEmbedding2bTest, TestGroupWiseQuantizedEmbeddingDeath2) {
  et_pal_init();
  TensorFactory<ScalarType::Byte> tfb;
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tfl;

  int64_t quant_min = -2;
  int64_t quant_max = 1;

  Tensor weight_scales = tf.make({2}, {0.5, 1.0});
  Tensor weight_zero_points = tf.make({2}, {1, -2});
  Tensor qweight = tfb.make({3, 1}, {59, 146, 27});
  Tensor indices = tfl.make({3}, {0, 2, 1});
  Tensor out = tf.zeros({3, 4});

  Tensor out = tf.zeros({3, 4});
  ET_EXPECT_DEATH(
      quantized_embedding_2bit_out(
          qweight,
          weight_scales,
          weight_zero_points,
          quant_min,
          quant_max,
          indices,
          out),
      "");
}
