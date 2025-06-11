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
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::ET_RUNTIME_NAMESPACE::KernelRuntimeContext;
using std::optional;
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

  // -2,  1,  0, 1, -> 0, 3, 2, 3 -> (reverse) 11 10 11 00 -> 236
  //  0, -1, -2, 0, -> 2, 1, 0, 2 -> (reverse) 10 00 01 10 -> 134
  // -2,  -1, 0, 1, -> 0, 1, 2, 3 -> (reverse) 11 10 01 00 -> 228

  Tensor qweight = tfb.make({3, 1}, {236, 134, 228});

  Tensor indices = tfl.make({3}, {0, 2, 1});

  Tensor out = tf.zeros({3, 4});
  Tensor expected = tf.make(
      {3, 4}, {-1.5, 0.0, -0.5, 0.0, -3.0, -1.5, 0.0, 1.5, 2.0, 1.0, 0.0, 2.0});

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
  torch::executor::native::quantized_embedding_2bit_out(
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

  // -2,  1,  0, 1, -> 0, 3, 2, 3 -> (reverse) 11 10 11 00 -> 236
  //  0, -1, -2, 0, -> 2, 1, 0, 2 -> (reverse) 10 00 01 10 -> 134
  // -2,  -1, 0, 1, -> 0, 1, 2, 3 -> (reverse) 11 10 01 00 -> 228

  qweight = tfb.make({3, 1}, {236, 134, 228});

  indices = tfl.make({3}, {0, 2, 1});

  out = tf.zeros({3, 4});
  expected = tf.make(
      {3, 4}, {-1.5, 0.0, 2.0, 3.0, 0.0, 2.5, 3.0, 6.0, 0.0, -1.5, -6.0, -2.0});

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
  Tensor qweight = tfb.make({3, 1}, {236, 134, 228});
  Tensor indices = tfl.make({3}, {0, 2, 1});
  Tensor out = tf.zeros({3, 4});

  // qvals are incompatible shape with scales/zeros
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
  Tensor qweight = tfb.make({3, 1}, {236, 134, 228});
  Tensor indices = tfl.make({3}, {0, 2, 1});
  Tensor out = tf.zeros({3, 4});

  // qvals are incompatible shape with scales/zeros
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

TEST(OpQuantizedEmbedding2bTest, TestGroupWiseQuantizedEmbeddingDeath3) {
  et_pal_init();
  TensorFactory<ScalarType::Byte> tfb;
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> tfl;

  int64_t quant_min = -2;
  int64_t quant_max = 1;

  Tensor weight_scales = tf.make({2, 3}, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
  Tensor weight_zero_points = tf.make({2, 3}, {0, 0, 0, 0, 0, 0});
  Tensor qweight = tfb.make({2, 1}, {236, 134});
  Tensor indices = tfl.make({2}, {0, 2});
  Tensor out = tf.zeros({2, 8});

  // scales/zeros imply 3 groups, which does not divide embed dimension from
  // qvals (8)
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
