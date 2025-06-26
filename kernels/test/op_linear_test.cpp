/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/kernels/test/supported_features.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>
#include <limits>

namespace {

using executorch::aten::ArrayRef;
using executorch::aten::Scalar;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpLinearOutTest : public OperatorTest {
 protected:
  Tensor& op_linear_out(const Tensor& self, const Tensor& mat2, Tensor& out) {
    return torch::executor::aten::linear_outf(context_, self, mat2, {}, out);
  }

  Tensor& op_linear_out(
      const Tensor& self,
      const Tensor& mat2,
      const Tensor& bias,
      Tensor& out) {
    return torch::executor::aten::linear_outf(context_, self, mat2, bias, out);
  }

  template <class CTYPE, ScalarType DTYPE>
  void test_dtype() {
    TensorFactory<DTYPE> tf;

    if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
      if (DTYPE == ScalarType::Half) {
        GTEST_SKIP()
            << "skip Half because torch::executor::aten::mm_out does not support Half";
        return;
      }
    }

    // matmul gives 19 * 2 * 3 = 114
    Tensor x = tf.full({3, 19}, 2);
    Tensor y = tf.full({5, 19}, 3);

    // Output shape should be (3, 5)
    Tensor out = tf.zeros({3, 5});

    op_linear_out(x, y, out);

    Tensor expected = tf.full({3, 5}, 114);

    EXPECT_TENSOR_EQ(out, expected);
  }
};

TEST_F(OpLinearOutTest, OutputDim) {
  TensorFactory<ScalarType::Int> tf;

  // 3 tensors with compatible dimensions: (3, 5), (3, 4) and (4, 5).
  Tensor x = tf.ones({3, 4});
  Tensor y = tf.ones({5, 4});
  Tensor out = tf.zeros({3, 5});

  Tensor ret = op_linear_out(x, y, out);

  // Should always return the provided out Tensor.
  EXPECT_TENSOR_EQ(ret, out);

  // Expected tensor, filled with 4.
  Tensor expected = tf.full({3, 5}, 4);

  EXPECT_TENSOR_EQ(out, expected);
}

/// A generic smoke test that works for any dtype that supports ones() and
/// zeros().
TEST_F(OpLinearOutTest, AllDtypesSupported) {
#define TEST_ENTRY(ctype, dtype) test_dtype<ctype, ScalarType::dtype>();
  ET_FORALL_REALHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
  // TODO: Also add tests for half, complex, quantized, and other types. Easiest
  // way to do that would be to make TensorFactory support zeros() and ones()
  // for those types.
}

TEST_F(OpLinearOutTest, BiasTest) {
  TensorFactory<ScalarType::Int> tf;

  // Initialize input tensors.
  constexpr int kReduceDim = 4;
  constexpr int kDimX = 3, kDimY = 2;
  constexpr int kValueX = 1;
  constexpr int kValueY = 2;
  constexpr int kValueBias0 = 4, kValueBias1 = 7;
  const Tensor x = tf.full({kDimX, kReduceDim}, kValueX);
  const Tensor y = tf.full({kDimY, kReduceDim}, kValueY);
  const Tensor b = tf.make({kDimY}, {kValueBias0, kValueBias1});
  // Output matrix is also empty
  Tensor out = tf.zeros({kDimX, kDimY});
  // Initialize expected tensor.
  constexpr int kValueExpected0 = kValueX * kValueY * kReduceDim + kValueBias0;
  constexpr int kValueExpected1 = kValueX * kValueY * kReduceDim + kValueBias1;
  // Check that the bias is added to the correct position in the output matrix.
  const Tensor expected = tf.make(
      {kDimX, kDimY},
      {kValueExpected0,
       kValueExpected1,
       kValueExpected0,
       kValueExpected1,
       kValueExpected0,
       kValueExpected1});

  EXPECT_TENSOR_EQ(op_linear_out(x, y, b, out), expected);
}

TEST_F(OpLinearOutTest, BiasBroadcastTest) {
  TensorFactory<ScalarType::Int> tf;

  // Initialize input tensors.
  constexpr int kReduceDim = 4;
  constexpr int kDimX = 3, kDimY = 5;
  constexpr int kValueX = 1;
  constexpr int kValueY = 2;
  constexpr int kValueBias = 4;
  const Tensor x = tf.full({kDimX, kReduceDim}, kValueX);
  const Tensor y = tf.full({kDimY, kReduceDim}, kValueY);
  const Tensor b = tf.full({1}, kValueBias);
  // Output matrix is also empty
  Tensor out = tf.zeros({kDimX, kDimY});
  // Initialize expected tensor.
  constexpr int kValueExpected = kValueX * kValueY * kReduceDim + kValueBias;
  const Tensor expected = tf.full({kDimX, kDimY}, kValueExpected);

  EXPECT_TENSOR_EQ(op_linear_out(x, y, b, out), expected);
}

TEST_F(OpLinearOutTest, BiasDtypeMismatch) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Short> tf_bias;

  // Initialize input tensors.
  constexpr int kReduceDim = 4;
  constexpr int kDimX = 3, kDimY = 5;
  constexpr int kValueX = 1;
  constexpr int kValueY = 2;
  constexpr int kValueBias = 4;
  Tensor x = tf.full({kDimX, kReduceDim}, kValueX);
  Tensor y = tf.full({kDimY, kReduceDim}, kValueY);
  // Same size as output.
  Tensor b = tf_bias.full({kDimY}, kValueBias);
  // Output matrix is also empty
  Tensor out = tf.zeros({kDimX, kDimY});
  // Initialize expected tensor.
  constexpr int kValueExpected = kValueX * kValueY * kReduceDim + kValueBias;
  Tensor expected = tf.full({kDimX, kDimY}, kValueExpected);

  ET_EXPECT_KERNEL_FAILURE(context_, op_linear_out(x, y, b, out));
}

TEST_F(OpLinearOutTest, EmptyInputWithEmptyOutTensorPasses) {
  TensorFactory<ScalarType::Float> tf;

  // Empty input matrices
  Tensor x = tf.make({0, 3}, {});
  Tensor y = tf.make({0, 3}, {});

  // Output matrix is also empty
  Tensor out = tf.make({0, 0}, {});

  Tensor expected = tf.make({0, 0}, {});

  EXPECT_TENSOR_EQ(op_linear_out(x, y, out), expected);
}

TEST_F(OpLinearOutTest, InfinityTensorPasses) {
  TensorFactory<ScalarType::Float> tff;

  Tensor x = tff.full({3, 4}, std::numeric_limits<float>::infinity());
  Tensor y = tff.full({5, 4}, 3);

  // Output shape should be (3, 5)
  Tensor out = tff.zeros({3, 5});

  Tensor expected = tff.full({3, 5}, std::numeric_limits<float>::infinity());

  EXPECT_TENSOR_EQ(op_linear_out(x, y, out), expected);
}

TEST_F(OpLinearOutTest, MismatchedDimensionsDies) {
  TensorFactory<ScalarType::Int> tf;

  Tensor x = tf.full({2, 2}, 3);

  Tensor wrong_y = tf.full({1, 3}, 1);
  Tensor right_y = tf.full({2, 2}, 1);

  // Make an empty out tensor and demonstrate that it's empty.
  Tensor out = tf.full({2, 2}, 0);

  Tensor expected = tf.full({2, 2}, 6);
  ET_EXPECT_KERNEL_FAILURE(context_, op_linear_out(x, wrong_y, out));

  EXPECT_TENSOR_EQ(op_linear_out(x, right_y, out), expected);
}

TEST_F(OpLinearOutTest, MismatchedDimensionSizeDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched dimension size";
  }
  TensorFactory<ScalarType::Int> tf;
  Tensor x = tf.full({2, 2}, 3);

  // wrong_y has incompatible dim
  Tensor wrong_y = tf.full({2, 2, 2}, 1);
  Tensor right_y = tf.full({2, 2}, 1);

  // wrong_out has incompatible dim
  Tensor right_out = tf.ones({2, 2});
  Tensor wrong_out = tf.ones({2, 2, 3});

  ET_EXPECT_KERNEL_FAILURE(context_, op_linear_out(x, right_y, wrong_out));
  ET_EXPECT_KERNEL_FAILURE(context_, op_linear_out(x, wrong_y, right_out));
}

TEST_F(OpLinearOutTest, WrongOutShapeDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle wrong out shape";
  }
  TensorFactory<ScalarType::Int> tf;
  Tensor x = tf.ones({10, 3});

  Tensor y = tf.ones({4, 3});

  // wrong_out has incompatible shape
  Tensor right_out = tf.ones({10, 4});
  Tensor wrong_out = tf.ones({7, 5});

  ET_EXPECT_KERNEL_FAILURE(context_, op_linear_out(x, y, wrong_out));

  EXPECT_TENSOR_EQ(op_linear_out(x, y, right_out), tf.full({10, 4}, 3));
}

TEST_F(OpLinearOutTest, DynamicShapeUpperBoundSameAsExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.17412060499191284,
       0.34793388843536377,
       0.8187907934188843,
       0.9979893565177917,
       0.7049332857131958,
       0.4255824089050293});
  Tensor y = tf.make(
      {4, 2},
      {0.8071839213371277,
       0.31638312339782715,
       0.13667285442352295,
       0.3691965937614441,
       0.9002121090888977,
       0.09420186281204224,
       0.9070476293563843,
       0.9310881495475769});
  Tensor expected_result = tf.make(
      {3, 4},
      {0.2506277561187744,
       0.15225356817245483,
       0.18952149152755737,
       0.48189279437065125,
       0.976661741733551,
       0.480360746383667,
       0.8310978412628174,
       1.6718982458114624,
       0.703657865524292,
       0.2534688115119934,
       0.6746801733970642,
       1.0356627702713013});

  Tensor out =
      tf.zeros({3, 4}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_linear_out(x, y, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpLinearOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.17412060499191284,
       0.34793388843536377,
       0.8187907934188843,
       0.9979893565177917,
       0.7049332857131958,
       0.4255824089050293});
  Tensor y = tf.make(
      {4, 2},
      {0.8071839213371277,
       0.31638312339782715,
       0.13667285442352295,
       0.3691965937614441,
       0.9002121090888977,
       0.09420186281204224,
       0.9070476293563843,
       0.9310881495475769});
  Tensor expected_result = tf.make(
      {3, 4},
      {0.2506277561187744,
       0.15225356817245483,
       0.18952149152755737,
       0.48189279437065125,
       0.976661741733551,
       0.480360746383667,
       0.8310978412628174,
       1.6718982458114624,
       0.703657865524292,
       0.2534688115119934,
       0.6746801733970642,
       1.0356627702713013});

  Tensor out =
      tf.zeros({10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_linear_out(x, y, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpLinearOutTest, DynamicShapeUnbound) {
  GTEST_SKIP() << "Dynamic shape not supported";
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.17412060499191284,
       0.34793388843536377,
       0.8187907934188843,
       0.9979893565177917,
       0.7049332857131958,
       0.4255824089050293});
  Tensor y = tf.make(
      {4, 2},
      {0.8071839213371277,
       0.31638312339782715,
       0.13667285442352295,
       0.3691965937614441,
       0.9002121090888977,
       0.09420186281204224,
       0.9070476293563843,
       0.9310881495475769});
  Tensor expected_result = tf.make(
      {3, 4},
      {0.2506277561187744,
       0.15225356817245483,
       0.18952149152755737,
       0.48189279437065125,
       0.976661741733551,
       0.480360746383667,
       0.8310978412628174,
       1.6718982458114624,
       0.703657865524292,
       0.2534688115119934,
       0.6746801733970642,
       1.0356627702713013});

  Tensor out =
      tf.zeros({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  Tensor ret = op_linear_out(x, y, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}
} // namespace
