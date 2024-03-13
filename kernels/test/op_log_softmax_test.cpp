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

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ArrayRef;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::SupportedFeatures;
using torch::executor::testing::TensorFactory;

class OpLogSoftmaxOutTest : public OperatorTest {
 protected:
  Tensor& op_log_softmax_out(
      const Tensor& self,
      int64_t dim,
      bool half_to_float,
      Tensor& out) {
    return torch::executor::aten::_log_softmax_outf(
        context_, self, dim, half_to_float, out);
  }

  // A generic smoke test that works for the supported dtypes.
  template <class CTYPE, exec_aten::ScalarType DTYPE>
  void test_dtype() {
    TensorFactory<DTYPE> tf;

    // Input tensor with shape (2, 3) and values (0, 1, 2, 3, 4, 5).
    // clang-format off
    Tensor x = tf.make(
      {2, 3},
      {
        0, 1, 2,
        3, 4, 5
      });
    // clang-format on

    Tensor out = tf.zeros({2, 3});

    op_log_softmax_out(x, /*dim=*/1, /*half_to_float*/ false, out);

    // clang-format off
    Tensor expected = tf.make(
      {2, 3},
      {
        -2.40761, -1.40761, -0.407606,
        -2.40761, -1.40761, -0.407606
      });
    // clang-format on

    EXPECT_TENSOR_CLOSE(out, expected);
  }
};

TEST_F(OpLogSoftmaxOutTest, Smoke) {
  TensorFactory<ScalarType::Float> tff;
  std::vector<int32_t> sizes = {1, 3};
  Tensor in = tff.make(sizes, {0, 1, 2});
  Tensor out = tff.zeros(sizes);

  Tensor ret = op_log_softmax_out(in, /*dim=*/1, /*half_to_float=*/false, out);

  // Should always return the provided out Tensor.
  EXPECT_TENSOR_EQ(ret, out);

  // Expected tensor.
  Tensor expected = tff.make({1, 3}, {-2.40761, -1.40761, -0.407606});

  EXPECT_TENSOR_CLOSE(out, expected);
}

TEST_F(OpLogSoftmaxOutTest, AllDtypesSupported) {
  if (!SupportedFeatures::get()->op_log_softmax_dtype_double) {
    GTEST_SKIP() << "This kernel does not support dtype double";
  }

  test_dtype<float, ScalarType::Float>();
  test_dtype<double, ScalarType::Double>();
  // TODO: Also add tests for half, complex, quantized, and other types. Easiest
  // way to do that would be to make TensorFactory support zeros() and ones()
  // for those types.
}

TEST_F(OpLogSoftmaxOutTest, MismatchedDimensionsDies) {
  if (SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen currently supports mismatched dimensions";
  }

  TensorFactory<ScalarType::Float> tff;

  // Input tensor with shape (1, 3) and values (0, 1, 2).
  Tensor x = tff.make({1, 3}, {0, 1, 2});

  // Output shape should be (1, 3)
  Tensor out = tff.zeros({1, 3});

  // Dim out of bounds
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_log_softmax_out(x, /*dim=*/3, /*half_to_float*/ false, out));
}

TEST_F(OpLogSoftmaxOutTest, MismatchedDimensionSizeDies) {
  if (SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen currently supports mismatched dimension size";
  }

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.ones({3, 4});

  // wrong_out has incompatible dim
  Tensor wrong_out = tf.zeros({2, 10, 4});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_log_softmax_out(x, /*dim=*/1, /*half_to_float*/ false, wrong_out));
}

TEST_F(OpLogSoftmaxOutTest, TestWithLargeNumber) {
  if (!SupportedFeatures::get()->op_log_softmax_dtype_double) {
    GTEST_SKIP() << "This kernel does not support dtype double";
  }

  if (SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen does not support mixing float and double";
  }

  TensorFactory<ScalarType::Double> tf;

  // Input tensor with shape (1, 2) and values (-1e5, 1e5).
  // clang-format off
  Tensor x = tf.make(
    {1, 2},
    {
      -1e5, 1e5
    });
  // clang-format on

  Tensor out = tf.zeros({1, 2});

  op_log_softmax_out(x, /*dim=*/1, /*half_to_float*/ false, out);

  // clang-format off
  Tensor expected = tf.make(
    {1, 2},
    {
      -200000, 0
    });
  // clang-format on

  EXPECT_TENSOR_CLOSE(out, expected);
}

TEST_F(OpLogSoftmaxOutTest, NegativeDim) {
  if (!SupportedFeatures::get()->op_log_softmax_dtype_double) {
    GTEST_SKIP() << "This kernel does not support dtype double";
  }

  if (SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen does not support negative dim";
  }

  TensorFactory<ScalarType::Float> tf;

  // Input tensor with shape (2, 3) and values (0, 1, 2, 3, 4, 5).
  // clang-format off
  Tensor x = tf.make(
    {2, 3},
    {
      0, 1, 2,
      3, 4, 5
    });
  // clang-format on

  Tensor out = tf.zeros({2, 3});
  Tensor out_negative_dim = tf.zeros({2, 3});

  op_log_softmax_out(x, /*dim=*/1, /*half_to_float=*/false, out);
  op_log_softmax_out(x, /*dim=*/-1, /*half_to_float=*/false, out_negative_dim);

  // clang-format off
  Tensor expected = tf.make(
    {2, 3},
    {
      -2.40761, -1.40761, -0.407606,
      -2.40761, -1.40761, -0.407606
    });
  // clang-format on

  EXPECT_TENSOR_CLOSE(out, expected);
  EXPECT_TENSOR_CLOSE(out_negative_dim, expected);

  op_log_softmax_out(x, /*dim=*/0, /*half_to_float=*/false, out);
  op_log_softmax_out(x, /*dim=*/-2, /*half_to_float=*/false, out_negative_dim);

  // clang-format off
  expected = tf.make(
    {2, 3},
    {
        -3.04859, -3.04859, -3.04859,
        -0.0485874, -0.0485874, -0.0485874
    });
  // clang-format on

  EXPECT_TENSOR_CLOSE(out, expected);
  EXPECT_TENSOR_CLOSE(out_negative_dim, expected);
}

#if !defined(USE_ATEN_LIB)
TEST_F(OpLogSoftmaxOutTest, UpperBoundOutTensor) {
  TensorFactory<ScalarType::Float> tff;

  // Input tensor with shape (2, 3) and values (0, 1, 2, 3, 4, 5).
  // clang-format off
  Tensor x = tff.make(
    {2, 3},
    {
      0, 1, 2,
      3, 4, 5
    });
  // clang-format on

  Tensor out =
      tff.zeros({5, 9}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);

  op_log_softmax_out(x, /*dim=*/1, /*half_to_float*/ false, out);

  // clang-format off
  Tensor expected = tff.make(
    {2, 3},
    {
      -2.40761, -1.40761, -0.407606,
      -2.40761, -1.40761, -0.407606
    });
  // clang-format on

  EXPECT_TENSOR_CLOSE(out, expected);
}
#endif

TEST_F(OpLogSoftmaxOutTest, SimpleGeneratedCase) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {10, 10},
      {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
  Tensor expected_result = tf.make(
      {10, 10}, {-2.3025851249694824, -2.3025851249694824, -2.3025851249694824,
                 -2.3025851249694824, -2.3025851249694824, -2.3025851249694824,
                 -2.3025851249694824, -2.3025851249694824, -2.3025851249694824,
                 -2.3025851249694824, -2.3025851249694824, -2.3025851249694824,
                 -2.3025851249694824, -2.3025851249694824, -2.3025851249694824,
                 -2.3025851249694824, -2.3025851249694824, -2.3025851249694824,
                 -2.3025851249694824, -2.3025851249694824, -2.3025851249694824,
                 -2.3025851249694824, -2.3025851249694824, -2.3025851249694824,
                 -2.3025851249694824, -2.3025851249694824, -2.3025851249694824,
                 -2.3025851249694824, -2.3025851249694824, -2.3025851249694824,
                 -2.3025851249694824, -2.3025851249694824, -2.3025851249694824,
                 -2.3025851249694824, -2.3025851249694824, -2.3025851249694824,
                 -2.3025851249694824, -2.3025851249694824, -2.3025851249694824,
                 -2.3025851249694824, -2.3025851249694824, -2.3025851249694824,
                 -2.3025851249694824, -2.3025851249694824, -2.3025851249694824,
                 -2.3025851249694824, -2.3025851249694824, -2.3025851249694824,
                 -2.3025851249694824, -2.3025851249694824, -2.3025851249694824,
                 -2.3025851249694824, -2.3025851249694824, -2.3025851249694824,
                 -2.3025851249694824, -2.3025851249694824, -2.3025851249694824,
                 -2.3025851249694824, -2.3025851249694824, -2.3025851249694824,
                 -2.3025851249694824, -2.3025851249694824, -2.3025851249694824,
                 -2.3025851249694824, -2.3025851249694824, -2.3025851249694824,
                 -2.3025851249694824, -2.3025851249694824, -2.3025851249694824,
                 -2.3025851249694824, -2.3025851249694824, -2.3025851249694824,
                 -2.3025851249694824, -2.3025851249694824, -2.3025851249694824,
                 -2.3025851249694824, -2.3025851249694824, -2.3025851249694824,
                 -2.3025851249694824, -2.3025851249694824, -2.3025851249694824,
                 -2.3025851249694824, -2.3025851249694824, -2.3025851249694824,
                 -2.3025851249694824, -2.3025851249694824, -2.3025851249694824,
                 -2.3025851249694824, -2.3025851249694824, -2.3025851249694824,
                 -2.3025851249694824, -2.3025851249694824, -2.3025851249694824,
                 -2.3025851249694824, -2.3025851249694824, -2.3025851249694824,
                 -2.3025851249694824, -2.3025851249694824, -2.3025851249694824,
                 -2.3025851249694824});

  Tensor out = tf.zeros({10, 10});
  Tensor ret = op_log_softmax_out(x, 1, false, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpLogSoftmaxOutTest, DynamicShapeUpperBoundSameAsExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.754019558429718,
       0.8973914980888367,
       0.34469079971313477,
       0.40464818477630615,
       0.36159539222717285,
       0.1138353943824768});
  Tensor expected_result = tf.make(
      {3, 2},
      {-0.7674003839492798,
       -0.6240284442901611,
       -0.7235751748085022,
       -0.6636177897453308,
       -0.576920747756958,
       -0.824680745601654});

  Tensor out =
      tf.zeros({3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_log_softmax_out(x, 1, false, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpLogSoftmaxOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.754019558429718,
       0.8973914980888367,
       0.34469079971313477,
       0.40464818477630615,
       0.36159539222717285,
       0.1138353943824768});
  Tensor expected_result = tf.make(
      {3, 2},
      {-0.7674003839492798,
       -0.6240284442901611,
       -0.7235751748085022,
       -0.6636177897453308,
       -0.576920747756958,
       -0.824680745601654});

  Tensor out =
      tf.zeros({10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_log_softmax_out(x, 1, false, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpLogSoftmaxOutTest, DynamicShapeUnbound) {
  GTEST_SKIP() << "Dynamic shape not supported";
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.754019558429718,
       0.8973914980888367,
       0.34469079971313477,
       0.40464818477630615,
       0.36159539222717285,
       0.1138353943824768});
  Tensor expected_result = tf.make(
      {3, 2},
      {-0.7674003839492798,
       -0.6240284442901611,
       -0.7235751748085022,
       -0.6636177897453308,
       -0.576920747756958,
       -0.824680745601654});

  Tensor out =
      tf.zeros({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  Tensor ret = op_log_softmax_out(x, 1, false, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}
