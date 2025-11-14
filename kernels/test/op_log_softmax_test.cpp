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
using executorch::aten::ArrayRef;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
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
  template <class CTYPE, executorch::aten::ScalarType DTYPE>
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

    if constexpr (DTYPE == ScalarType::BFloat16) {
      EXPECT_TENSOR_CLOSE_WITH_TOL(
          out,
          expected,
          1e-2,
          executorch::runtime::testing::internal::kDefaultAtol);
    } else {
      EXPECT_TENSOR_CLOSE(out, expected);
    }
  }

  template <class CTYPE, executorch::aten::ScalarType DTYPE>
  void test_dtype_noncontiguous_dim() {
    TensorFactory<DTYPE> tf;

    // Dim 0 must be longer than the vector width of the machine (for
    // float, this is 4 for ARM64 and 8 for AVX2) to exhibit problems.
    // clang-format off
    Tensor x = tf.make(
      {9, 3},
      {
        0, 9,  18,
        1, 10, 19,
        2, 11, 20,
        3, 12, 21,
        4, 13, 22,
        5, 14, 23,
        6, 15, 24,
        7, 16, 25,
        8, 17, 26,
      });
    // clang-format on

    Tensor out = tf.zeros({9, 3});

    op_log_softmax_out(x, /*dim=*/0, /*half_to_float*/ false, out);

    // clang-format off
    Tensor expected = tf.make(
      {9, 3},
      {
        -8.45855, -8.45855, -8.45855,
        -7.45855, -7.45855, -7.45855,
        -6.45855, -6.45855, -6.45855,
        -5.45855, -5.45855, -5.45855,
        -4.45855, -4.45855, -4.45855,
        -3.45855, -3.45855, -3.45855,
        -2.45855, -2.45855, -2.45855,
        -1.45855, -1.45855, -1.45855,
        -0.458552, -0.458552, -0.458552
      });
    // clang-format on

    if constexpr (DTYPE == ScalarType::BFloat16) {
      EXPECT_TENSOR_CLOSE_WITH_TOL(
          out,
          expected,
          1e-2,
          executorch::runtime::testing::internal::kDefaultAtol);
    } else {
      EXPECT_TENSOR_CLOSE(out, expected);
    }
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

#define TEST_ENTRY(ctype, dtype) test_dtype<ctype, ScalarType::dtype>();
  ET_FORALL_FLOATHBF16_TYPES(TEST_ENTRY)
#undef TEST_ENTRY
}

TEST_F(OpLogSoftmaxOutTest, NonContiguous) {
  test_dtype_noncontiguous_dim<float, ScalarType::Float>();
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

TEST_F(OpLogSoftmaxOutTest, DoubleCase) {
  TensorFactory<ScalarType::Double> tf;

  // Test case with specific inputs:
  // Input tensor: torch.float64 (8, 5, 7)
  // Dim: 2
  // half_to_float: False
  Tensor input = tf.zeros({8, 5, 7});
  auto in_data = input.mutable_data_ptr<double>();

  // Fill with some test data (sequential values scaled)
  for (int i = 0; i < 8 * 5 * 7; i++) {
    in_data[i] = static_cast<double>(i) * 0.01;
  }

  // Output tensor with same shape
  Tensor out = tf.zeros({8, 5, 7});

  // Apply log_softmax along dimension 2 (the last dimension with size 7)
  op_log_softmax_out(input, /*dim=*/2, /*half_to_float=*/false, out);

  if (!SupportedFeatures::get()->op_log_softmax_dtype_double) {
    // For optimized kernels, we expect the call above to fail gracefully
    expect_failure();
    GTEST_SKIP() << "This kernel does not support dtype double";
  }

  // Verify output dimensions
  EXPECT_EQ(out.size(0), 8);
  EXPECT_EQ(out.size(1), 5);
  EXPECT_EQ(out.size(2), 7);

  // Verify that output has reasonable values
  auto out_data = out.const_data_ptr<double>();

  // Check for NaN or Inf values
  for (int i = 0; i < 8 * 5 * 7; i++) {
    EXPECT_FALSE(std::isnan(out_data[i]))
        << "Output should not contain NaN at index " << i;
    EXPECT_FALSE(std::isinf(out_data[i]))
        << "Output should not contain Inf at index " << i;
  }

  // For log_softmax, all values should be <= 0 (since softmax values are <= 1,
  // log is <= 0)
  for (int i = 0; i < 8 * 5 * 7; i++) {
    EXPECT_LE(out_data[i], 0.0)
        << "Log softmax values should be <= 0 at index " << i;
  }

  // Verify that each slice along dimension 2 sums to approximately 1 when exp'd
  // This tests the core property of softmax: sum(softmax(x)) = 1
  for (int batch = 0; batch < 8; batch++) {
    for (int channel = 0; channel < 5; channel++) {
      double sum_exp = 0.0;
      for (int dim2 = 0; dim2 < 7; dim2++) {
        int idx = batch * 5 * 7 + channel * 7 + dim2;
        sum_exp += std::exp(out_data[idx]);
      }
      EXPECT_NEAR(sum_exp, 1.0, 1e-6)
          << "Sum of exp(log_softmax) should be 1.0 for batch=" << batch
          << ", channel=" << channel;
    }
  }
}
