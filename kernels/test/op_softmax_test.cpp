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
using torch::executor::testing::TensorFactory;

class OpSoftmaxOutTest : public OperatorTest {
 protected:
  Tensor& op_softmax_out(
      const Tensor& self,
      int64_t dim,
      bool half_to_float,
      Tensor& out) {
    return torch::executor::aten::_softmax_outf(
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

    op_softmax_out(x, /*dim=*/1, /*half_to_float*/ false, out);

    // clang-format off
    Tensor expected = tf.make(
      {2, 3},
      {
        0.0900306, 0.244728, 0.665241,
        0.0900306, 0.244728, 0.665241
      });
    // clang-format on

    EXPECT_TENSOR_CLOSE(out, expected);
  }
};

TEST_F(OpSoftmaxOutTest, Smoke) {
  TensorFactory<ScalarType::Float> tff;
  std::vector<int32_t> sizes = {1, 3};
  Tensor in = tff.make(sizes, {0, 1, 2});
  Tensor out = tff.zeros(sizes);

  Tensor ret = op_softmax_out(in, /*dim=*/1, /*half_to_float=*/false, out);

  // Should always return the provided out Tensor.
  EXPECT_TENSOR_EQ(ret, out);

  // Expected tensor.
  Tensor expected = tff.make({1, 3}, {0.0900306, 0.244728, 0.665241});

  EXPECT_TENSOR_CLOSE(out, expected);
}

TEST_F(OpSoftmaxOutTest, HalfSupport) {
  TensorFactory<ScalarType::Half> tfh;
  std::vector<int32_t> sizes = {1, 4};
  Tensor in = tfh.ones(sizes);
  Tensor out = tfh.zeros(sizes);

  Tensor ret = op_softmax_out(in, /*dim=*/1, /*half_to_float=*/false, out);

  // Should always return the provided out Tensor.
  EXPECT_TENSOR_EQ(ret, out);

  // Expected tensor.
  Tensor expected = tfh.make({1, 4}, {0.25, 0.25, 0.25, 0.25});

  EXPECT_TENSOR_CLOSE(out, expected);
}

TEST_F(OpSoftmaxOutTest, AllDtypesSupported) {
  test_dtype<float, ScalarType::Float>();
  test_dtype<double, ScalarType::Double>();
  // TODO: Also add tests for half, complex, quantized, and other types. Easiest
  // way to do that would be to make TensorFactory support zeros() and ones()
  // for those types.
}

TEST_F(OpSoftmaxOutTest, MismatchedDimensionsDies) {
  TensorFactory<ScalarType::Float> tff;

  // Input tensor with shape (1, 3) and values (0, 1, 2).
  Tensor x = tff.make({1, 3}, {0, 1, 2});

  // Output shape should be (1, 3)
  Tensor out = tff.zeros({1, 3});

  // Dim out of bounds
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_softmax_out(x, /*dim=*/3, /*half_to_float*/ false, out));
}

TEST_F(OpSoftmaxOutTest, MismatchedDimensionSizeDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched dimension size";
  }
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.ones({3, 4});

  // wrong_out has incompatible dim
  Tensor wrong_out = tf.zeros({2, 10, 4});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_softmax_out(x, /*dim=*/1, /*half_to_float*/ false, wrong_out));
}

TEST_F(OpSoftmaxOutTest, NegativeDim) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel test fails";
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

  op_softmax_out(x, /*dim=*/1, /*half_to_float=*/false, out);
  op_softmax_out(x, /*dim=*/-1, /*half_to_float=*/false, out_negative_dim);

  // clang-format off
  Tensor expected = tf.make(
    {2, 3},
    {
      0.0900306, 0.244728, 0.665241,
      0.0900306, 0.244728, 0.665241
    });
  // clang-format on

  EXPECT_TENSOR_CLOSE(out, expected);
  EXPECT_TENSOR_CLOSE(out_negative_dim, expected);

  op_softmax_out(x, /*dim=*/0, /*half_to_float=*/false, out);
  op_softmax_out(x, /*dim=*/-2, /*half_to_float=*/false, out_negative_dim);

  // clang-format off
  expected = tf.make(
    {2, 3},
    {
        0.0474259, 0.0474259, 0.0474259,
        0.952574, 0.952574, 0.952574
    });
  // clang-format on

  EXPECT_TENSOR_CLOSE(out, expected);
  EXPECT_TENSOR_CLOSE(out_negative_dim, expected);
}

TEST_F(OpSoftmaxOutTest, SimpleGeneratedCase) {
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
      {10, 10}, {0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                 0.10000000149011612});

  Tensor out = tf.zeros({10, 10});
  Tensor ret = op_softmax_out(x, 1, false, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpSoftmaxOutTest, DynamicShapeUpperBoundSameAsExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.3893158435821533,
       0.4583776593208313,
       0.14476794004440308,
       0.44050133228302,
       0.2491583228111267,
       0.8098345994949341});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.4827413856983185,
       0.5172585844993591,
       0.426600843667984,
       0.5733991861343384,
       0.3633909821510315,
       0.6366089582443237});

  Tensor out =
      tf.zeros({3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_softmax_out(x, 1, false, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpSoftmaxOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.3893158435821533,
       0.4583776593208313,
       0.14476794004440308,
       0.44050133228302,
       0.2491583228111267,
       0.8098345994949341});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.4827413856983185,
       0.5172585844993591,
       0.426600843667984,
       0.5733991861343384,
       0.3633909821510315,
       0.6366089582443237});

  Tensor out =
      tf.zeros({10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_softmax_out(x, 1, false, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpSoftmaxOutTest, DynamicShapeUnbound) {
  GTEST_SKIP() << "Dynamic shape unbound not supported";
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.3893158435821533,
       0.4583776593208313,
       0.14476794004440308,
       0.44050133228302,
       0.2491583228111267,
       0.8098345994949341});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.4827413856983185,
       0.5172585844993591,
       0.426600843667984,
       0.5733991861343384,
       0.3633909821510315,
       0.6366089582443237});

  Tensor out =
      tf.zeros({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  Tensor ret = op_softmax_out(x, 1, false, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}
