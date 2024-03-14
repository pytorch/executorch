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

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::SupportedFeatures;
using torch::executor::testing::TensorFactory;

class OpLogOutTest : public OperatorTest {
 protected:
  Tensor& op_log_out(const Tensor& a, Tensor& out) {
    return torch::executor::aten::log_outf(context_, a, out);
  }

  // Common testing for log operator
  template <ScalarType DTYPE, ScalarType OUT_DTYPE>
  void test__log_out() {
    TensorFactory<DTYPE> tf;
    TensorFactory<OUT_DTYPE> tf_out;

    const std::vector<int32_t> sizes = {2, 2};

    Tensor out = tf_out.zeros(sizes);

    // Valid input should give the expected output
    op_log_out(tf.make(sizes, /*data=*/{0, 1, 2, 4}), out);
    EXPECT_TENSOR_CLOSE(
        out, tf_out.make(sizes, /*data=*/{-INFINITY, 0, 0.693147, 1.386294}));
  }

  // Unhandled output dtypes.
  template <ScalarType OUTPUT_DTYPE>
  void test_log_invalid_output_dtype_dies() {
    TensorFactory<ScalarType::Float> tf_float;
    TensorFactory<OUTPUT_DTYPE> tf_out;

    const std::vector<int32_t> sizes = {2, 5};

    Tensor in = tf_float.ones(sizes);
    Tensor out = tf_out.zeros(sizes);

    ET_EXPECT_KERNEL_FAILURE(context_, op_log_out(in, out));
  }
};

TEST_F(OpLogOutTest, AllRealInputHalfOutputSupport) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "Test Half support only for ExecuTorch mode";
  }
#define TEST_ENTRY(ctype, dtype) \
  test__log_out<ScalarType::dtype, ScalarType::Half>();
  ET_FORALL_REALH_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpLogOutTest, AllRealInputFloatOutputSupport) {
#define TEST_ENTRY(ctype, dtype) \
  test__log_out<ScalarType::dtype, ScalarType::Float>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpLogOutTest, AllRealInputDoubleOutputSupport) {
#define TEST_ENTRY(ctype, dtype) \
  test__log_out<ScalarType::dtype, ScalarType::Double>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpLogOutTest, HandleBoolInput) {
  // op_log_out() handles Bool as input.
  TensorFactory<ScalarType::Bool> tf_bool;
  TensorFactory<ScalarType::Float> tf_float;

  const std::vector<int32_t> sizes = {1, 2};

  Tensor a = tf_bool.make(sizes, /*data=*/{true, false});
  Tensor out = tf_float.zeros(sizes);
  Tensor res = tf_float.make(sizes, /*data=*/{0, -INFINITY});

  EXPECT_TENSOR_EQ(op_log_out(a, out), res);
}

// Mismatched shape tests.
TEST_F(OpLogOutTest, MismatchedShapesDies) {
  if (SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched shapes";
  }

  TensorFactory<ScalarType::Int> tf_int;
  TensorFactory<ScalarType::Float> tf_float;

  Tensor a = tf_int.ones(/*sizes=*/{4});
  Tensor out = tf_float.ones(/*sizes=*/{2, 2});

  ET_EXPECT_KERNEL_FAILURE(context_, op_log_out(a, out));
}

TEST_F(OpLogOutTest, AllNonFloatOutputDTypeDies) {
#define TEST_ENTRY(ctype, dtype) \
  test_log_invalid_output_dtype_dies<ScalarType::dtype>();
  ET_FORALL_INT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpLogOutTest, SimpleGeneratedCase) {
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
      {10, 10},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

  Tensor out = tf.zeros({10, 10});
  Tensor ret = op_log_out(x, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpLogOutTest, DynamicShapeUpperBoundSameAsExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.6879220604896545,
       0.8289883136749268,
       0.7889447808265686,
       0.6339777112007141,
       0.8719115853309631,
       0.4185197353363037});
  Tensor expected_result = tf.make(
      {3, 2},
      {-0.37407973408699036,
       -0.18754921853542328,
       -0.23705895245075226,
       -0.4557414948940277,
       -0.1370672583580017,
       -0.8710312247276306});

  Tensor out =
      tf.zeros({3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_log_out(x, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpLogOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.6879220604896545,
       0.8289883136749268,
       0.7889447808265686,
       0.6339777112007141,
       0.8719115853309631,
       0.4185197353363037});
  Tensor expected_result = tf.make(
      {3, 2},
      {-0.37407973408699036,
       -0.18754921853542328,
       -0.23705895245075226,
       -0.4557414948940277,
       -0.1370672583580017,
       -0.8710312247276306});

  Tensor out =
      tf.zeros({10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_log_out(x, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpLogOutTest, DynamicShapeUnbound) {
  GTEST_SKIP() << "Dynamic shape unbound not supported";
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.6879220604896545,
       0.8289883136749268,
       0.7889447808265686,
       0.6339777112007141,
       0.8719115853309631,
       0.4185197353363037});
  Tensor expected_result = tf.make(
      {3, 2},
      {-0.37407973408699036,
       -0.18754921853542328,
       -0.23705895245075226,
       -0.4557414948940277,
       -0.1370672583580017,
       -0.8710312247276306});

  Tensor out =
      tf.zeros({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  Tensor ret = op_log_out(x, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}
