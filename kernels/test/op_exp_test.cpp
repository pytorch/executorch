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

class OpExpOutTest : public OperatorTest {
 protected:
  Tensor& op_exp_out(const Tensor& a, Tensor& out) {
    return torch::executor::aten::exp_outf(context_, a, out);
  }

  template <typename CTYPE>
  CTYPE apply_log(double x) {
    return static_cast<CTYPE>(std::log(x));
  }

  // Common testing for log operator
  template <typename CTYPE_IN, ScalarType DTYPE, ScalarType DTYPE_OUT>
  void test__exp_out() {
    TensorFactory<DTYPE> tf;
    TensorFactory<DTYPE_OUT> tf_out;

    const std::vector<int32_t> sizes = {2, 2};

    // clang-format off
    Tensor x = tf.make(
        sizes,
        {
            apply_log<CTYPE_IN>(1.),  apply_log<CTYPE_IN>(2.),
            apply_log<CTYPE_IN>(4.),  apply_log<CTYPE_IN>(8.),
        });
    // clang-format on

    // clang-format off
    Tensor expected = tf_out.make(
        sizes,
        {
            1.,  2.,
            4.,  8.,
        });
    // clang-format on

    Tensor out = tf_out.zeros(sizes);

    op_exp_out(x, out);
    EXPECT_TENSOR_CLOSE(out, expected);
  }

  // Unhandled output dtypes.
  template <ScalarType OUTPUT_DTYPE>
  void test_exp_invalid_output_dtype_dies() {
    TensorFactory<ScalarType::Float> tf_float;
    TensorFactory<OUTPUT_DTYPE> tf_out;

    const std::vector<int32_t> sizes = {2, 5};

    Tensor in = tf_float.ones(sizes);
    Tensor out = tf_out.zeros(sizes);

    ET_EXPECT_KERNEL_FAILURE(context_, op_exp_out(in, out));
  }
};

TEST_F(OpExpOutTest, AllFloatInputFloatOutputSupport) {
#define TEST_ENTRY(ctype, dtype) \
  test__exp_out<ctype, ScalarType::dtype, ScalarType::Float>();
  ET_FORALL_FLOAT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpExpOutTest, AllFloatInputDoubleOutputSupport) {
#define TEST_ENTRY(ctype, dtype) \
  test__exp_out<ctype, ScalarType::dtype, ScalarType::Double>();
  ET_FORALL_FLOAT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpExpOutTest, HandleBoolInput) {
  // op_exp_out() handles Bool as input.
  TensorFactory<ScalarType::Bool> tf_bool;
  TensorFactory<ScalarType::Float> tf_float;

  const std::vector<int32_t> sizes = {1, 2};

  Tensor a = tf_bool.make(sizes, /*data=*/{true, false});
  Tensor out = tf_float.zeros(sizes);
  Tensor res = tf_float.make(sizes, /*data=*/{2.718282, 1});

  EXPECT_TENSOR_CLOSE(op_exp_out(a, out), res);
}

TEST_F(OpExpOutTest, HandleHalfInput) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "Test Half support only for ExecuTorch mode";
  }
  TensorFactory<ScalarType::Half> tf_half;

  const std::vector<int32_t> sizes = {1, 2};

  Tensor a = tf_half.make(sizes, /*data=*/{-2.5, -3.0});
  Tensor out = tf_half.zeros(sizes);
  Tensor res = tf_half.make(sizes, /*data=*/{0.082085, 0.049787});

  EXPECT_TENSOR_CLOSE(op_exp_out(a, out), res);
}

// Mismatched shape tests.
TEST_F(OpExpOutTest, MismatchedShapesDies) {
  if (SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched shapes";
  }

  TensorFactory<ScalarType::Int> tf_int;
  TensorFactory<ScalarType::Float> tf_float;

  Tensor a = tf_int.ones(/*sizes=*/{4});
  Tensor out = tf_float.ones(/*sizes=*/{2, 2});

  ET_EXPECT_KERNEL_FAILURE(context_, op_exp_out(a, out));
}

TEST_F(OpExpOutTest, AllNonFloatOutputDTypeDies) {
#define TEST_ENTRY(ctype, dtype) \
  test_exp_invalid_output_dtype_dies<ScalarType::dtype>();
  ET_FORALL_INT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

#ifndef USE_ATEN_LIB
TEST_F(OpExpOutTest, DynamicOutputShape) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Float> tf_out;

  const std::vector<int32_t> sizes = {4, 2};
  const std::vector<int32_t> out_size = {8, 1};

  // clang-format off
  Tensor x = tf.make(
      sizes,
      {
          apply_log<float>(1.),  apply_log<float>(2.),
          apply_log<float>(4.),  apply_log<float>(8.),
          apply_log<float>(3.),  apply_log<float>(6.),
          apply_log<float>(7.),  apply_log<float>(5.),
      });
  // clang-format on

  // clang-format off
  Tensor expected = tf_out.make(
      sizes,
      {
          1.,  2.,
          4.,  8.,
          3.,  6.,
          7.,  5.,
      });
  // clang-format on

  Tensor out =
      tf.zeros(out_size, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);

  op_exp_out(x, out);
  EXPECT_TENSOR_CLOSE(out, expected);
}
#endif
