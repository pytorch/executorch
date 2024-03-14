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
using torch::executor::testing::TensorFactory;

class OpEqScalarOutTest : public OperatorTest {
 protected:
  Tensor& op_eq_scalar_out(const Tensor& self, Scalar& other, Tensor& out) {
    return torch::executor::aten::eq_outf(context_, self, other, out);
  }

  // Common testing for eq operator
  template <ScalarType DTYPE>
  void test_eq_scalar_out() {
    TensorFactory<DTYPE> tf;
    TensorFactory<ScalarType::Bool> tf_out;

    const std::vector<int32_t> sizes = {2, 2};
    // Destination for the eq
    Tensor out = tf_out.ones(sizes);
    Scalar other = 3;

    // Valid input should give the expected output
    op_eq_scalar_out(tf.make(sizes, /*data=*/{2, 3, 3, 3}), other, out);
    EXPECT_TENSOR_EQ(
        out, tf_out.make(sizes, /*data=*/{false, true, true, true}));
  }

  // Handle all output dtypes.
  template <ScalarType OUTPUT_DTYPE>
  void test_eq_all_output_dtypes() {
    TensorFactory<ScalarType::Float> tf_float;
    TensorFactory<OUTPUT_DTYPE> tf_out;

    const std::vector<int32_t> sizes = {2, 5};

    Tensor in = tf_float.ones(sizes);
    Tensor out = tf_out.zeros(sizes);
    Scalar other = 1;

    op_eq_scalar_out(in, other, out);
    EXPECT_TENSOR_EQ(out, tf_out.ones(sizes));
  }
};

TEST_F(OpEqScalarOutTest, AllRealInputBoolOutputSupport) {
#define TEST_ENTRY(ctype, dtype) test_eq_scalar_out<ScalarType::dtype>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpEqScalarOutTest, BoolInputDtype) {
  TensorFactory<ScalarType::Bool> tf_bool;

  const std::vector<int32_t> sizes = {2, 2};
  Tensor a = tf_bool.make(sizes, /*data=*/{false, true, false, true});
  Tensor out = tf_bool.zeros(sizes);
  Scalar other = 1;

  op_eq_scalar_out(a, other, out);
  EXPECT_TENSOR_EQ(
      out, tf_bool.make(sizes, /*data=*/{false, true, false, true}));
}

// Mismatched shape tests.
TEST_F(OpEqScalarOutTest, MismatchedShapesDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched shapes";
  }
  TensorFactory<ScalarType::Int> tf_int;
  TensorFactory<ScalarType::Bool> tf_bool;

  Tensor a = tf_int.ones(/*sizes=*/{4});
  Tensor out = tf_bool.ones(/*sizes=*/{2, 2});
  Scalar other = 3;

  ET_EXPECT_KERNEL_FAILURE(context_, op_eq_scalar_out(a, other, out));
}

TEST_F(OpEqScalarOutTest, AllRealOutputDTypes) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle non-bool output dtype";
  }
#define TEST_ENTRY(ctype, dtype) test_eq_all_output_dtypes<ScalarType::dtype>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

/* %python
import torch
torch.manual_seed(0)
x = torch.randint(3, (3, 2))
res = torch.eq(x, 2)
op = "op_eq_scalar_out"
opt_setup_params = """
  Scalar other = 2;
"""
opt_extra_params = "other,"
dtype = "ScalarType::Int"
out_dtype = "ScalarType::Bool"
check = "EXPECT_TENSOR_EQ" */

TEST_F(OpEqScalarOutTest, DynamicShapeUpperBoundSameAsExpected) {
  /* %python
  out_args = "{3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op_out_dtype) */

  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tfOut;

  Tensor x = tf.make({3, 2}, {2, 0, 2, 0, 1, 0});
  Tensor expected =
      tfOut.make({3, 2}, {true, false, true, false, false, false});

  Scalar other = 2;

  Tensor out =
      tfOut.zeros({3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_eq_scalar_out(x, other, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpEqScalarOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  /* %python
  out_args = "{10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op_out_dtype) */

  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tfOut;

  Tensor x = tf.make({3, 2}, {2, 0, 2, 0, 1, 0});
  Tensor expected =
      tfOut.make({3, 2}, {true, false, true, false, false, false});

  Scalar other = 2;

  Tensor out = tfOut.zeros(
      {10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_eq_scalar_out(x, other, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpEqScalarOutTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape unbound not supported";
  }
  /* %python
  out_args = "{1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND"
  %rewrite(unary_op_out_dtype) */

  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tfOut;

  Tensor x = tf.make({3, 2}, {2, 0, 2, 0, 1, 0});
  Tensor expected =
      tfOut.make({3, 2}, {true, false, true, false, false, false});

  Scalar other = 2;

  Tensor out = tfOut.zeros(
      {1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  op_eq_scalar_out(x, other, out);
  EXPECT_TENSOR_EQ(out, expected);
}
