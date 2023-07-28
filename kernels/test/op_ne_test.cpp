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
using exec_aten::RuntimeContext;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

Tensor& ne_scalar_out(const Tensor& self, Scalar& other, Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::ne_outf(context, self, other, out);
}

// Common testing for ne operator
template <ScalarType DTYPE>
void test_ne_scalar_out() {
  TensorFactory<DTYPE> tf;
  TensorFactory<ScalarType::Bool> tf_out;

  const std::vector<int32_t> sizes = {2, 2};
  // Destination for the ne
  Tensor out = tf_out.ones(sizes);
  Scalar other = 2;

  // Valid input should give the expected output
  ne_scalar_out(tf.make(sizes, /*data=*/{2, 3, 2, 3}), other, out);
  EXPECT_TENSOR_EQ(
      out, tf_out.make(sizes, /*data=*/{false, true, false, true}));
}

TEST(OpNeScalarOutKernelTest, AllRealInputBoolOutputSupport) {
#define TEST_ENTRY(ctype, dtype) test_ne_scalar_out<ScalarType::dtype>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(OpNeScalarOutKernelTest, BoolInputDtype) {
  TensorFactory<ScalarType::Bool> tf_bool;

  const std::vector<int32_t> sizes = {2, 2};
  Tensor a = tf_bool.make(sizes, /*data=*/{false, true, false, true});
  Tensor out = tf_bool.zeros(sizes);
  Scalar other = 1;

  ne_scalar_out(a, other, out);
  EXPECT_TENSOR_EQ(
      out, tf_bool.make(sizes, /*data=*/{true, false, true, false}));
}

// Mismatched shape tests.
TEST(OpNeScalarOutKernelTest, MismatchedShapesDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched shapes";
  }
  TensorFactory<ScalarType::Int> tf_int;
  TensorFactory<ScalarType::Bool> tf_bool;

  Tensor a = tf_int.ones(/*sizes=*/{4});
  Tensor out = tf_bool.ones(/*sizes=*/{2, 2});
  Scalar other = 3;

  ET_EXPECT_KERNEL_FAILURE(ne_scalar_out(a, other, out));
}

// Handle all output dtypes.
template <ScalarType OUTPUT_DTYPE>
void test_ne_all_output_dtypes() {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<OUTPUT_DTYPE> tf_out;

  const std::vector<int32_t> sizes = {2, 5};

  Tensor in = tf_float.ones(sizes);
  Tensor out = tf_out.zeros(sizes);
  Scalar other = 3;

  ne_scalar_out(in, other, out);
  EXPECT_TENSOR_EQ(out, tf_out.ones(sizes));
}

TEST(OpNeScalarOutKernelTest, AllRealOutputDTypesSupported) {
#define TEST_ENTRY(ctype, dtype) test_ne_all_output_dtypes<ScalarType::dtype>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

template <class CTYPE, ScalarType DTYPE>
void test_dtype() {
  TensorFactory<DTYPE> tf_input;
  TensorFactory<ScalarType::Bool> tf_bool;
  Tensor a = tf_input.make(/*sizes=*/{2, 2}, /*data=*/{2, 3, 2, 4});
  Tensor b = tf_input.make({2, 2}, {2, 2, 2, 2});
  Tensor out = tf_bool.zeros({2, 2});
  RuntimeContext context{};

  torch::executor::aten::ne_outf(context, a, b, out);
  EXPECT_TENSOR_EQ(out, tf_bool.make({2, 2}, {false, true, false, true}));
}

TEST(OpNeTest, AllDtypesSupported) {
#define TEST_ENTRY(ctype, dtype) test_dtype<ctype, ScalarType::dtype>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

/* %python
import torch
torch.manual_seed(0)
x = torch.randint(3, (3, 2))
res = torch.ne(x, 2)
op = "ne_scalar_out"
opt_setup_params = """
  Scalar other = 2;
"""
opt_extra_params = "other,"
dtype = "ScalarType::Int"
out_dtype = "ScalarType::Bool"
check = "EXPECT_TENSOR_EQ" */

TEST(OpNeScalarOutKernelTest, DynamicShapeUpperBoundSameAsExpected) {
  /* %python
  out_args = "{3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op_out_dtype) */

  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tfOut;

  Tensor x = tf.make({3, 2}, {2, 0, 2, 0, 1, 0});
  Tensor expected = tfOut.make({3, 2}, {false, true, false, true, true, true});

  Scalar other = 2;

  Tensor out =
      tfOut.zeros({3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  ne_scalar_out(x, other, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpNeScalarOutKernelTest, DynamicShapeUpperBoundLargerThanExpected) {
  /* %python
  out_args = "{10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op_out_dtype) */

  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tfOut;

  Tensor x = tf.make({3, 2}, {2, 0, 2, 0, 1, 0});
  Tensor expected = tfOut.make({3, 2}, {false, true, false, true, true, true});

  Scalar other = 2;

  Tensor out = tfOut.zeros(
      {10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  ne_scalar_out(x, other, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpNeScalarOutKernelTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape unbound not supported";
  }
  /* %python
  out_args = "{1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND"
  %rewrite(unary_op_out_dtype) */

  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tfOut;

  Tensor x = tf.make({3, 2}, {2, 0, 2, 0, 1, 0});
  Tensor expected = tfOut.make({3, 2}, {false, true, false, true, true, true});

  Scalar other = 2;

  Tensor out = tfOut.zeros(
      {1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  ne_scalar_out(x, other, out);
  EXPECT_TENSOR_EQ(out, expected);
}
