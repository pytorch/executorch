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

Tensor& _bitwise_not_out(const Tensor& a, Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::bitwise_not_outf(context, a, out);
}

// Common testing for bitwise_not operator
template <ScalarType DTYPE>
void test_bitwise_not_out() {
  TensorFactory<DTYPE> tf;

  const std::vector<int32_t> sizes = {2, 2};

  // Destination for the bitwise_not operator.
  Tensor out = tf.zeros(sizes);

  // Check that it matches the expected output.
  _bitwise_not_out(tf.make(sizes, /*data=*/{0, -1, -2, 3}), out);
  EXPECT_TENSOR_EQ(out, tf.make(sizes, /*data=*/{-1, 0, 1, -4}));
}

template <>
void test_bitwise_not_out<ScalarType::Byte>() {
  TensorFactory<ScalarType::Byte> tf;

  const std::vector<int32_t> sizes = {2, 2};

  // Destination for the bitwise_not operator.
  Tensor out = tf.zeros(sizes);

  // Check that it matches the expected output.
  _bitwise_not_out(tf.make(sizes, /*data=*/{0, 1, 2, 3}), out);
  EXPECT_TENSOR_EQ(out, tf.make(sizes, /*data=*/{255, 254, 253, 252}));
}

template <>
void test_bitwise_not_out<ScalarType::Bool>() {
  TensorFactory<ScalarType::Bool> tf;

  const std::vector<int32_t> sizes = {2, 2};

  // Destination for the bitwise_not operator.
  Tensor out = tf.zeros(sizes);

  // Check that it matches the expected output.
  _bitwise_not_out(tf.make(sizes, /*data=*/{true, false, true, false}), out);
  EXPECT_TENSOR_EQ(out, tf.make(sizes, /*data=*/{false, true, false, true}));
}

TEST(OpBitwiseNotOutKernelTest, AllIntInputOutputSupport) {
#define TEST_ENTRY(ctype, dtype) test_bitwise_not_out<ScalarType::dtype>();
  ET_FORALL_INT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(OpBitwiseNotOutKernelTest, BoolInputOutputSupport) {
  test_bitwise_not_out<ScalarType::Bool>();
}

// Mismatched shape tests.
TEST(OpBitwiseNotOutKernelTest, MismatchedShapesDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched shapes";
  }
  TensorFactory<ScalarType::Int> tf;

  Tensor a = tf.ones(/*sizes=*/{4});
  Tensor out = tf.ones(/*sizes=*/{2, 2});

  ET_EXPECT_KERNEL_FAILURE(_bitwise_not_out(a, out));
}

// Unhandled output dtypes.
template <ScalarType DTYPE>
void test_bitwise_not_invalid_dtype_dies() {
  TensorFactory<DTYPE> tf;

  const std::vector<int32_t> sizes = {2, 5};

  Tensor in = tf.ones(sizes);
  Tensor out = tf.zeros(sizes);

  ET_EXPECT_KERNEL_FAILURE(_bitwise_not_out(in, out));
}

TEST(OpBitwiseNotOutKernelTest, AllFloatInputDTypeDies) {
#define TEST_ENTRY(ctype, dtype) \
  test_bitwise_not_invalid_dtype_dies<ScalarType::dtype>();
  ET_FORALL_FLOAT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

/* %python
import torch
torch.manual_seed(0)
x = torch.randint(10, (3, 2))
res = torch.bitwise_not(x)
op = "_bitwise_not_out"
dtype = "ScalarType::Int"
check = "EXPECT_TENSOR_EQ" */

TEST(OpBitwiseNotOutKernelTest, DynamicShapeUpperBoundSameAsExpected) {
  /* %python
  out_args = "{3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Int> tf;

  Tensor x = tf.make({3, 2}, {4, 9, 3, 0, 3, 9});
  Tensor expected = tf.make({3, 2}, {-5, -10, -4, -1, -4, -10});

  Tensor out =
      tf.zeros({3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  _bitwise_not_out(x, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpBitwiseNotOutKernelTest, DynamicShapeUpperBoundLargerThanExpected) {
  /* %python
  out_args = "{10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Int> tf;

  Tensor x = tf.make({3, 2}, {4, 9, 3, 0, 3, 9});
  Tensor expected = tf.make({3, 2}, {-5, -10, -4, -1, -4, -10});

  Tensor out =
      tf.zeros({10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  _bitwise_not_out(x, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpBitwiseNotOutKernelTest, DynamicShapeUnbound) {
  GTEST_SKIP() << "Dynamic shape unbound not supported";
  /* %python
  out_args = "{1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Int> tf;

  Tensor x = tf.make({3, 2}, {4, 9, 3, 0, 3, 9});
  Tensor expected = tf.make({3, 2}, {-5, -10, -4, -1, -4, -10});

  Tensor out =
      tf.zeros({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  _bitwise_not_out(x, out);
  EXPECT_TENSOR_EQ(out, expected);
}
