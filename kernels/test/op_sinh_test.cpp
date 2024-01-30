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
using exec_aten::ScalarType;
using exec_aten::Tensor;
using exec_aten::TensorShapeDynamism;
using torch::executor::testing::TensorFactory;

Tensor& op_sinh_out(const Tensor& self, Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::sinh_outf(context, self, out);
}

TEST(OpSinhOutKernelTest, HandleBoolInput) {
  TensorFactory<ScalarType::Bool> tf_bool;
  TensorFactory<ScalarType::Float> tf_float;

  const std::vector<int32_t> sizes = {1, 2};

  Tensor a = tf_bool.make(sizes, /*data=*/{false, true});
  Tensor out = tf_float.zeros(sizes);
  Tensor res = tf_float.make(sizes, /*data=*/{0.000000, 1.175201});

  EXPECT_TENSOR_CLOSE(op_sinh_out(a, out), res);
}

// Common testing for sinh operator and all kinds of supported input types
template <ScalarType IN_DTYPE, ScalarType OUT_DTYPE>
void test_floating_point_sinh_out(
    const std::vector<int32_t>& out_shape = {1, 5},
    TensorShapeDynamism dynamism = TensorShapeDynamism::STATIC) {
  TensorFactory<IN_DTYPE> tf_in;
  TensorFactory<OUT_DTYPE> tf_out;

  // Destination for the sinh operator.
  Tensor out = tf_out.zeros(out_shape, dynamism);

  // clang-format off
  op_sinh_out(tf_in.make({1, 5}, { 0, 1, 3, 5, 10 }), out);

  // Check that it matches (or close to) the expected output.
  EXPECT_TENSOR_CLOSE(
      out,
      tf_out.make({1, 5}, { 0.000000e+00, 1.175201e+00, 1.001787e+01, 7.420321e+01, 1.101323e+04 }));
  // clang-format on
}

TEST(OpSinhOutKernelTest, AllRealInputHalfOutputStaticDynamismSupport) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "Test Half support only for ExecuTorch mode";
  }
#define TEST_ENTRY(ctype, dtype) \
  test_floating_point_sinh_out<ScalarType::dtype, ScalarType::Half>();
  ET_FORALL_REALH_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(OpSinhOutKernelTest, AllRealInputFloatOutputStaticDynamismSupport) {
#define TEST_ENTRY(ctype, dtype) \
  test_floating_point_sinh_out<ScalarType::dtype, ScalarType::Float>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(OpSinhOutKernelTest, AllRealInputDoubleOutputStaticDynamismSupport) {
#define TEST_ENTRY(ctype, dtype) \
  test_floating_point_sinh_out<ScalarType::dtype, ScalarType::Double>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(OpSinhOutKernelTest, AllRealInputHalfOutputBoundDynamismSupport) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "Test Half support only for ExecuTorch mode";
  }
#define TEST_ENTRY(ctype, dtype)                                     \
  test_floating_point_sinh_out<ScalarType::dtype, ScalarType::Half>( \
      {10, 10}, TensorShapeDynamism::DYNAMIC_BOUND);
  ET_FORALL_REALH_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(OpSinhOutKernelTest, AllRealInputFloatOutputBoundDynamismSupport) {
#define TEST_ENTRY(ctype, dtype)                                      \
  test_floating_point_sinh_out<ScalarType::dtype, ScalarType::Float>( \
      {10, 10}, TensorShapeDynamism::DYNAMIC_BOUND);
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(OpSinhOutKernelTest, AllRealInputDoubleOutputBoundDynamismSupport) {
#define TEST_ENTRY(ctype, dtype)                                       \
  test_floating_point_sinh_out<ScalarType::dtype, ScalarType::Double>( \
      {10, 10}, TensorShapeDynamism::DYNAMIC_BOUND);
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(OpSinhOutKernelTest, AllRealInputFloatOutputUnboundDynamismSupport) {
  if (!torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "Dynamic shape unbound not supported";
  }
#define TEST_ENTRY(ctype, dtype)                                      \
  test_floating_point_sinh_out<ScalarType::dtype, ScalarType::Float>( \
      {1, 1}, TensorShapeDynamism::DYNAMIC_UNBOUND);
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(OpSinhOutKernelTest, AllRealInputDoubleOutputUnboundDynamismSupport) {
  if (!torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "Dynamic shape unbound not supported";
  }
#define TEST_ENTRY(ctype, dtype)                                       \
  test_floating_point_sinh_out<ScalarType::dtype, ScalarType::Double>( \
      {1, 1}, TensorShapeDynamism::DYNAMIC_UNBOUND);
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

// Unhandled output dtypes.
template <ScalarType INPUT_DTYPE, ScalarType OUTPUT_DTYPE>
void test_sinh_invalid_output_dtype_dies() {
  TensorFactory<INPUT_DTYPE> tf;
  TensorFactory<OUTPUT_DTYPE> tf_out;

  const std::vector<int32_t> sizes = {2, 5};

  Tensor in = tf.ones(sizes);
  Tensor out = tf_out.zeros(sizes);

  ET_EXPECT_KERNEL_FAILURE(op_sinh_out(in, out));
}

TEST(OpSinhOutKernelTest, AllNonFloatOutputDTypeDies) {
#define TEST_ENTRY(ctype, dtype) \
  test_sinh_invalid_output_dtype_dies<ScalarType::Float, ScalarType::dtype>();
  ET_FORALL_INT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

// Mismatched shape tests.
TEST(OpSinhOutKernelTest, MismatchedInputShapesDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched input shapes";
  }
  TensorFactory<ScalarType::Float> tf;

  Tensor a = tf.ones(/*sizes=*/{4});
  Tensor out = tf.ones(/*sizes=*/{2, 2});

  ET_EXPECT_KERNEL_FAILURE(op_sinh_out(a, out));
}
