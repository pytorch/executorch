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

class OpAcosOutTest : public OperatorTest {
 protected:
  Tensor& op_acos_out(const Tensor& self, Tensor& out) {
    return torch::executor::aten::acos_outf(context_, self, out);
  }

  // Common testing for acos operator and all kinds of supported input types
  template <ScalarType IN_DTYPE, ScalarType OUT_DTYPE>
  void test_floating_point_acos_out(
      const std::vector<int32_t>& out_shape = {1, 6},
      TensorShapeDynamism dynamism = TensorShapeDynamism::STATIC) {
    TensorFactory<IN_DTYPE> tf_in;
    TensorFactory<OUT_DTYPE> tf_out;

    // Destination for the acos operator.
    Tensor out = tf_out.zeros(out_shape, dynamism);

    // clang-format off
    op_acos_out(tf_in.make({1, 6}, { 0, 1, 3, 5, 10, 100 }), out);

    // Check that it matches (or close to) the expected output.
    EXPECT_TENSOR_CLOSE(
        out,
        tf_out.make({1, 6}, { 1.570796, 0.000000, NAN, NAN, NAN, NAN }));
    // clang-format on
  }

  // Unhandled output dtypes.
  template <ScalarType INPUT_DTYPE, ScalarType OUTPUT_DTYPE>
  void test_acos_invalid_output_dtype_dies() {
    TensorFactory<INPUT_DTYPE> tf;
    TensorFactory<OUTPUT_DTYPE> tf_out;

    const std::vector<int32_t> sizes = {2, 5};

    Tensor in = tf.ones(sizes);
    Tensor out = tf_out.zeros(sizes);

    ET_EXPECT_KERNEL_FAILURE(context_, op_acos_out(in, out));
  }
};

TEST_F(OpAcosOutTest, HandleBoolInput) {
  TensorFactory<ScalarType::Bool> tf_bool;
  TensorFactory<ScalarType::Float> tf_float;

  const std::vector<int32_t> sizes = {1, 2};

  Tensor a = tf_bool.make(sizes, /*data=*/{false, true});
  Tensor out = tf_float.zeros(sizes);
  Tensor res = tf_float.make(sizes, /*data=*/{1.570796, 0.000000});

  EXPECT_TENSOR_CLOSE(op_acos_out(a, out), res);
}

TEST_F(OpAcosOutTest, AllRealInputHalfOutputStaticDynamismSupport) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "Test Half support only for ExecuTorch mode";
  }
#define TEST_ENTRY(ctype, dtype) \
  test_floating_point_acos_out<ScalarType::dtype, ScalarType::Half>();
  ET_FORALL_REALH_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpAcosOutTest, AllRealInputFloatOutputStaticDynamismSupport) {
#define TEST_ENTRY(ctype, dtype) \
  test_floating_point_acos_out<ScalarType::dtype, ScalarType::Float>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpAcosOutTest, AllRealInputDoubleOutputStaticDynamismSupport) {
#define TEST_ENTRY(ctype, dtype) \
  test_floating_point_acos_out<ScalarType::dtype, ScalarType::Double>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpAcosOutTest, AllRealInputHalfOutputBoundDynamismSupport) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "Test Half support only for ExecuTorch mode";
  }
#define TEST_ENTRY(ctype, dtype)                                     \
  test_floating_point_acos_out<ScalarType::dtype, ScalarType::Half>( \
      {10, 10}, TensorShapeDynamism::DYNAMIC_BOUND);
  ET_FORALL_REALH_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpAcosOutTest, AllRealInputFloatOutputBoundDynamismSupport) {
#define TEST_ENTRY(ctype, dtype)                                      \
  test_floating_point_acos_out<ScalarType::dtype, ScalarType::Float>( \
      {10, 10}, TensorShapeDynamism::DYNAMIC_BOUND);
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpAcosOutTest, AllRealInputDoubleOutputBoundDynamismSupport) {
#define TEST_ENTRY(ctype, dtype)                                       \
  test_floating_point_acos_out<ScalarType::dtype, ScalarType::Double>( \
      {10, 10}, TensorShapeDynamism::DYNAMIC_BOUND);
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpAcosOutTest, AllRealInputFloatOutputUnboundDynamismSupport) {
  if (!torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "Dynamic shape unbound not supported";
  }
#define TEST_ENTRY(ctype, dtype)                                      \
  test_floating_point_acos_out<ScalarType::dtype, ScalarType::Float>( \
      {1, 1}, TensorShapeDynamism::DYNAMIC_UNBOUND);
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpAcosOutTest, AllRealInputDoubleOutputUnboundDynamismSupport) {
  if (!torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "Dynamic shape unbound not supported";
  }
#define TEST_ENTRY(ctype, dtype)                                       \
  test_floating_point_acos_out<ScalarType::dtype, ScalarType::Double>( \
      {1, 1}, TensorShapeDynamism::DYNAMIC_UNBOUND);
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpAcosOutTest, AllNonFloatOutputDTypeDies) {
#define TEST_ENTRY(ctype, dtype) \
  test_acos_invalid_output_dtype_dies<ScalarType::Float, ScalarType::dtype>();
  ET_FORALL_INT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

// Mismatched shape tests.
TEST_F(OpAcosOutTest, MismatchedInputShapesDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched input shapes";
  }
  TensorFactory<ScalarType::Float> tf;

  Tensor a = tf.ones(/*sizes=*/{4});
  Tensor out = tf.ones(/*sizes=*/{2, 2});

  ET_EXPECT_KERNEL_FAILURE(context_, op_acos_out(a, out));
}
