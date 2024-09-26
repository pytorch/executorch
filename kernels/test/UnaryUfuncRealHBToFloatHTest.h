/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/kernels/test/TestUtil.h>
#include <executorch/kernels/test/supported_features.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

namespace torch::executor::testing {
// Generic test harness for ops that use unary_ufunc_realhb_to_floath
// -- in other words, ops that just apply an elementwise function
// mapping to a float or half.
class UnaryUfuncRealHBToFloatHTest : public OperatorTest {
 protected:
  // Implement this to call the torch::executor::aten::op_outf function for the
  // top.
  virtual exec_aten::Tensor& op_out(
      const exec_aten::Tensor& self,
      exec_aten::Tensor& out) = 0;

  // Scalar reference implementation of the function in question for testing.
  virtual double op_reference(double x) const = 0;

  template <exec_aten::ScalarType IN_DTYPE, exec_aten::ScalarType OUT_DTYPE>
  void test_floating_point_op_out(
      const std::vector<int32_t>& out_shape = {1, 6},
      exec_aten::TensorShapeDynamism dynamism =
          exec_aten::TensorShapeDynamism::STATIC) {
    TensorFactory<IN_DTYPE> tf_in;
    TensorFactory<OUT_DTYPE> tf_out;

    exec_aten::Tensor out = tf_out.zeros(out_shape, dynamism);

    std::vector<typename decltype(tf_in)::ctype> test_vector = {
        0, 1, 3, 5, 10, 100};
    std::vector<typename decltype(tf_out)::ctype> expected_vector;
    std::transform(
        test_vector.begin(),
        test_vector.end(),
        std::back_inserter(expected_vector),
        [this](auto x) { return this->op_reference(x); });

    // clang-format off
    op_out(tf_in.make({1, 6}, test_vector), out);

    EXPECT_TENSOR_CLOSE(
        out,
        tf_out.make({1, 6}, expected_vector));
    // clang-format on
  }
  // Unhandled output dtypes.
  template <
      exec_aten::ScalarType INPUT_DTYPE,
      exec_aten::ScalarType OUTPUT_DTYPE>
  void test_op_invalid_output_dtype_dies() {
    TensorFactory<INPUT_DTYPE> tf;
    TensorFactory<OUTPUT_DTYPE> tf_out;

    const std::vector<int32_t> sizes = {2, 5};

    exec_aten::Tensor in = tf.ones(sizes);
    exec_aten::Tensor out = tf_out.zeros(sizes);

    ET_EXPECT_KERNEL_FAILURE(context_, op_out(in, out));
  }

  void test_bool_input() {
    TensorFactory<exec_aten::ScalarType::Bool> tf_bool;
    TensorFactory<exec_aten::ScalarType::Float> tf_float;

    const std::vector<int32_t> sizes = {1, 2};

    exec_aten::Tensor a = tf_bool.make(sizes, /*data=*/{false, true});
    exec_aten::Tensor out = tf_float.zeros(sizes);
    exec_aten::Tensor res = tf_float.make(
        sizes,
        /*data=*/{(float)op_reference(false), (float)op_reference(true)});

    EXPECT_TENSOR_CLOSE(op_out(a, out), res);
  }

  void test_mismatched_input_shapes_dies() {
    if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
      GTEST_SKIP() << "ATen kernel can handle mismatched input shapes";
    }
    TensorFactory<exec_aten::ScalarType::Float> tf;

    exec_aten::Tensor a = tf.ones(/*sizes=*/{4});
    exec_aten::Tensor out = tf.ones(/*sizes=*/{2, 2});

    ET_EXPECT_KERNEL_FAILURE(context_, op_out(a, out));
  }

  void test_all_real_input_half_output_static_dynamism_support() {
    if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
      GTEST_SKIP() << "Test Half support only for ExecuTorch mode";
    }
#define TEST_ENTRY(ctype, dtype)    \
  test_floating_point_op_out<       \
      exec_aten::ScalarType::dtype, \
      exec_aten::ScalarType::Half>();
    ET_FORALL_REALH_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
  }

  void test_all_real_input_float_output_static_dynamism_support() {
#define TEST_ENTRY(ctype, dtype)    \
  test_floating_point_op_out<       \
      exec_aten::ScalarType::dtype, \
      exec_aten::ScalarType::Float>();
    ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
  }

  void test_all_real_input_double_output_static_dynamism_support() {
#define TEST_ENTRY(ctype, dtype)    \
  test_floating_point_op_out<       \
      exec_aten::ScalarType::dtype, \
      exec_aten::ScalarType::Double>();
    ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
  }

  void test_all_real_input_half_output_bound_dynamism_support() {
    if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
      GTEST_SKIP() << "Test Half support only for ExecuTorch mode";
    }
#define TEST_ENTRY(ctype, dtype)    \
  test_floating_point_op_out<       \
      exec_aten::ScalarType::dtype, \
      exec_aten::ScalarType::Half>( \
      {10, 10}, exec_aten::TensorShapeDynamism::DYNAMIC_BOUND);
    ET_FORALL_REALH_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
  }

  void test_all_real_input_float_output_bound_dynamism_support() {
#define TEST_ENTRY(ctype, dtype)     \
  test_floating_point_op_out<        \
      exec_aten::ScalarType::dtype,  \
      exec_aten::ScalarType::Float>( \
      {10, 10}, exec_aten::TensorShapeDynamism::DYNAMIC_BOUND);
    ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
  }

  void test_all_real_input_double_output_bound_dynamism_support() {
#define TEST_ENTRY(ctype, dtype)      \
  test_floating_point_op_out<         \
      exec_aten::ScalarType::dtype,   \
      exec_aten::ScalarType::Double>( \
      {10, 10}, exec_aten::TensorShapeDynamism::DYNAMIC_BOUND);
    ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
  }

  void test_all_real_input_float_output_unbound_dynamism_support() {
    if (!torch::executor::testing::SupportedFeatures::get()->is_aten) {
      GTEST_SKIP() << "Dynamic shape unbound not supported";
    }
#define TEST_ENTRY(ctype, dtype)     \
  test_floating_point_op_out<        \
      exec_aten::ScalarType::dtype,  \
      exec_aten::ScalarType::Float>( \
      {1, 1}, exec_aten::TensorShapeDynamism::DYNAMIC_UNBOUND);
    ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
  }

  void test_all_real_input_double_output_unbound_dynamism_support() {
    if (!torch::executor::testing::SupportedFeatures::get()->is_aten) {
      GTEST_SKIP() << "Dynamic shape unbound not supported";
    }
#define TEST_ENTRY(ctype, dtype)      \
  test_floating_point_op_out<         \
      exec_aten::ScalarType::dtype,   \
      exec_aten::ScalarType::Double>( \
      {1, 1}, exec_aten::TensorShapeDynamism::DYNAMIC_UNBOUND);
    ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
  }

  void test_non_float_output_dtype_dies() {
#define TEST_ENTRY(ctype, dtype)     \
  test_op_invalid_output_dtype_dies< \
      exec_aten::ScalarType::Float,  \
      exec_aten::ScalarType::dtype>();
    ET_FORALL_INT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
  }
};

#define IMPLEMENT_UNARY_UFUNC_REALHB_TO_FLOATH_TEST(TestName)        \
  TEST_F(TestName, HandleBoolInput) {                                \
    test_bool_input();                                               \
  }                                                                  \
  TEST_F(TestName, AllRealInputHalfOutputStaticDynamismSupport) {    \
    test_all_real_input_half_output_static_dynamism_support();       \
  }                                                                  \
                                                                     \
  TEST_F(TestName, AllRealInputFloatOutputStaticDynamismSupport) {   \
    test_all_real_input_float_output_static_dynamism_support();      \
  }                                                                  \
                                                                     \
  TEST_F(TestName, AllRealInputDoubleOutputStaticDynamismSupport) {  \
    test_all_real_input_double_output_static_dynamism_support();     \
  }                                                                  \
                                                                     \
  TEST_F(TestName, AllRealInputHalfOutputBoundDynamismSupport) {     \
    test_all_real_input_half_output_bound_dynamism_support();        \
  }                                                                  \
                                                                     \
  TEST_F(TestName, AllRealInputFloatOutputBoundDynamismSupport) {    \
    test_all_real_input_float_output_bound_dynamism_support();       \
  }                                                                  \
                                                                     \
  TEST_F(TestName, AllRealInputDoubleOutputBoundDynamismSupport) {   \
    test_all_real_input_double_output_bound_dynamism_support();      \
  }                                                                  \
                                                                     \
  TEST_F(TestName, AllRealInputFloatOutputUnboundDynamismSupport) {  \
    test_all_real_input_float_output_unbound_dynamism_support();     \
  }                                                                  \
                                                                     \
  TEST_F(TestName, AllRealInputDoubleOutputUnboundDynamismSupport) { \
    test_all_real_input_double_output_unbound_dynamism_support();    \
  }                                                                  \
                                                                     \
  TEST_F(TestName, AllNonFloatOutputDTypeDies) {                     \
    test_non_float_output_dtype_dies();                              \
  }                                                                  \
                                                                     \
  TEST_F(TestName, MismatchedInputShapesDies) {                      \
    test_mismatched_input_shapes_dies();                             \
  }

} // namespace torch::executor::testing
