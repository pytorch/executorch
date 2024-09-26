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
class UnaryUfuncRealHBBF16ToFloatHBF16Test : public OperatorTest {
 protected:
  // Implement this to call the torch::executor::aten::op_outf function for the
  // op.
  virtual exec_aten::Tensor& op_out(
      const exec_aten::Tensor& self,
      exec_aten::Tensor& out) = 0;

  // Scalar reference implementation of the function in question for testing.
  virtual double op_reference(double x) const = 0;

  // The SupportedFeatures system assumes that it can build each test
  // target with a separate SupportedFeatures (really just one
  // portable, one optimzed but between one and the infinite, two is
  // ridiculous and can't exist). We work around that by calling
  // SupportedFeatures::get() in the concrete test translation
  // unit. You need to declare an override, but we implement it for you
  // in IMPLEMENT_UNARY_UFUNC_REALHB_TO_FLOATH_TEST.
  virtual SupportedFeatures* get_supported_features() const = 0;

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

  void test_bool_input();

  void test_mismatched_input_shapes_dies();

  void test_all_real_input_half_output_static_dynamism_support();

  void test_all_real_input_bfloat16_output_static_dynamism_support();

  void test_all_real_input_float_output_static_dynamism_support();

  void test_all_real_input_double_output_static_dynamism_support();

  void test_all_real_input_half_output_bound_dynamism_support();

  void test_all_real_input_bfloat16_output_bound_dynamism_support();

  void test_all_real_input_float_output_bound_dynamism_support();

  void test_all_real_input_double_output_bound_dynamism_support();

  void test_all_real_input_float_output_unbound_dynamism_support();

  void test_all_real_input_double_output_unbound_dynamism_support();

  void test_non_float_output_dtype_dies();
};

#define IMPLEMENT_UNARY_UFUNC_REALHB_TO_FLOATH_TEST(TestName)         \
  torch::executor::testing::SupportedFeatures*                        \
  TestName::get_supported_features() const {                          \
    return torch::executor::testing::SupportedFeatures::get();        \
  }                                                                   \
  TEST_F(TestName, HandleBoolInput) {                                 \
    test_bool_input();                                                \
  }                                                                   \
  TEST_F(TestName, AllRealInputHalfOutputStaticDynamismSupport) {     \
    test_all_real_input_half_output_static_dynamism_support();        \
  }                                                                   \
                                                                      \
  TEST_F(TestName, AllRealInputBFloat16OutputStaticDynamismSupport) { \
    test_all_real_input_bfloat16_output_static_dynamism_support();    \
  }                                                                   \
                                                                      \
  TEST_F(TestName, AllRealInputFloatOutputStaticDynamismSupport) {    \
    test_all_real_input_float_output_static_dynamism_support();       \
  }                                                                   \
                                                                      \
  TEST_F(TestName, AllRealInputDoubleOutputStaticDynamismSupport) {   \
    test_all_real_input_double_output_static_dynamism_support();      \
  }                                                                   \
                                                                      \
  TEST_F(TestName, AllRealInputBFloat16OutputBoundDynamismSupport) {  \
    test_all_real_input_bfloat16_output_bound_dynamism_support();     \
  }                                                                   \
                                                                      \
  TEST_F(TestName, AllRealInputFloatOutputBoundDynamismSupport) {     \
    test_all_real_input_float_output_bound_dynamism_support();        \
  }                                                                   \
                                                                      \
  TEST_F(TestName, AllRealInputDoubleOutputBoundDynamismSupport) {    \
    test_all_real_input_double_output_bound_dynamism_support();       \
  }                                                                   \
                                                                      \
  TEST_F(TestName, AllRealInputFloatOutputUnboundDynamismSupport) {   \
    test_all_real_input_float_output_unbound_dynamism_support();      \
  }                                                                   \
                                                                      \
  TEST_F(TestName, AllRealInputDoubleOutputUnboundDynamismSupport) {  \
    test_all_real_input_double_output_unbound_dynamism_support();     \
  }                                                                   \
                                                                      \
  TEST_F(TestName, AllNonFloatOutputDTypeDies) {                      \
    test_non_float_output_dtype_dies();                               \
  }                                                                   \
                                                                      \
  TEST_F(TestName, MismatchedInputShapesDies) {                       \
    test_mismatched_input_shapes_dies();                              \
  }

} // namespace torch::executor::testing
