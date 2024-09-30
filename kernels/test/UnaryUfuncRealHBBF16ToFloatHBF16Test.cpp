/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/UnaryUfuncRealHBBF16ToFloatHBF16Test.h>

namespace torch::executor::testing {
void UnaryUfuncRealHBBF16ToFloatHBF16Test::test_bool_input() {
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

void UnaryUfuncRealHBBF16ToFloatHBF16Test::test_mismatched_input_shapes_dies() {
  if (get_supported_features()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched input shapes";
  }
  TensorFactory<exec_aten::ScalarType::Float> tf;

  exec_aten::Tensor a = tf.ones(/*sizes=*/{4});
  exec_aten::Tensor out = tf.ones(/*sizes=*/{2, 2});

  ET_EXPECT_KERNEL_FAILURE(context_, op_out(a, out));
}

void UnaryUfuncRealHBBF16ToFloatHBF16Test::
    test_all_real_input_half_output_static_dynamism_support() {
#define TEST_ENTRY(ctype, dtype)    \
  test_floating_point_op_out<       \
      exec_aten::ScalarType::dtype, \
      exec_aten::ScalarType::Half>();
  ET_FORALL_REALHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

void UnaryUfuncRealHBBF16ToFloatHBF16Test::
    test_all_real_input_bfloat16_output_static_dynamism_support() {
#define TEST_ENTRY(ctype, dtype)    \
  test_floating_point_op_out<       \
      exec_aten::ScalarType::dtype, \
      exec_aten::ScalarType::BFloat16>();
  ET_FORALL_REALHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

void UnaryUfuncRealHBBF16ToFloatHBF16Test::
    test_all_real_input_float_output_static_dynamism_support() {
#define TEST_ENTRY(ctype, dtype)    \
  test_floating_point_op_out<       \
      exec_aten::ScalarType::dtype, \
      exec_aten::ScalarType::Float>();
  ET_FORALL_REALH_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

void UnaryUfuncRealHBBF16ToFloatHBF16Test::
    test_all_real_input_double_output_static_dynamism_support() {
#define TEST_ENTRY(ctype, dtype)    \
  test_floating_point_op_out<       \
      exec_aten::ScalarType::dtype, \
      exec_aten::ScalarType::Double>();
  ET_FORALL_REALH_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

void UnaryUfuncRealHBBF16ToFloatHBF16Test::
    test_all_real_input_half_output_bound_dynamism_support() {
#define TEST_ENTRY(ctype, dtype)    \
  test_floating_point_op_out<       \
      exec_aten::ScalarType::dtype, \
      exec_aten::ScalarType::Half>( \
      {10, 10}, exec_aten::TensorShapeDynamism::DYNAMIC_BOUND);
  ET_FORALL_REALHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

void UnaryUfuncRealHBBF16ToFloatHBF16Test::
    test_all_real_input_bfloat16_output_bound_dynamism_support() {
#define TEST_ENTRY(ctype, dtype)        \
  test_floating_point_op_out<           \
      exec_aten::ScalarType::dtype,     \
      exec_aten::ScalarType::BFloat16>( \
      {10, 10}, exec_aten::TensorShapeDynamism::DYNAMIC_BOUND);
  ET_FORALL_REALHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

void UnaryUfuncRealHBBF16ToFloatHBF16Test::
    test_all_real_input_float_output_bound_dynamism_support() {
#define TEST_ENTRY(ctype, dtype)     \
  test_floating_point_op_out<        \
      exec_aten::ScalarType::dtype,  \
      exec_aten::ScalarType::Float>( \
      {10, 10}, exec_aten::TensorShapeDynamism::DYNAMIC_BOUND);
  ET_FORALL_REALH_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

void UnaryUfuncRealHBBF16ToFloatHBF16Test::
    test_all_real_input_double_output_bound_dynamism_support() {
#define TEST_ENTRY(ctype, dtype)      \
  test_floating_point_op_out<         \
      exec_aten::ScalarType::dtype,   \
      exec_aten::ScalarType::Double>( \
      {10, 10}, exec_aten::TensorShapeDynamism::DYNAMIC_BOUND);
  ET_FORALL_REALH_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

void UnaryUfuncRealHBBF16ToFloatHBF16Test::
    test_all_real_input_float_output_unbound_dynamism_support() {
  if (!get_supported_features()->is_aten) {
    GTEST_SKIP() << "Dynamic shape unbound not supported";
  }
#define TEST_ENTRY(ctype, dtype)     \
  test_floating_point_op_out<        \
      exec_aten::ScalarType::dtype,  \
      exec_aten::ScalarType::Float>( \
      {1, 1}, exec_aten::TensorShapeDynamism::DYNAMIC_UNBOUND);
  ET_FORALL_REALH_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

void UnaryUfuncRealHBBF16ToFloatHBF16Test::
    test_all_real_input_double_output_unbound_dynamism_support() {
  if (!get_supported_features()->is_aten) {
    GTEST_SKIP() << "Dynamic shape unbound not supported";
  }
#define TEST_ENTRY(ctype, dtype)      \
  test_floating_point_op_out<         \
      exec_aten::ScalarType::dtype,   \
      exec_aten::ScalarType::Double>( \
      {1, 1}, exec_aten::TensorShapeDynamism::DYNAMIC_UNBOUND);
  ET_FORALL_REALH_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

void UnaryUfuncRealHBBF16ToFloatHBF16Test::test_non_float_output_dtype_dies() {
#define TEST_ENTRY(ctype, dtype)     \
  test_op_invalid_output_dtype_dies< \
      exec_aten::ScalarType::Float,  \
      exec_aten::ScalarType::dtype>();
  ET_FORALL_INT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

} // namespace torch::executor::testing
