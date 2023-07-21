// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

Tensor& _nonzero_out(const Tensor& self, Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::nonzero_outf(context, self, out);
}

template <class CTYPE, exec_aten::ScalarType DTYPE>
void test_dtype() {
  TensorFactory<DTYPE> tf_input;
  TensorFactory<ScalarType::Long> tf_long;
  // clang-format off
  Tensor a = tf_input.make(/*sizes=*/{2, 2}, /*data=*/{2, 0,
                                                       2, 4});
  // clang-format on
  Tensor out = tf_long.zeros({3, 2});

  _nonzero_out(a, out);
  // clang-format off
  EXPECT_TENSOR_EQ(out, tf_long.make({3, 2}, {0, 0,
                                              1, 0,
                                              1, 1}));
  // clang-format on
}

TEST(OpNonzeroTest, AllDtypesSupported) {
#define TEST_ENTRY(ctype, dtype) test_dtype<ctype, ScalarType::dtype>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

#if !defined(USE_ATEN_LIB)
TEST(OpNonzeroTest, StaticShapeInconsistentSize) {
  TensorFactory<ScalarType::Float> tf_input;
  TensorFactory<ScalarType::Long> tf_long;
  // clang-format off
  Tensor a = tf_input.make(/*sizes=*/{2, 2}, /*data=*/{2, 0,
                                                       2, 4});
  // clang-format on
  // If we use static size here (by default), it won't work unless we know the
  // output size
  Tensor out =
      tf_long.zeros({4, 2}, torch::executor::TensorShapeDynamism::STATIC);

  ET_EXPECT_KERNEL_FAILURE(_nonzero_out(a, out));
}

TEST(OpNonzeroTest, DynamicShape) {
  TensorFactory<ScalarType::Float> tf_input;
  TensorFactory<ScalarType::Long> tf_long;
  // clang-format off
  Tensor a = tf_input.make(/*sizes=*/{2, 2}, /*data=*/{2, 0,
                                                       2, 4});
  // clang-format on
  Tensor out = tf_long.zeros(
      {4, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);

  _nonzero_out(a, out);
  // clang-format off
  EXPECT_TENSOR_EQ(out, tf_long.make({3, 2}, {0, 0,
                                              1, 0,
                                              1, 1}));
  // clang-format on
}

TEST(OpNonzeroTest, DynamicShapeInsufficientBuffer) {
  TensorFactory<ScalarType::Float> tf_input;
  TensorFactory<ScalarType::Long> tf_long;
  // clang-format off
  Tensor a = tf_input.make(/*sizes=*/{2, 2}, /*data=*/{2, 0,
                                                       2, 4});
  // clang-format on
  Tensor out = tf_long.zeros(
      {2, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);

  ET_EXPECT_KERNEL_FAILURE(_nonzero_out(a, out));
}
#endif
