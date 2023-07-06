// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <executorch/core/kernel_types/kernel_types.h>
#include <executorch/core/kernel_types/testing/TensorFactory.h>
#include <executorch/core/kernel_types/testing/TensorUtil.h>
#include <executorch/core/kernel_types/util/ScalarTypeUtil.h>
#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/kernels/test/supported_features.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>
#include <cstdint>
#include <limits>

using namespace ::testing;
using exec_aten::ArrayRef;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;

using torch::executor::testing::TensorFactory;

Tensor& arange_out(const Scalar& end, Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::arange_outf(context, end, out);
}

class OpArangeOutTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    torch::executor::runtime_init();
  }
};

/// A generic smoke test that works for any dtype that supports  zeros().
template <class CTYPE, exec_aten::ScalarType DTYPE>
void test_dtype() {
  TensorFactory<DTYPE> tf;

  Scalar end = Scalar(static_cast<CTYPE>(10));

  Tensor out = tf.zeros({10});

  Tensor ret = arange_out(end, out);

  // Should always return the provided out Tensor.
  EXPECT_TENSOR_EQ(ret, out);

  // Expected tensor, filled with 0, 1, ..., 9
  Tensor expected = tf.make({10}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpArangeOutTest, AllRealDtypesSupported) {
#define TEST_ENTRY(ctype, dtype) test_dtype<ctype, ScalarType::dtype>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpArangeOutTest, BoolDtypeSupported) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel test fails";
  }
  TensorFactory<ScalarType::Bool> tf;

  Scalar end = Scalar(2);

  Tensor out = tf.make({2}, {true, false});

  Tensor ret = arange_out(end, out);

  // Should always return the provided out Tensor.
  EXPECT_TENSOR_EQ(ret, out);

  // Expected tensor, filled with 0, 1, a,k,a false, true
  Tensor expected = tf.make({2}, {false, true});

  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpArangeOutTest, FloatNumberNotEqualIntSupport) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel test fails";
  }
  TensorFactory<ScalarType::Float> tf;

  // end = any floating point number between [a, a+1) where a is an arbitrary
  // integer should have same result as end = a. So here arage(end = 5.5) ==
  // arange(5)
  Scalar end = Scalar(5.5);

  Tensor out = tf.zeros({5});

  Tensor ret = arange_out(end, out);

  // Should always return the provided out Tensor.
  EXPECT_TENSOR_EQ(ret, out);

  // Expected tensor, equal
  Tensor expected = tf.make({5}, {0.0, 1.0, 2.0, 3.0, 4.0});

  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpArangeOutTest, EndOutTypeMismatchDie) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle type mismatch";
  }
  TensorFactory<ScalarType::Float> tf;

  Scalar end = Scalar(5);

  Tensor out = tf.zeros({5});

  // Scalar end and Tensor out type mismatch: one is int another is float.
  ET_EXPECT_KERNEL_FAILURE(arange_out(end, out));
}

TEST_F(OpArangeOutTest, OutDimUnsupportedDie) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched out dim";
  }
  TensorFactory<ScalarType::Float> tf;

  Scalar end = Scalar(5);

  Tensor out = tf.zeros({5, 1});

  // out.dim() should be 1, not 2
  ET_EXPECT_KERNEL_FAILURE(arange_out(end, out));
}

TEST_F(OpArangeOutTest, DynamicShapeUpperBoundSameAsExpected) {
  GTEST_SKIP() << "Dynamic shape is not supported";
  TensorFactory<ScalarType::Float> tf;

  Tensor expected_result = tf.make({5}, {0, 1, 2, 3, 4});

  Tensor out =
      tf.zeros({5, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = arange_out(Scalar(5), out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpArangeOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  GTEST_SKIP() << "Dynamic shape is not supported";
  TensorFactory<ScalarType::Float> tf;

  Tensor expected_result = tf.make({5}, {0, 1, 2, 3, 4});

  Tensor out =
      tf.zeros({10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = arange_out(Scalar(5), out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpArangeOutTest, DynamicShapeUnbound) {
  GTEST_SKIP() << "Dynamic shape is not supported";
  TensorFactory<ScalarType::Float> tf;

  Tensor expected_result = tf.make({5}, {0, 1, 2, 3, 4});

  Tensor out =
      tf.zeros({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  Tensor ret = arange_out(Scalar(5), out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}
