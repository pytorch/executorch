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

class OpFillTest : public OperatorTest {
 protected:
  Tensor&
  op_fill_scalar_out(const Tensor& self, const Scalar& other, Tensor& out) {
    return torch::executor::aten::fill_outf(context_, self, other, out);
  }

  Tensor&
  op_fill_tensor_out(const Tensor& self, const Tensor& other, Tensor& out) {
    return torch::executor::aten::fill_outf(context_, self, other, out);
  }

  template <ScalarType DTYPE>
  void test_fill_scalar_out(std::vector<int32_t>&& sizes) {
    TensorFactory<DTYPE> tf;

    // Before: `out` consists of 0s.
    Tensor self = tf.zeros(sizes);
    Tensor out = tf.zeros(sizes);

    // After: `out` consists of 1s.
    Scalar other = 1;
    if (DTYPE == ScalarType::Bool) {
      other = false;
    }
    op_fill_scalar_out(self, other, out);

    Tensor exp_out = tf.full(sizes, 1);
    if (DTYPE == ScalarType::Bool) {
      exp_out = tf.full(sizes, false);
    }

    // Check `out` matches expected output.
    EXPECT_TENSOR_EQ(out, exp_out);
  }

  template <ScalarType DTYPE>
  void test_fill_tensor_out(std::vector<int32_t>&& sizes) {
    TensorFactory<DTYPE> tf;

    // Before: `out` consists of 0s.
    Tensor self = tf.zeros(sizes);
    Tensor out = tf.zeros(sizes);

    // After: `out` consists of 1s.
    Tensor other = tf.ones({});
    op_fill_tensor_out(self, other, out);

    Tensor exp_out = tf.full(sizes, 1);

    // Check `out` matches expected output.
    EXPECT_TENSOR_EQ(out, exp_out);
  }
};

// A macro for defining tests for both scalar and tensor variants of
// `fill_out`. Here the `self` and `out` tensors will be created according
// to the sizes provided, while the scalar/tensor will be a singleton.
#define TEST_FILL_OUT(FN, DTYPE)    \
  FN<ScalarType::DTYPE>({});        \
  FN<ScalarType::DTYPE>({1});       \
  FN<ScalarType::DTYPE>({1, 1, 1}); \
  FN<ScalarType::DTYPE>({2, 0, 4}); \
  FN<ScalarType::DTYPE>({2, 3, 4});

// Create input support tests for scalar variant.
#define GENERATE_SCALAR_INPUT_SUPPORT_TEST(_, DTYPE) \
  TEST_F(OpFillTest, DTYPE##ScalarInputSupport) {    \
    TEST_FILL_OUT(test_fill_scalar_out, DTYPE);      \
  }

ET_FORALL_REAL_TYPES_AND(Bool, GENERATE_SCALAR_INPUT_SUPPORT_TEST)

// Create input support tests for tensor variant.
#define GENERATE_TENSOR_INPUT_SUPPORT_TEST(_, DTYPE) \
  TEST_F(OpFillTest, DTYPE##TensorInputSupport) {    \
    TEST_FILL_OUT(test_fill_tensor_out, DTYPE);      \
  }

ET_FORALL_REAL_TYPES_AND(Bool, GENERATE_TENSOR_INPUT_SUPPORT_TEST)

TEST_F(OpFillTest, MismatchedOtherPropertiesDies) {
  TensorFactory<ScalarType::Int> tf;

  // `self` and `out` have different shapes but same dtype.
  Tensor self = tf.zeros({1});
  Tensor out = tf.zeros({1});

  // Create `other` tensors with incompatible shapes (`dim()` >=1) and/or
  // elements (`numel()` > 1).

  Tensor other1 = tf.zeros({1});
  EXPECT_EQ(other1.dim(), 1);
  EXPECT_EQ(other1.numel(), 1);

  Tensor other2 = tf.zeros({2});
  EXPECT_EQ(other2.dim(), 1);
  EXPECT_EQ(other2.numel(), 2);

  Tensor other3 = tf.zeros({3, 3});
  EXPECT_EQ(other3.dim(), 2);
  EXPECT_EQ(other3.numel(), 9);

  // Assert `other` tensors with incompatible properties fails.
  ET_EXPECT_KERNEL_FAILURE(context_, op_fill_tensor_out(self, other1, out));
  ET_EXPECT_KERNEL_FAILURE(context_, op_fill_tensor_out(self, other2, out));
  ET_EXPECT_KERNEL_FAILURE(context_, op_fill_tensor_out(self, other3, out));
}

TEST_F(OpFillTest, MismatchedOutputShapesDies) {
  // Skip ATen test since it supports `self` and `out` having different shapes.
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched output shape";
  }

  TensorFactory<ScalarType::Int> tf;

  // `self` and `out` have different shapes but same dtype.
  Tensor self = tf.zeros({1});
  Tensor out = tf.zeros({2, 2});

  // Assert `out` can't be filled due to incompatible shapes.
  ET_EXPECT_KERNEL_FAILURE(context_, op_fill_scalar_out(self, 0, out));
}

TEST_F(OpFillTest, MismatchedOutputDtypeDies) {
  TensorFactory<ScalarType::Byte> tf_byte;
  TensorFactory<ScalarType::Float> tf_float;

  // `self` and `out` have different dtypes but same shape.
  Tensor self = tf_byte.zeros({2, 2});
  Tensor out = tf_float.ones({2, 2});

  // Assert `out` can't be filled due to incompatible dtype.
  ET_EXPECT_KERNEL_FAILURE(context_, op_fill_scalar_out(self, 0.0, out));
}
