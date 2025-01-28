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
using executorch::runtime::KernelRuntimeContext;
using torch::executor::testing::TensorFactory;

class OpLeScalarOutTest : public OperatorTest {
 protected:
  Tensor& op_le_scalar_out(const Tensor& self, Scalar& other, Tensor& out) {
    return torch::executor::aten::le_outf(context_, self, other, out);
  }

  template <ScalarType DTYPE_IN, ScalarType DTYPE_OUT>
  void test_le_scalar_out() {
    TensorFactory<DTYPE_IN> tf;
    TensorFactory<DTYPE_OUT> tf_out;

    const std::vector<int32_t> sizes = {2, 2};
    Tensor out = tf_out.ones(sizes);
    Scalar other = 2;

    // Valid input should give the expected output
    op_le_scalar_out(tf.make(sizes, /*data=*/{3, 1, 2, 4}), other, out);
    EXPECT_TENSOR_EQ(
        out, tf_out.make(sizes, /*data=*/{false, true, true, false}));
  }
};

class OpLeTensorOutTest : public OperatorTest {
 protected:
  Tensor&
  op_le_tensor_out(const Tensor& self, const Tensor& other, Tensor& out) {
    return torch::executor::aten::le_outf(context_, self, other, out);
  }

  template <ScalarType DTYPE_IN, ScalarType DTYPE_OUT>
  void test_dtype() {
    TensorFactory<DTYPE_IN> tf_input;
    TensorFactory<DTYPE_OUT> tf_out;
    Tensor a = tf_input.make(/*sizes=*/{2, 2}, /*data=*/{2, 3, 2, 4});
    Tensor b = tf_input.make({2, 2}, {1, 4, 2, 3});
    Tensor out = tf_out.zeros({2, 2});

    op_le_tensor_out(a, b, out);
    EXPECT_TENSOR_EQ(out, tf_out.make({2, 2}, {false, true, true, false}));
  }
};

TEST_F(OpLeScalarOutTest, AllRealInputBoolOutputSupport) {
#define TEST_ENTRY(ctype_in, dtype_in, ctype_out, dtype_out) \
  test_le_scalar_out<ScalarType::dtype_in, ScalarType::dtype_out>();

#define TEST_FORALL_OUT_TYPES(ctype_in, dtype_in)            \
  ET_FORALL_REAL_TYPES_WITH2(ctype_in, dtype_in, TEST_ENTRY) \
  test_le_scalar_out<ScalarType::dtype_in, ScalarType::Bool>();

  ET_FORALL_REAL_TYPES(TEST_FORALL_OUT_TYPES)

#undef TEST_FORALL_OUT_TYPES
#undef TEST_ENTRY
}

TEST_F(OpLeScalarOutTest, BoolInputDtype) {
  TensorFactory<ScalarType::Bool> tf_bool;

  const std::vector<int32_t> sizes = {2, 2};
  Tensor a = tf_bool.make(sizes, /*data=*/{false, true, false, true});
  Tensor out = tf_bool.zeros(sizes);
  Scalar other = 0.5;

  op_le_scalar_out(a, other, out);
  EXPECT_TENSOR_EQ(
      out, tf_bool.make(sizes, /*data=*/{true, false, true, false}));
}

// Mismatched shape tests.
TEST_F(OpLeScalarOutTest, MismatchedInOutShapesDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched shapes";
  }
  TensorFactory<ScalarType::Int> tf_int;
  TensorFactory<ScalarType::Bool> tf_bool;

  Tensor a = tf_int.ones(/*sizes=*/{4});
  Tensor out = tf_bool.ones(/*sizes=*/{2, 2});
  Scalar other = 3;

  ET_EXPECT_KERNEL_FAILURE(context_, op_le_scalar_out(a, other, out));
}

TEST_F(OpLeScalarOutTest, DynamicOutShapeTest) {
  TensorFactory<ScalarType::Int> tf;

  const std::vector<int32_t> sizes = {2, 2};
  const std::vector<int32_t> out_sizes = {4, 1};

  Tensor out =
      tf.zeros(out_sizes, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Scalar other = 2;

  // Valid input should give the expected output
  op_le_scalar_out(tf.make(sizes, /*data=*/{3, 1, 2, 4}), other, out);
  EXPECT_TENSOR_EQ(out, tf.make(sizes, /*data=*/{false, true, true, false}));
}

TEST_F(OpLeTensorOutTest, AllDtypesSupported) {
#define TEST_ENTRY(ctype_in, dtype_in, ctype_out, dtype_out) \
  test_dtype<ScalarType::dtype_in, ScalarType::dtype_out>();

#define TEST_FORALL_OUT_TYPES(ctype_in, dtype_in)            \
  ET_FORALL_REAL_TYPES_WITH2(ctype_in, dtype_in, TEST_ENTRY) \
  test_dtype<ScalarType::dtype_in, ScalarType::Bool>();

  ET_FORALL_REAL_TYPES(TEST_FORALL_OUT_TYPES);

#undef TEST_FORALL_OUT_TYPES
#undef TEST_ENTRY
}

TEST_F(OpLeTensorOutTest, MismatchedInShapesDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched shapes";
  }
  TensorFactory<ScalarType::Int> tf_int;
  TensorFactory<ScalarType::Bool> tf_bool;

  Tensor a = tf_int.ones(/*sizes=*/{4});
  Tensor b = tf_int.ones(/*sizes=*/{2, 2});
  Tensor out = tf_bool.ones(/*sizes=*/{4});

  ET_EXPECT_KERNEL_FAILURE(context_, op_le_tensor_out(a, b, out));
}

TEST_F(OpLeTensorOutTest, MismatchedInOutShapesDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched shapes";
  }
  TensorFactory<ScalarType::Int> tf_int;
  TensorFactory<ScalarType::Bool> tf_bool;

  Tensor a = tf_int.ones(/*sizes=*/{4});
  Tensor b = tf_int.ones(/*sizes=*/{4});
  Tensor out = tf_bool.ones(/*sizes=*/{2, 2});

  ET_EXPECT_KERNEL_FAILURE(context_, op_le_tensor_out(a, b, out));
}

TEST_F(OpLeTensorOutTest, DynamicOutShapeTest) {
  TensorFactory<ScalarType::Int> tf;

  Tensor a = tf.make(/*sizes=*/{2, 2}, /*data=*/{2, 3, 2, 4});
  Tensor b = tf.make({2, 2}, {1, 4, 2, 3});

  Tensor out =
      tf.zeros({1, 4}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);

  op_le_tensor_out(a, b, out);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {false, true, true, false}));
}
