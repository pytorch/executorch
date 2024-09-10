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
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::IntArrayRef;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpScalarTensorOutTest : public OperatorTest {
 protected:
  Tensor& op_scalar_tensor_out(const Scalar& s, Tensor& out) {
    return torch::executor::aten::scalar_tensor_outf(context_, s, out);
  }

  template <typename CTYPE, ScalarType DTYPE>
  void test_scalar_tensor_out_0d(CTYPE value) {
    TensorFactory<DTYPE> tf;

    std::vector<int32_t> sizes{};
    Tensor expected = tf.make(sizes, /*data=*/{value});

    Tensor out = tf.ones(sizes);
    op_scalar_tensor_out(value, out);

    EXPECT_TENSOR_EQ(out, expected);
  }

  template <typename CTYPE, ScalarType DTYPE>
  void test_scalar_tensor_out_1d(CTYPE value) {
    TensorFactory<DTYPE> tf;

    std::vector<int32_t> sizes{1};
    Tensor out = tf.ones(sizes);

    ET_EXPECT_KERNEL_FAILURE(context_, op_scalar_tensor_out(value, out));
  }

  template <typename CTYPE, ScalarType DTYPE>
  void test_scalar_tensor_out_2d(CTYPE value) {
    TensorFactory<DTYPE> tf;

    std::vector<int32_t> sizes{1, 1};
    Tensor out = tf.ones(sizes);

    ET_EXPECT_KERNEL_FAILURE(context_, op_scalar_tensor_out(value, out));
  }

  template <typename CTYPE, ScalarType DTYPE>
  void test_scalar_tensor_out_3d(CTYPE value) {
    TensorFactory<DTYPE> tf;

    std::vector<int32_t> sizes{1, 1, 1};
    Tensor out = tf.ones(sizes);

    ET_EXPECT_KERNEL_FAILURE(context_, op_scalar_tensor_out(value, out));
  }
};

#define GENERATE_TEST_0D(ctype, dtype)                      \
  TEST_F(OpScalarTensorOutTest, dtype##TensorsDim0) {       \
    test_scalar_tensor_out_0d<ctype, ScalarType::dtype>(4); \
    test_scalar_tensor_out_0d<ctype, ScalarType::dtype>(8); \
    test_scalar_tensor_out_0d<ctype, ScalarType::dtype>(9); \
  }

ET_FORALL_REAL_TYPES_AND3(Half, Bool, BFloat16, GENERATE_TEST_0D)

#define GENERATE_TEST(ctype, dtype)                                    \
  TEST_F(OpScalarTensorOutTest, dtype##Tensors) {                      \
    if (torch::executor::testing::SupportedFeatures::get()->is_aten) { \
      GTEST_SKIP() << "ATen kernel resizes output to shape {}";        \
    }                                                                  \
    test_scalar_tensor_out_1d<ctype, ScalarType::dtype>(2);            \
    test_scalar_tensor_out_2d<ctype, ScalarType::dtype>(2);            \
    test_scalar_tensor_out_3d<ctype, ScalarType::dtype>(2);            \
    test_scalar_tensor_out_1d<ctype, ScalarType::dtype>(4);            \
    test_scalar_tensor_out_2d<ctype, ScalarType::dtype>(4);            \
    test_scalar_tensor_out_3d<ctype, ScalarType::dtype>(4);            \
    test_scalar_tensor_out_1d<ctype, ScalarType::dtype>(7);            \
    test_scalar_tensor_out_2d<ctype, ScalarType::dtype>(7);            \
    test_scalar_tensor_out_3d<ctype, ScalarType::dtype>(7);            \
  }

ET_FORALL_REAL_TYPES_AND3(Half, Bool, BFloat16, GENERATE_TEST)

TEST_F(OpScalarTensorOutTest, InvalidOutShapeFails) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel will reshape output";
  }

  TensorFactory<ScalarType::Int> tf;
  std::vector<int32_t> sizes{1, 2, 1};

  Tensor out = tf.ones(sizes);
  ET_EXPECT_KERNEL_FAILURE(context_, op_scalar_tensor_out(7, out));
}

TEST_F(OpScalarTensorOutTest, HalfSupport) {
  TensorFactory<ScalarType::Half> tf;
  Tensor out = tf.zeros({});

  op_scalar_tensor_out(false, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({}, {0}));

  op_scalar_tensor_out(true, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({}, {1}));

  op_scalar_tensor_out(7, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({}, {7}));

  op_scalar_tensor_out(2.5, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({}, {2.5}));

  op_scalar_tensor_out(INFINITY, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({}, {INFINITY}));
}
