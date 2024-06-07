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
#include <cmath>

using namespace ::testing;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpDetachCopyOutTest : public OperatorTest {
 protected:
  Tensor& op_detach_copy_out(const Tensor& self, Tensor& out) {
    return torch::executor::aten::detach_copy_outf(context_, self, out);
  }

  // Common testing for eq operator
  template <ScalarType DTYPE>
  void test_detach_copy_out() {
    TensorFactory<DTYPE> tf;
    const std::vector<int32_t> sizes = {2, 2};

    Tensor in = tf.make(sizes, {1, 2, 3, 4});
    Tensor out = tf.zeros(sizes);

    // Valid input should give the expected output
    op_detach_copy_out(in, out);
    EXPECT_TENSOR_EQ(out, tf.make(sizes, {1, 2, 3, 4}));
  }

  template <ScalarType DTYPE>
  void test_detach_copy_out_invalid_shape() {
    TensorFactory<DTYPE> tf;

    const std::vector<int32_t> in_sizes = {2, 2};
    const std::vector<int32_t> out_sizes = {4};

    Tensor in = tf.ones(in_sizes);
    Tensor out = tf.zeros(out_sizes);

    ET_EXPECT_KERNEL_FAILURE(context_, op_detach_copy_out(in, out));
  }
};

template <>
void OpDetachCopyOutTest::test_detach_copy_out<ScalarType::Bool>() {
  TensorFactory<ScalarType::Bool> tf;
  const std::vector<int32_t> sizes = {2, 2};
  Tensor out = tf.zeros(sizes);

  // Valid input should give the expected output
  op_detach_copy_out(tf.make(sizes, /*data=*/{true, false, true, false}), out);
  EXPECT_TENSOR_EQ(out, tf.make(sizes, /*data=*/{true, false, true, false}));
}

template <>
void OpDetachCopyOutTest::test_detach_copy_out<ScalarType::Float>() {
  TensorFactory<ScalarType::Float> tf;
  const std::vector<int32_t> sizes = {2, 2};
  Tensor out = tf.zeros(sizes);

  // Valid input should give the expected output
  op_detach_copy_out(
      tf.make(sizes, /*data=*/{3.14, INFINITY, -INFINITY, NAN}), out);
  EXPECT_TENSOR_EQ(
      out, tf.make(sizes, /*data=*/{3.14, INFINITY, -INFINITY, NAN}));
}

TEST_F(OpDetachCopyOutTest, AllScalarInputOutputSupport) {
#define TEST_ENTRY(ctype, dtype) test_detach_copy_out<ScalarType::dtype>();
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
}

// Mismatched shape tests.
TEST_F(OpDetachCopyOutTest, MismatchedShapesDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched shapes";
  }
#define TEST_ENTRY(ctype, dtype) \
  test_detach_copy_out_invalid_shape<ScalarType::dtype>();
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpDetachCopyOutTest, MismatchedInputDtypesDies) {
  TensorFactory<ScalarType::Byte> tf_byte;
  TensorFactory<ScalarType::Char> tf_char;

  const std::vector<int32_t> sizes = {2, 2};

  Tensor in = tf_byte.ones(sizes);
  Tensor out = tf_char.ones(sizes);

  ET_EXPECT_KERNEL_FAILURE(context_, op_detach_copy_out(in, out));
}

TEST_F(OpDetachCopyOutTest, SimpleGeneratedCase) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {10, 10},
      {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
  Tensor expected_result = tf.make(
      {10, 10},
      {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});

  Tensor out = tf.zeros({10, 10});
  Tensor ret = op_detach_copy_out(x, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpDetachCopyOutTest, DynamicShapeUpperBoundSameAsExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.18719732761383057,
       0.03402292728424072,
       0.944246232509613,
       0.8801798820495605,
       0.0012360215187072754,
       0.5935860276222229});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.18719732761383057,
       0.03402292728424072,
       0.944246232509613,
       0.8801798820495605,
       0.0012360215187072754,
       0.5935860276222229});

  Tensor out =
      tf.zeros({3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_detach_copy_out(x, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpDetachCopyOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.18719732761383057,
       0.03402292728424072,
       0.944246232509613,
       0.8801798820495605,
       0.0012360215187072754,
       0.5935860276222229});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.18719732761383057,
       0.03402292728424072,
       0.944246232509613,
       0.8801798820495605,
       0.0012360215187072754,
       0.5935860276222229});

  Tensor out =
      tf.zeros({10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_detach_copy_out(x, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpDetachCopyOutTest, DynamicShapeUnbound) {
  GTEST_SKIP() << "Dynamic shape unbound not supported";
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.18719732761383057,
       0.03402292728424072,
       0.944246232509613,
       0.8801798820495605,
       0.0012360215187072754,
       0.5935860276222229});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.18719732761383057,
       0.03402292728424072,
       0.944246232509613,
       0.8801798820495605,
       0.0012360215187072754,
       0.5935860276222229});

  Tensor out =
      tf.zeros({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  Tensor ret = op_detach_copy_out(x, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}
