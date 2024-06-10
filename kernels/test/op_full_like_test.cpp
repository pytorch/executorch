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
#include <executorch/test/utils/DeathTest.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::MemoryFormat;
using exec_aten::optional;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpFullLikeTest : public OperatorTest {
 protected:
  Tensor& op_full_like_out(
      const Tensor& self,
      const Scalar& fill_value,
      optional<MemoryFormat> memory_format,
      Tensor& out) {
    return torch::executor::aten::full_like_outf(
        context_, self, fill_value, memory_format, out);
  }

  template <ScalarType DTYPE>
  void test_full_like_out() {
    TensorFactory<DTYPE> tf;
    const std::vector<int32_t> sizes = {2, 2};
    Tensor in = tf.zeros(sizes);
    Tensor out = tf.zeros(sizes);
    Scalar value = 42;
    MemoryFormat memory_format = MemoryFormat::Contiguous;

    // Check that it matches the expected output.
    op_full_like_out(in, value, memory_format, out);
    EXPECT_TENSOR_EQ(out, tf.make(sizes, /*data=*/{42, 42, 42, 42}));

    value = 1;
    op_full_like_out(in, value, memory_format, out);
    EXPECT_TENSOR_EQ(out, tf.ones(sizes));
  }

  template <ScalarType DTYPE>
  void test_full_like_out_mismatched_shape() {
    TensorFactory<DTYPE> tf;
    const std::vector<int32_t> sizes = {2, 2};
    Tensor in = tf.zeros(/*sizes=*/{2, 2});
    Tensor out = tf.zeros(/*sizes=*/{4, 2});
    Scalar value = 42;
    MemoryFormat memory_format;

    ET_EXPECT_KERNEL_FAILURE(
        context_, op_full_like_out(in, value, memory_format, out));
  }
};

template <>
void OpFullLikeTest::test_full_like_out<ScalarType::Bool>() {
  TensorFactory<ScalarType::Bool> tf;
  const std::vector<int32_t> sizes = {2, 2};
  Tensor in = tf.zeros(sizes);
  Tensor out = tf.zeros(sizes);
  Scalar value = true;
  MemoryFormat memory_format = MemoryFormat::Contiguous;

  // Check that it matches the expected output.
  op_full_like_out(in, value, memory_format, out);
  EXPECT_TENSOR_EQ(out, tf.make(sizes, /*data=*/{true, true, true, true}));

  value = false;
  op_full_like_out(in, value, memory_format, out);
  EXPECT_TENSOR_EQ(out, tf.zeros(sizes));
}

TEST_F(OpFullLikeTest, AllRealOutputPasses) {
#define TEST_ENTRY(ctype, dtype) test_full_like_out<ScalarType::dtype>();
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpFullLikeTest, MismatchedShapeDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched shapes";
  }
#define TEST_ENTRY(ctype, dtype) \
  test_full_like_out_mismatched_shape<ScalarType::dtype>();
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpFullLikeTest, SimpleGeneratedCase) {
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
      {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
       3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
       3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
       3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
       3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
       3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
       3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
       3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0});

  Tensor out = tf.zeros({10, 10});
  Tensor ret = op_full_like_out(x, Scalar(3.0), MemoryFormat::Contiguous, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpFullLikeTest, DynamicShapeUpperBoundSameAsExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.04876953363418579,
       0.816348671913147,
       0.44230276346206665,
       0.2767965793609619,
       0.8998266458511353,
       0.09595239162445068});
  Tensor expected_result = tf.make({3, 2}, {3.0, 3.0, 3.0, 3.0, 3.0, 3.0});

  Tensor out =
      tf.zeros({3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_full_like_out(x, Scalar(3.0), MemoryFormat::Contiguous, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpFullLikeTest, DynamicShapeUpperBoundLargerThanExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.04876953363418579,
       0.816348671913147,
       0.44230276346206665,
       0.2767965793609619,
       0.8998266458511353,
       0.09595239162445068});
  Tensor expected_result = tf.make({3, 2}, {3.0, 3.0, 3.0, 3.0, 3.0, 3.0});

  Tensor out =
      tf.zeros({10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_full_like_out(x, Scalar(3.0), MemoryFormat::Contiguous, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpFullLikeTest, DynamicShapeUnbound) {
  GTEST_SKIP() << "Dynamic shape unbound not supported";
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.04876953363418579,
       0.816348671913147,
       0.44230276346206665,
       0.2767965793609619,
       0.8998266458511353,
       0.09595239162445068});
  Tensor expected_result = tf.make({3, 2}, {3.0, 3.0, 3.0, 3.0, 3.0, 3.0});

  Tensor out =
      tf.zeros({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  Tensor ret = op_full_like_out(x, Scalar(3.0), MemoryFormat::Contiguous, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpFullLikeTest, HalfSupport) {
  TensorFactory<ScalarType::Half> tf;
  optional<MemoryFormat> memory_format;
  Tensor in = tf.ones({2, 3});
  Tensor out = tf.zeros({2, 3});

  op_full_like_out(in, false, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, tf.full({2, 3}, 0));

  op_full_like_out(in, true, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, tf.full({2, 3}, 1));

  op_full_like_out(in, 7, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, tf.full({2, 3}, 7));

  op_full_like_out(in, 2.5, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, tf.full({2, 3}, 2.5));

  op_full_like_out(in, INFINITY, memory_format, out);
  EXPECT_TENSOR_CLOSE(out, tf.full({2, 3}, INFINITY));
}
