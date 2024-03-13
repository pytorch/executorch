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
using exec_aten::MemoryFormat;
using exec_aten::optional;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpCloneTest : public OperatorTest {
 protected:
  Tensor& op_clone_out(
      const Tensor& self,
      optional<MemoryFormat> memory_format,
      Tensor& out) {
    return torch::executor::aten::clone_outf(
        context_, self, memory_format, out);
  }

  // test if clone.out works well under all kinds of legal input type.
  template <class CTYPE, exec_aten::ScalarType DTYPE>
  void test_dtype() {
    TensorFactory<DTYPE> tf;
    Tensor input = tf.make(/*sizes=*/{2, 4}, /*data=*/{2, 3, 2, 4, 1, 5, 1, 6});
    Tensor out_nullopt = tf.zeros(/*sizes=*/{2, 4});
    Tensor out_contiguous = tf.zeros(/*sizes=*/{2, 4});

    // we only support contiguous memory, the memory type shall be either
    // nullopt or MemoryFormat::Contiguous.
    Tensor out_nullopt_ret = op_clone_out(
        /*self=*/input,
        /*memory_format=*/exec_aten::nullopt,
        /*out=*/out_nullopt);
    Tensor out_contiguous_ret = op_clone_out(
        /*self=*/input,
        /*memory_format=*/exec_aten::MemoryFormat::Contiguous,
        /*out=*/out_contiguous);

    // The original tensor a should share same value with the out variable and
    // return variable of clone function
    EXPECT_TENSOR_EQ(input, out_nullopt);
    EXPECT_TENSOR_EQ(input, out_nullopt_ret);

    EXPECT_TENSOR_EQ(input, out_contiguous);
    EXPECT_TENSOR_EQ(input, out_contiguous_ret);
  }

  template <class CTYPE, ScalarType DTYPE>
  void test_empty_input() {
    TensorFactory<DTYPE> tf;
    Tensor input = tf.make(/*sizes=*/{3, 0, 1, 2}, /*data=*/{});
    Tensor out = tf.zeros({3, 0, 1, 2});
    op_clone_out(input, /*memory_format=*/exec_aten::nullopt, out);
    // check a and out share same value, but are different object
    EXPECT_TENSOR_EQ(input, out);
  }
};

// regular test for clone.out
TEST_F(OpCloneTest, AllDtypesSupported) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel test fails";
  }
#define TEST_ENTRY(ctype, dtype) test_dtype<ctype, ScalarType::dtype>();
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpCloneTest, EmptyInputSupported) {
#define TEST_ENTRY(ctype, dtype) test_empty_input<ctype, ScalarType::dtype>();
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpCloneTest, MismatchedSizesDie) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched sizes";
  }
  TensorFactory<ScalarType::Int> tf;
  Tensor input = tf.make(/*sizes=*/{3, 1, 1, 2}, /*data=*/{1, 2, 3, 4, 5, 6});
  Tensor out = tf.zeros({3, 2, 1, 1});
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_clone_out(input, /*memory_format=*/exec_aten::nullopt, out));
}

TEST_F(OpCloneTest, MismatchedTypesDie) {
  TensorFactory<ScalarType::Int> tf_in;
  TensorFactory<ScalarType::Float> tf_out;
  Tensor input =
      tf_in.make(/*sizes=*/{3, 1, 1, 2}, /*data=*/{1, 2, 3, 4, 5, 6});
  Tensor out = tf_out.zeros({3, 1, 1, 2});
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_clone_out(input, /*memory_format=*/exec_aten::nullopt, out));
}

// Only contiguous memory is supported, the memory type other than nullopt or
// MemoryFormat::Contiguous should not be allowed. The function is expected
// depth if using the illegal memory format.
TEST_F(OpCloneTest, MismatchedMemoryFormatDie) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle non contiguous memory formats";
  }
  TensorFactory<ScalarType::Float> tf_in;
  TensorFactory<ScalarType::Float> tf_out;
  Tensor input =
      tf_in.make(/*sizes=*/{3, 1, 1, 2}, /*data=*/{1, 2, 3, 4, 5, 6});
  Tensor out = tf_out.zeros({3, 1, 1, 2});
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_clone_out(input, static_cast<exec_aten::MemoryFormat>(55), out));
}

TEST_F(OpCloneTest, SimpleGeneratedCase) {
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
  Tensor ret = op_clone_out(x, exec_aten::MemoryFormat::Contiguous, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpCloneTest, DynamicShapeUpperBoundSameAsExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.04876953363418579,
       0.816348671913147,
       0.44230276346206665,
       0.2767965793609619,
       0.8998266458511353,
       0.09595239162445068});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.04876953363418579,
       0.816348671913147,
       0.44230276346206665,
       0.2767965793609619,
       0.8998266458511353,
       0.09595239162445068});

  Tensor out =
      tf.zeros({3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_clone_out(x, exec_aten::MemoryFormat::Contiguous, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpCloneTest, DynamicShapeUpperBoundLargerThanExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.04876953363418579,
       0.816348671913147,
       0.44230276346206665,
       0.2767965793609619,
       0.8998266458511353,
       0.09595239162445068});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.04876953363418579,
       0.816348671913147,
       0.44230276346206665,
       0.2767965793609619,
       0.8998266458511353,
       0.09595239162445068});

  Tensor out =
      tf.zeros({10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_clone_out(x, exec_aten::MemoryFormat::Contiguous, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpCloneTest, DynamicShapeUnbound) {
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
  Tensor expected_result = tf.make(
      {3, 2},
      {0.04876953363418579,
       0.816348671913147,
       0.44230276346206665,
       0.2767965793609619,
       0.8998266458511353,
       0.09595239162445068});

  Tensor out =
      tf.zeros({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  Tensor ret = op_clone_out(x, exec_aten::MemoryFormat::Contiguous, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}
