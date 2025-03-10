/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

#include <executorch/kernels/test/TestUtil.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpUnfoldTest : public OperatorTest {
 protected:
  Tensor& op_unfold_copy_out(
      const Tensor& self,
      int64_t dim,
      int64_t size,
      int64_t step,
      Tensor& out) {
    return torch::executor::aten::unfold_copy_outf(
        context_, self, dim, size, step, out);
  }

  template <class CTYPE, ScalarType DTYPE>
  void test_unfold_copy_dtype() {
    TensorFactory<DTYPE> tf;

    auto input = tf.make({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto expected = tf.make({3, 2, 2}, {1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8, 9});
    auto actual_out = tf.zeros_like(expected);
    op_unfold_copy_out(input, /*dim=*/1, /*size=*/2, /*step=*/1, actual_out);
    EXPECT_TENSOR_CLOSE(actual_out, expected);
  }
};

TEST_F(OpUnfoldTest, SmokeTest) {
  TensorFactory<ScalarType::Float> tf;
  const auto input = tf.make({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  const auto expected = tf.make({3, 1, 2}, {1, 2, 4, 5, 7, 8});
  auto output = tf.zeros_like(expected);

  op_unfold_copy_out(input, /*dim=*/1, /*size=*/2, /*step=*/2, output);
  EXPECT_TENSOR_CLOSE(output, expected);
}

TEST_F(OpUnfoldTest, DType) {
#define TEST_ENTRY(ctype, dtype) \
  test_unfold_copy_dtype<ctype, ScalarType::dtype>();
  ET_FORALL_REALHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpUnfoldTest, ZeroDimension) {
  TensorFactory<ScalarType::Float> tf;
  const auto input = tf.make({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  const auto expected =
      tf.make({2, 3, 2}, {1, 4, 2, 5, 3, 6, 4, 7, 5, 8, 6, 9});
  auto output = tf.zeros_like(expected);

  op_unfold_copy_out(input, /*dim=*/0, /*size=*/2, /*step=*/1, output);
  EXPECT_TENSOR_CLOSE(output, expected);
}

TEST_F(OpUnfoldTest, NegativeDimension) {
  TensorFactory<ScalarType::Float> tf;
  const auto input = tf.make({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  const auto expected = tf.make({3, 1, 2}, {1, 2, 4, 5, 7, 8});
  auto output = tf.zeros_like(expected);

  op_unfold_copy_out(input, /*dim=*/-1, /*size=*/2, /*step=*/2, output);
  EXPECT_TENSOR_CLOSE(output, expected);
}

TEST_F(OpUnfoldTest, LargeStep) {
  TensorFactory<ScalarType::Float> tf;
  const auto input = tf.make({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  const auto expected = tf.make({3, 1, 2}, {1, 2, 4, 5, 7, 8});
  auto output = tf.zeros_like(expected);

  op_unfold_copy_out(input, /*dim=*/-1, /*size=*/2, /*step=*/5, output);
  EXPECT_TENSOR_CLOSE(output, expected);
}

TEST_F(OpUnfoldTest, ZeroSize) {
  TensorFactory<ScalarType::Float> tf;
  const auto input = tf.make({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  const auto expected = tf.make({3, 4, 0}, {});
  auto output = tf.zeros_like(expected);

  op_unfold_copy_out(input, /*dim=*/1, /*size=*/0, /*step=*/1, output);
  EXPECT_TENSOR_CLOSE(output, expected);
}

TEST_F(OpUnfoldTest, NegativeSizeAndNegativeStepDies) {
  TensorFactory<ScalarType::Float> tf;
  const auto input = tf.make({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto output = tf.zeros({3, 1, 2});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_unfold_copy_out(input, /*dim=*/1, /*size=*/-1, /*step=*/1, output));
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_unfold_copy_out(input, /*dim=*/1, /*size=*/1, /*step=*/-1, output));
}

TEST_F(OpUnfoldTest, InvalidDimAndSizeTooLargeDies) {
  TensorFactory<ScalarType::Float> tf;
  const auto input = tf.make({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto output = tf.zeros({3, 1, 2});
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_unfold_copy_out(input, /*dim=*/3, /*size=*/2, /*step=*/1, output));
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_unfold_copy_out(input, /*dim=*/1, /*size=*/10, /*step=*/1, output));
}
