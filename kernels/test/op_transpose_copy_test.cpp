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
using exec_aten::ArrayRef;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpTransposeIntCopyTest : public OperatorTest {
 protected:
  Tensor& op_transpose_copy_int_out(
      const Tensor& self,
      int64_t dim0,
      int64_t dim1,
      Tensor& out) {
    return torch::executor::aten::transpose_copy_outf(
        context_, self, dim0, dim1, out);
  }
};

TEST_F(OpTransposeIntCopyTest, TwoDTranspose) {
  TensorFactory<ScalarType::Int> tf;

  // clang-format off
  Tensor t_int = tf.make({2, 3}, {
    // 2x3 data block
    0, 1, 2,
    3, 4, 5
  });
  // clang-format on

  const std::vector<int32_t> new_sizes = {3, 2};
  Tensor out = tf.zeros(new_sizes);

  op_transpose_copy_int_out(t_int, 1, 0, out);
  // clang-format off
  EXPECT_TENSOR_EQ(out, tf.make(new_sizes, {
    // 3x2 data block
    0, 3,
    1, 4,
    2, 5
  }));
  // clang-format on
}

TEST_F(OpTransposeIntCopyTest, TwoDNegativeIndices) {
  TensorFactory<ScalarType::Int> tf;

  // clang-format off
  Tensor t_int = tf.make({2, 3}, {
    // 2x3 data block
    0, 1, 2,
    3, 4, 5
  });
  // clang-format on

  const std::vector<int32_t> new_sizes = {3, 2};
  Tensor out = tf.zeros(new_sizes);

  op_transpose_copy_int_out(t_int, -1, -2, out);
  // clang-format off
  EXPECT_TENSOR_EQ(out, tf.make(new_sizes, {
    // 3x2 data block
    0, 3,
    1, 4,
    2, 5
  }));
  // clang-format on
}

TEST_F(OpTransposeIntCopyTest, TransposeNoDatachange) {
  TensorFactory<ScalarType::Int> tf;

  // clang-format off
  Tensor t_int = tf.make({2, 1, 3}, {
    // 2 1x3 data blocks
    0, 1, 2,

    3, 4, 5
  });
  // clang-format on

  const std::vector<int32_t> new_sizes = {2, 3, 1};
  Tensor out = tf.zeros(new_sizes);

  op_transpose_copy_int_out(t_int, 1, 2, out);
  // clang-format off
  EXPECT_TENSOR_EQ(out, tf.make(new_sizes, {
  // 2 3x1 data blocks
    0,
    1,
    2,

    3,
    4,
    5,
  }));
  // clang-format on
}

TEST_F(OpTransposeIntCopyTest, ThreeDTranspose) {
  TensorFactory<ScalarType::Int> tf;

  // clang-format off
  Tensor t_int = tf.make({2, 2, 3}, {
    // 2 2x3 data blocks
    0, 1, 2,
    3, 4, 5,

    6, 7, 8,
    9, 10, 11
  });
  // clang-format on

  const std::vector<int32_t> new_sizes = {3, 2, 2};
  Tensor out = tf.zeros(new_sizes);

  op_transpose_copy_int_out(t_int, 0, 2, out);
  // clang-format off
  EXPECT_TENSOR_EQ(out, tf.make(new_sizes, {
  // 3 2x2 data blocks
    0, 6,
    3, 9,

    1, 7,
    4, 10,

    2,  8,
    5, 11
  }));
  // clang-format on
}

// transpose an out of bounds dim
TEST_F(OpTransposeIntCopyTest, OutOfBoundDimDies) {
  TensorFactory<ScalarType::Float> tf;

  Tensor a = tf.ones(/*sizes=*/{2, 3});
  Tensor out = tf.ones(/*sizes=*/{3, 2});

  ET_EXPECT_KERNEL_FAILURE(context_, op_transpose_copy_int_out(a, 0, -3, out));
}

// transpose a 3d tensor into a 2d one
TEST_F(OpTransposeIntCopyTest, MismatchedDimDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched dimensions";
  }
  TensorFactory<ScalarType::Float> tf;

  Tensor a = tf.ones(/*sizes=*/{4, 2, 3});
  Tensor out = tf.ones(/*sizes=*/{2, 2});

  ET_EXPECT_KERNEL_FAILURE(context_, op_transpose_copy_int_out(a, 0, 1, out));
}

/* %python
import torch
torch.manual_seed(0)
x = torch.randint(10, (2, 2, 3))
res = torch.transpose(x, 0, 2)
op = "op_transpose_copy_int_out"
opt_extra_params = "0, 2,"
dtype = "ScalarType::Int"
check = "EXPECT_TENSOR_EQ" */

TEST_F(OpTransposeIntCopyTest, DynamicShapeUpperBoundSameAsExpected) {
  /* %python
  out_args = "{3, 2, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Int> tf;

  Tensor x = tf.make({2, 2, 3}, {4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6});
  Tensor expected = tf.make({3, 2, 2}, {4, 7, 0, 3, 9, 3, 3, 1, 3, 7, 9, 6});

  Tensor out =
      tf.zeros({3, 2, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_transpose_copy_int_out(x, 0, 2, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpTransposeIntCopyTest, DynamicShapeUpperBoundLargerThanExpected) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  /* %python
  out_args = "{5, 5, 5}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Int> tf;

  Tensor x = tf.make({2, 2, 3}, {4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6});
  Tensor expected = tf.make({3, 2, 2}, {4, 7, 0, 3, 9, 3, 3, 1, 3, 7, 9, 6});

  Tensor out =
      tf.zeros({5, 5, 5}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_transpose_copy_int_out(x, 0, 2, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpTransposeIntCopyTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  /* %python
  out_args = "{1, 1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Int> tf;

  Tensor x = tf.make({2, 2, 3}, {4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6});
  Tensor expected = tf.make({3, 2, 2}, {4, 7, 0, 3, 9, 3, 3, 1, 3, 7, 9, 6});

  Tensor out = tf.zeros(
      {1, 1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  op_transpose_copy_int_out(x, 0, 2, out);
  EXPECT_TENSOR_EQ(out, expected);
}
