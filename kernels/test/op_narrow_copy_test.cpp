/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ArrayRef;
using exec_aten::optional;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpNarrowCopyOutTest : public OperatorTest {
 protected:
  Tensor& op_narrow_copy_out(
      const Tensor& in,
      int64_t dim,
      int64_t start,
      int64_t length,
      Tensor& out) {
    return torch::executor::aten::narrow_copy_outf(
        context_, in, dim, start, length, out);
  }

  template <class CTYPE, exec_aten::ScalarType DTYPE>
  void test_dtype() {
    TensorFactory<DTYPE> tf;

    // clang-format off
    Tensor input = tf.make(
      /*sizes=*/{3, 4},
      /*data=*/{
        1,   2,   3,   4, // [0, :]
        5,   6,   7,   8, // [1, :]
        9,  10,  11,  12, // [2, :]
      });
  
    Tensor expected = tf.make(
      /*sizes=*/{2, 4},
      /*data=*/{
        1,   2,   3,   4, // [0, :]
        5,   6,   7,   8, // [1, :]
      });
    // clang-format on

    Tensor out = tf.zeros({2, 4});
    Tensor ret =
        op_narrow_copy_out(input, /*dim=*/0, /*start=*/0, /*length=*/2, out);

    EXPECT_TENSOR_EQ(out, ret);
    EXPECT_TENSOR_EQ(out, expected);
  }
};

TEST_F(OpNarrowCopyOutTest, AllDtypesSupported) {
#define TEST_ENTRY(ctype, dtype) test_dtype<ctype, ScalarType::dtype>();
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpNarrowCopyOutTest, EmptyInputSupported) {
  TensorFactory<ScalarType::Int> tf;

  Tensor input = tf.ones({1, 0, 1});
  Tensor out = tf.zeros({1, 0, 1});

  Tensor expect = tf.ones({1, 0, 1});

  Tensor ret =
      op_narrow_copy_out(input, /*dim=*/0, /*start=*/0, /*length=*/1, out);
  EXPECT_TENSOR_EQ(ret, out);
  EXPECT_TENSOR_EQ(ret, expect);

  ret = op_narrow_copy_out(input, /*dim=*/1, /*start=*/0, /*length=*/0, out);
  EXPECT_TENSOR_EQ(ret, out);
  EXPECT_TENSOR_EQ(ret, expect);

  ret = op_narrow_copy_out(input, /*dim=*/2, /*start=*/0, /*length=*/1, out);
  EXPECT_TENSOR_EQ(ret, out);
  EXPECT_TENSOR_EQ(ret, expect);
}

TEST_F(OpNarrowCopyOutTest, ZeroLengthSupported) {
  TensorFactory<ScalarType::Int> tf;

  Tensor input = tf.ones({2, 3});
  Tensor out = tf.ones({2, 0});

  Tensor expect = tf.ones({2, 0});

  Tensor ret =
      op_narrow_copy_out(input, /*dim=*/1, /*start=*/1, /*length=*/0, out);
  EXPECT_TENSOR_EQ(ret, out);
  EXPECT_TENSOR_EQ(ret, expect);

  ret = op_narrow_copy_out(input, /*dim=*/1, /*start=*/-1, /*length=*/0, out);
  EXPECT_TENSOR_EQ(ret, out);
  EXPECT_TENSOR_EQ(ret, expect);
}

TEST_F(OpNarrowCopyOutTest, ZeroDimInputDies) {
  TensorFactory<ScalarType::Int> tf;

  Tensor input = tf.ones({});
  Tensor out = tf.ones({});

  // The operation shall die whatever the end is.
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_narrow_copy_out(input, /*dim=*/0, /*start=*/0, /*length=*/0, out));
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_narrow_copy_out(input, /*dim=*/0, /*start=*/1, /*length=*/1, out));
}

TEST_F(OpNarrowCopyOutTest, InvalidStart) {
  TensorFactory<ScalarType::Int> tf;

  Tensor input = tf.ones({2, 3});
  Tensor out = tf.ones({2, 3});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_narrow_copy_out(input, /*dim=*/0, /*start=*/-3, /*length=*/0, out));
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_narrow_copy_out(input, /*dim=*/1, /*start=*/4, /*length=*/0, out));
}

TEST_F(OpNarrowCopyOutTest, InvalidStartLengthCombination) {
  TensorFactory<ScalarType::Int> tf;

  Tensor input = tf.ones({2, 3});
  Tensor out = tf.ones({2, 3});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_narrow_copy_out(input, /*dim=*/0, /*start=*/0, /*length=*/3, out));
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_narrow_copy_out(input, /*dim=*/1, /*start=*/-1, /*length=*/2, out));
}

TEST_F(OpNarrowCopyOutTest, NegativeLengthDies) {
  TensorFactory<ScalarType::Int> tf;

  Tensor input = tf.ones({1, 1, 1});
  Tensor out = tf.zeros({1, 1, 1});

  // Some invalid length values.
  const std::vector<int64_t> invalid_lengths = {-3, -2, -1};
  for (int64_t length : invalid_lengths) {
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_narrow_copy_out(
            input, /*dim=*/0, /*start=*/0, /*length=*/length, out));
  }
}

TEST_F(OpNarrowCopyOutTest, DimOutOfBoundDies) {
  TensorFactory<ScalarType::Int> tf;

  Tensor input = tf.ones({1, 1, 1});
  Tensor out = tf.zeros({1, 1, 1});

  // Some invalid dim values.
  const std::vector<int64_t> invalid_dims = {3, 4, 5, -4, -5, -6};
  for (int64_t dim : invalid_dims) {
    ET_EXPECT_KERNEL_FAILURE(
        context_,
        op_narrow_copy_out(input, dim, /*start=*/0, /*length=*/1, out));
  }
}

TEST_F(OpNarrowCopyOutTest, MismatchedDtypesDies) {
  TensorFactory<ScalarType::Int> tf_int;
  TensorFactory<ScalarType::Float> tf_float;
  Tensor input = tf_int.zeros({1, 2, 2});

  // Size is compatible to the output, but a mismatched dtype.
  Tensor out = tf_float.ones({1, 2, 2});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_narrow_copy_out(input, /*dim=*/0, /*start=*/0, /*length=*/1, out));
}
