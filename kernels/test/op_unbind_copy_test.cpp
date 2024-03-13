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
using exec_aten::ArrayRef;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using exec_aten::TensorList;
using torch::executor::testing::TensorFactory;
using torch::executor::testing::TensorListFactory;

class OpUnbindCopyIntOutTest : public OperatorTest {
 protected:
  void op_unbind_copy_int_out(const Tensor& self, int64_t dim, TensorList out) {
    return torch::executor::aten::unbind_copy_outf(context_, self, dim, out);
  }

  template <ScalarType DTYPE>
  Tensor make1x2x3(TensorFactory<DTYPE>& tf) {
    // clang-format off
    return tf.make(
        /*sizes=*/{1, 2, 3},
        /*data=*/
        {
             0,  1,  2, // tensor([[[ 0,  1,  2],
             3,  4,  5, //          [ 3,  4,  5]]])
        });
    // clang-format on
  }

  template <ScalarType DTYPE>
  void test_unbind_dim0() {
    TensorFactory<DTYPE> tf;
    TensorListFactory<DTYPE> tlf;

    // clang-format off
    std::vector<Tensor> expected_out = {
        tf.make(
            /*sizes=*/{2, 3},
            /*data=*/
            {
                 0,  1,  2, // tensor([[ 0,  1,  2],
                 3,  4,  5, //         [ 3,  4,  5]])
            }),
    };
    // clang-format on

    Tensor input = make1x2x3(tf);

    // Output list with the same shapes/dtypes as the expected outputs.
    TensorList out = tlf.zeros_like(expected_out);

    op_unbind_copy_int_out(input, /*dim=*/0, out);

    EXPECT_TENSOR_LISTS_EQ(expected_out, out);

    // Also show that python negative indexing works for this case.
    TensorList out2 = tlf.zeros_like(expected_out);
    op_unbind_copy_int_out(input, /*dim=*/-3, out2);
    EXPECT_TENSOR_LISTS_EQ(expected_out, out2);
  }

  template <ScalarType DTYPE>
  void test_unbind_dim1() {
    TensorFactory<DTYPE> tf;
    TensorListFactory<DTYPE> tlf;

    // clang-format off
    std::vector<Tensor> expected_out = {
        tf.make(
            /*sizes=*/{1, 3},
            /*data=*/
            {
                 0,  1,  2, // tensor([[ 0,  1,  2]])
            }),
        tf.make(
            /*sizes=*/{1, 3},
            /*data=*/
            {
                 3,  4,  5, // tensor([[ 3,  4,  5]])
            }),
    };
    // clang-format on

    Tensor input = make1x2x3(tf);

    // Output list with the same shapes/dtypes as the expected outputs.
    TensorList out = tlf.zeros_like(expected_out);

    op_unbind_copy_int_out(input, /*dim=*/1, out);

    EXPECT_TENSOR_LISTS_EQ(expected_out, out);

    // Also show that python negative indexing works for this case.
    TensorList out2 = tlf.zeros_like(expected_out);
    op_unbind_copy_int_out(input, /*dim=*/-2, out2);
    EXPECT_TENSOR_LISTS_EQ(expected_out, out2);
  }

  template <ScalarType DTYPE>
  void test_unbind_dim2() {
    TensorFactory<DTYPE> tf;
    TensorListFactory<DTYPE> tlf;

    // Splitting on dim=N with split_size=2 will produce a list of tensors where
    // the max dim[N] is 2, and the other dims are the same as the input.

    // clang-format off
    std::vector<Tensor> expected_out = {
        tf.make(
            /*sizes=*/{1, 2},
            /*data=*/
            {
                 0, // tensor([[ 0,
                 3, //           3]]),
            }),
        tf.make(
            /*sizes=*/{1, 2},
            /*data=*/
            {
                 1, // tensor([[ 1,
                 4, //           4]]),
            }),
        tf.make(
            /*sizes=*/{1, 2},
            /*data=*/
            {
                 2, // tensor([[ 2,
                 5, //           5]]),
            }),
    };
    // clang-format on

    Tensor input = make1x2x3(tf);

    // Output list with the same shapes/dtypes as the expected outputs.
    TensorList out = tlf.zeros_like(expected_out);

    op_unbind_copy_int_out(input, /*dim=*/2, out);

    EXPECT_TENSOR_LISTS_EQ(expected_out, out);

    // Also show that python negative indexing works for this case.
    TensorList out2 = tlf.zeros_like(expected_out);
    op_unbind_copy_int_out(input, /*dim=*/-1, out2);
    EXPECT_TENSOR_LISTS_EQ(expected_out, out2);
  }

  /* %python
  import torch
  torch.manual_seed(0)
  x = torch.randint(10, (2, 3, 4))
  res = torch.unbind(x, 1)
  op = "op_unbind_copy_int_out"
  opt_extra_params = "1,"
  out_args = [
    "out_shape, dynamism",
    "out_shape, dynamism",
    "out_shape, dynamism"
  ]
  dtype = "ScalarType::Int"
  check = "EXPECT_TENSOR_LISTS_EQ" */

  void test_dynamic_shape(
      const std::vector<int32_t>& out_shape,
      enum torch::executor::TensorShapeDynamism dynamism) {
    /* %python
    %rewrite(unary_op_tensor_list_out) */

    TensorFactory<ScalarType::Int> tf;

    Tensor x = tf.make({2, 3, 4}, {4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6,
                                   6, 9, 8, 6, 6, 8, 4, 3, 6, 9, 1, 4});
    std::vector<Tensor> expectedv = {
        tf.make({2, 4}, {4, 9, 3, 0, 6, 9, 8, 6}),
        tf.make({2, 4}, {3, 9, 7, 3, 6, 8, 4, 3}),
        tf.make({2, 4}, {7, 3, 1, 6, 6, 9, 1, 4})};
    TensorList expected(expectedv.data(), expectedv.size());

    std::vector<Tensor> outv = {
        tf.zeros(out_shape, dynamism),
        tf.zeros(out_shape, dynamism),
        tf.zeros(out_shape, dynamism)};
    TensorList out(outv.data(), outv.size());
    op_unbind_copy_int_out(x, 1, out);
    EXPECT_TENSOR_LISTS_EQ(out, expected);
  }
};

/**
 * Returns a 1x2x3 contiguous tensor where the underlying data counts from 0 to
 * 26.
 */
TEST_F(OpUnbindCopyIntOutTest, Unbind1x2x3OnDim0AllRealDtypes) {
#define TEST_ENTRY(ctype, dtype) test_unbind_dim0<ScalarType::dtype>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpUnbindCopyIntOutTest, Unbind1x2x3OnDim1AllRealDTypes) {
#define TEST_ENTRY(ctype, dtype) test_unbind_dim1<ScalarType::dtype>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpUnbindCopyIntOutTest, Unbind1x2x3OnDim2AllRealDTypes) {
#define TEST_ENTRY(ctype, dtype) test_unbind_dim2<ScalarType::dtype>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpUnbindCopyIntOutTest, ZeroDimensionalInputTensorDies) {
  TensorFactory<ScalarType::Int> tf;
  TensorListFactory<ScalarType::Int> tlf;

  Tensor input = tf.ones(/*sizes=*/{});
  // Arbitrary output shape since this input can't be split.
  TensorList out = tlf.zeros_like({input});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_unbind_copy_int_out(input, /*dim=*/0, out));
}

TEST_F(OpUnbindCopyIntOutTest, UnbindWorksWithZeroSizedTensors) {
  TensorFactory<ScalarType::Int> tf;
  TensorListFactory<ScalarType::Int> tlf;

  Tensor input = tf.ones(/*sizes=*/{1, 0, 2});
  EXPECT_EQ(input.numel(), 0);

  // unbind dim 0
  std::vector<Tensor> expected_out = {tf.ones({0, 2})};

  TensorList out = tlf.zeros_like(expected_out);

  op_unbind_copy_int_out(input, /*dim=*/0, out);
  EXPECT_TENSOR_LISTS_EQ(out, expected_out);

  // unbind dim 1
  expected_out = {};

  out = tlf.zeros_like(expected_out);

  op_unbind_copy_int_out(input, /*dim=*/1, out);
  EXPECT_TENSOR_LISTS_EQ(out, expected_out);

  // unbind dim 2
  expected_out = {tf.ones({1, 0}), tf.ones({1, 0})};

  out = tlf.zeros_like(expected_out);

  op_unbind_copy_int_out(input, /*dim=*/2, out);
  EXPECT_TENSOR_LISTS_EQ(out, expected_out);
}

TEST_F(OpUnbindCopyIntOutTest, UnbindFailsWithWronglyAllocatedOutput) {
  TensorFactory<ScalarType::Int> tf;
  TensorListFactory<ScalarType::Int> tlf;

  Tensor input = tf.ones(/*sizes=*/{1, 2, 3});

  // unbind dim 1
  std::vector<Tensor> expected_out = {tf.ones({1, 3})};

  TensorList out = tlf.zeros_like(expected_out);

  // Die because length of the list should be 2
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_unbind_copy_int_out(input, /*dim=*/1, out));

  expected_out = {tf.ones({1, 4}), tf.ones({1, 4})};

  out = tlf.zeros_like(expected_out);

  // Die because output tensors in the list should be of correct sizes
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_unbind_copy_int_out(input, /*dim=*/1, out));

  expected_out = {tf.ones({1}), tf.ones({1})};

  out = tlf.zeros_like(expected_out);

  // Die because output tensors in the list should have correct number of dims
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_unbind_copy_int_out(input, /*dim=*/1, out));
}

TEST_F(OpUnbindCopyIntOutTest, UnbindProduceScalarTensors) {
  TensorFactory<ScalarType::Int> tf;
  TensorListFactory<ScalarType::Int> tlf;

  Tensor input = tf.make(
      /*sizes=*/{3}, {4, 5, 6});

  // unbind dim 1
  std::vector<Tensor> expected_out = {
      tf.make({}, {4}),
      tf.make({}, {5}),
      tf.make({}, {6}),
  };

  TensorList out = tlf.zeros_like(expected_out);

  op_unbind_copy_int_out(input, /*dim=*/0, out);
  EXPECT_TENSOR_LISTS_EQ(out, expected_out);
}

TEST_F(OpUnbindCopyIntOutTest, UnbindProduceScalarLikeTensors) {
  TensorFactory<ScalarType::Int> tf;
  TensorListFactory<ScalarType::Int> tlf;

  Tensor input = tf.make(
      /*sizes=*/{3, 1}, {4, 5, 6});

  // unbind dim 0
  std::vector<Tensor> expected_out = {
      tf.make({1}, {4}),
      tf.make({1}, {5}),
      tf.make({1}, {6}),
  };

  TensorList out = tlf.zeros_like(expected_out);

  op_unbind_copy_int_out(input, /*dim=*/0, out);
  EXPECT_TENSOR_LISTS_EQ(out, expected_out);

  input = tf.make(
      /*sizes=*/{1, 3}, {4, 5, 6});

  // unbind dim 1
  expected_out = {
      tf.make({1}, {4}),
      tf.make({1}, {5}),
      tf.make({1}, {6}),
  };

  out = tlf.zeros_like(expected_out);

  op_unbind_copy_int_out(input, /*dim=*/1, out);
  EXPECT_TENSOR_LISTS_EQ(out, expected_out);
}

TEST_F(OpUnbindCopyIntOutTest, DynamicShapeUpperBoundSameAsExpected) {
  test_dynamic_shape(
      {2, 4}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(OpUnbindCopyIntOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  GTEST_SKIP() << "Dynamic shape not supported";
  test_dynamic_shape(
      {10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(OpUnbindCopyIntOutTest, DynamicShapeUnbound) {
  GTEST_SKIP() << "Dynamic shape not supported";
  test_dynamic_shape(
      {1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
}
