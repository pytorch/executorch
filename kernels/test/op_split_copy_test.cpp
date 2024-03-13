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

class OpSplitCopyTensorOutTest : public OperatorTest {
 protected:
  void op_split_copy_tensor_out(
      const Tensor& self,
      int64_t split_size,
      int64_t dim,
      TensorList out) {
    return torch::executor::aten::split_copy_outf(
        context_, self, split_size, dim, out);
  }

  template <ScalarType DTYPE>
  Tensor make3x3x3(TensorFactory<DTYPE>& tf) {
    // clang-format off
    return tf.make(
        /*sizes=*/{3, 3, 3},
        /*data=*/
        {
             0,  1,  2, // tensor([[[ 0,  1,  2],
             3,  4,  5, //          [ 3,  4,  5],
             6,  7,  8, //          [ 6,  7,  8]],

             9, 10, 11, //         [[ 9, 10, 11],
            12, 13, 14, //          [12, 13, 14],
            15, 16, 17, //          [15, 16, 17]],

            18, 19, 20, //         [[18, 19, 20],
            21, 22, 23, //          [21, 22, 23],
            24, 25, 26, //          [24, 25, 26]]])
        });
    // clang-format on
  }

  // A simple successful test case that will work for any real dtype and bool.
  template <ScalarType DTYPE>
  void test_dtype() {
    TensorFactory<DTYPE> tf;
    TensorListFactory<DTYPE> tlf;

    Tensor input = tf.make(/*sizes=*/{2, 2}, /*data=*/{1, 0, 0, 1});

    std::vector<Tensor> expected_out = {
        tf.make(/*sizes=*/{1, 2}, /*data=*/{1, 0}),
        tf.make(/*sizes=*/{1, 2}, /*data=*/{0, 1}),
    };
    TensorList out = tlf.zeros_like(expected_out);

    op_split_copy_tensor_out(input, /*split_size=*/1, /*dim=*/0, out);

    EXPECT_TENSOR_LISTS_EQ(out, expected_out);
  }

  /* %python
  import torch
  torch.manual_seed(0)
  x = torch.randint(10, (2, 9))
  res = torch.split(x, 3, 1)
  op = "op_split_copy_tensor_out"
  opt_extra_params = "3, 1,"
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

    Tensor x =
        tf.make({2, 9}, {4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6, 6, 9, 8, 6, 6, 8});
    std::vector<Tensor> expectedv = {
        tf.make({2, 3}, {4, 9, 3, 3, 1, 6}),
        tf.make({2, 3}, {0, 3, 9, 6, 9, 8}),
        tf.make({2, 3}, {7, 3, 7, 6, 6, 8})};
    TensorList expected(expectedv.data(), expectedv.size());

    std::vector<Tensor> outv = {
        tf.zeros(out_shape, dynamism),
        tf.zeros(out_shape, dynamism),
        tf.zeros(out_shape, dynamism)};
    TensorList out(outv.data(), outv.size());
    op_split_copy_tensor_out(x, 3, 1, out);
    EXPECT_TENSOR_LISTS_EQ(out, expected);
  }
};

/**
 * Returns a 3x3x3 contiguous tensor where the underlying data counts from 0 to
 * 26.
 */
TEST_F(OpSplitCopyTensorOutTest, Split3x3x3OnDim0) {
  TensorFactory<ScalarType::Int> tf;
  TensorListFactory<ScalarType::Int> tlf;

  // Splitting on dim=N with split_size=2 will produce a list of tensors where
  // the max dim[N] is 2, and the other dims are the same as the input.

  // clang-format off
  std::vector<Tensor> expected_out = {
      tf.make(
          /*sizes=*/{2, 3, 3},
          /*data=*/
          {
               0,  1,  2, // tensor([[[ 0,  1,  2],
               3,  4,  5, //          [ 3,  4,  5],
               6,  7,  8, //          [ 6,  7,  8]],

               9, 10, 11, //         [[ 9, 10, 11],
              12, 13, 14, //          [12, 13, 14],
              15, 16, 17, //          [15, 16, 17]]])
          }),
      tf.make(
          /*sizes=*/{1, 3, 3},
          /*data=*/
          {
              18, 19, 20, // tensor([[[18, 19, 20],
              21, 22, 23, //          [21, 22, 23],
              24, 25, 26, //          [24, 25, 26]]])
          }),
  };
  // clang-format on

  Tensor input = make3x3x3(tf);

  // Output list with the same shapes/dtypes as the expected outputs.
  TensorList out = tlf.zeros_like(expected_out);

  op_split_copy_tensor_out(input, /*split_size=*/2, /*dim=*/0, out);

  EXPECT_TENSOR_LISTS_EQ(expected_out, out);

  // Also show that python negative indexing works for this case.
  TensorList out2 = tlf.zeros_like(expected_out);
  op_split_copy_tensor_out(input, /*split_size=*/2, /*dim=*/-3, out2);
  EXPECT_TENSOR_LISTS_EQ(expected_out, out2);
}

TEST_F(OpSplitCopyTensorOutTest, Split3x3x3OnDim1) {
  TensorFactory<ScalarType::Int> tf;
  TensorListFactory<ScalarType::Int> tlf;

  // Splitting on dim=N with split_size=2 will produce a list of tensors where
  // the max dim[N] is 2, and the other dims are the same as the input.

  // clang-format off
  std::vector<Tensor> expected_out = {
      tf.make(
          /*sizes=*/{3, 2, 3},
          /*data=*/
          {
               0,  1,  2, // tensor([[[ 0,  1,  2],
               3,  4,  5, //          [ 3,  4,  5]],

               9, 10, 11, //         [[ 9, 10, 11],
              12, 13, 14, //          [12, 13, 14]],

              18, 19, 20, //         [[18, 19, 20],
              21, 22, 23, //          [21, 22, 23]]]),
          }),
      tf.make(
          /*sizes=*/{3, 1, 3},
          /*data=*/
          {
               6,  7,  8, // tensor([[[ 6,  7,  8]],

              15, 16, 17, //         [[15, 16, 17]],

              24, 25, 26, //         [[24, 25, 26]]])
          }),
  };
  // clang-format on

  Tensor input = make3x3x3(tf);

  // Output list with the same shapes/dtypes as the expected outputs.
  TensorList out = tlf.zeros_like(expected_out);

  op_split_copy_tensor_out(input, /*split_size=*/2, /*dim=*/1, out);

  EXPECT_TENSOR_LISTS_EQ(expected_out, out);

  // Also show that python negative indexing works for this case.
  TensorList out2 = tlf.zeros_like(expected_out);
  op_split_copy_tensor_out(input, /*split_size=*/2, /*dim=*/-2, out2);
  EXPECT_TENSOR_LISTS_EQ(expected_out, out2);
}

TEST_F(OpSplitCopyTensorOutTest, Split3x3x3OnDim2) {
  TensorFactory<ScalarType::Int> tf;
  TensorListFactory<ScalarType::Int> tlf;

  // Splitting on dim=N with split_size=2 will produce a list of tensors where
  // the max dim[N] is 2, and the other dims are the same as the input.

  // clang-format off
  std::vector<Tensor> expected_out = {
      tf.make(
          /*sizes=*/{3, 3, 2},
          /*data=*/
          {
               0,  1, // tensor([[[ 0,  1],
               3,  4, //          [ 3,  4],
               6,  7, //          [ 6,  7]],

               9, 10, //         [[ 9, 10],
              12, 13, //          [12, 13],
              15, 16, //          [15, 16]],

              18, 19, //         [[18, 19],
              21, 22, //          [21, 22],
              24, 25, //          [24, 25]]])
          }),
      tf.make(
          /*sizes=*/{3, 3, 1},
          /*data=*/
          {
               2, // tensor([[[ 2],
               5, //          [ 5],
               8, //          [ 8]],

              11, //         [[11],
              14, //          [14],
              17, //          [17]],

              20, //         [[20],
              23, //          [23],
              26, //          [26]]])
          }),
  };
  // clang-format on

  Tensor input = make3x3x3(tf);

  // Output list with the same shapes/dtypes as the expected outputs.
  TensorList out = tlf.zeros_like(expected_out);

  op_split_copy_tensor_out(input, /*split_size=*/2, /*dim=*/2, out);

  EXPECT_TENSOR_LISTS_EQ(expected_out, out);

  // Also show that python negative indexing works for this case.
  TensorList out2 = tlf.zeros_like(expected_out);
  op_split_copy_tensor_out(input, /*split_size=*/2, /*dim=*/-1, out2);
  EXPECT_TENSOR_LISTS_EQ(expected_out, out2);
}

TEST_F(OpSplitCopyTensorOutTest, LargerSplitSizeDoesNothing) {
  TensorFactory<ScalarType::Int> tf;
  TensorListFactory<ScalarType::Int> tlf;

  Tensor input = make3x3x3(tf);

  // Since split_size will be >= the largest dimension, slicing along any
  // dimension should return the unmodified input as the only output entry.
  std::vector<Tensor> expected_out = {input};

  for (int64_t split_size = 3; split_size < 6; ++split_size) {
    for (size_t dim = 0; dim < input.dim(); ++dim) {
      TensorList out = tlf.zeros_like({input});
      op_split_copy_tensor_out(input, split_size, dim, out);
      EXPECT_TENSOR_LISTS_EQ(out, expected_out);
    }
  }
}

TEST_F(OpSplitCopyTensorOutTest, AllDtypesSupported) {
#define TEST_ENTRY(ctype, dtype) test_dtype<ScalarType::dtype>();
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
  // TODO: Also add tests for half, complex, quantized, and other types. Easiest
  // way to do that would be to make TensorFactory support zeros() and ones()
  // for those types.
}

TEST_F(OpSplitCopyTensorOutTest, EmptyInputTensor) {
  TensorFactory<ScalarType::Int> tf;
  TensorListFactory<ScalarType::Int> tlf;

  Tensor input = tf.ones(/*sizes=*/{0});
  EXPECT_EQ(input.numel(), 0);

  std::vector<Tensor> expected_out = {input};

  // Splitting a zero-size tensor succeeds, even for split_size zero.
  TensorList out = tlf.zeros_like({input});
  for (int64_t split_size = 0; split_size < 3; ++split_size) {
    op_split_copy_tensor_out(input, split_size, /*dim=*/0, out);
    EXPECT_TENSOR_LISTS_EQ(out, expected_out);
  }
}

TEST_F(OpSplitCopyTensorOutTest, ZeroDimensionalInputTensorDies) {
  TensorFactory<ScalarType::Int> tf;
  TensorListFactory<ScalarType::Int> tlf;

  Tensor input = tf.ones(/*sizes=*/{});
  // Arbitrary output shape since this input can't be split.
  TensorList out = tlf.zeros_like({input});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_split_copy_tensor_out(input, /*split_size=*/1, /*dim=*/0, out));
}

TEST_F(OpSplitCopyTensorOutTest, ZeroSplitSizeOnlyWorksForZeroSizeDims) {
  TensorFactory<ScalarType::Int> tf;
  TensorListFactory<ScalarType::Int> tlf;

  Tensor input = tf.ones(/*sizes=*/{1, 0, 2});
  EXPECT_EQ(input.numel(), 0);

  std::vector<Tensor> expected_out = {input};

  TensorList out = tlf.zeros_like({input});

  // Fails when trying to split with size zero on a dim with size > 0.
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_split_copy_tensor_out(input, /*split_size=*/0, /*dim=*/0, out));

  // Successfully splits with size zero on a dim with size == 0.
  op_split_copy_tensor_out(input, /*split_size=*/0, /*dim=*/1, out);
  EXPECT_TENSOR_LISTS_EQ(out, expected_out);

  // Fails again when trying to split with size zero on a dim with size > 0.
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_split_copy_tensor_out(input, /*split_size=*/0, /*dim=*/2, out));
}

TEST_F(OpSplitCopyTensorOutTest, NegativeSplitSizeFails) {
  TensorFactory<ScalarType::Int> tf;
  TensorListFactory<ScalarType::Int> tlf;

  Tensor input = tf.ones(/*sizes=*/{2, 2});
  // Arbitrary output shape since there's no actual valid size.
  TensorList out = tlf.zeros_like({input});

  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_split_copy_tensor_out(input, /*split_size=*/-1, /*dim=*/0, out));
}

TEST_F(OpSplitCopyTensorOutTest, OutOfRangeDimsDie) {
  TensorFactory<ScalarType::Int> tf;
  TensorListFactory<ScalarType::Int> tlf;

  Tensor input = tf.ones(/*sizes=*/{2, 2});

  std::vector<int64_t> good_dims = {-2, -1, 0, 1};
  std::vector<int64_t> bad_dims = {-4, -3, 2, 3};

  // Since split_size is >= the largest dimension, slicing along any
  // dimension should return the unmodified input as the only output entry.
  constexpr int64_t split_size = 2;
  std::vector<Tensor> expected_out = {input};

  for (auto dim : good_dims) {
    TensorList out = tlf.zeros_like({input});
    op_split_copy_tensor_out(input, split_size, dim, out);
    EXPECT_TENSOR_LISTS_EQ(out, expected_out);
  }

  for (auto dim : bad_dims) {
    TensorList out = tlf.zeros_like({input});
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_split_copy_tensor_out(input, split_size, dim, out));
  }
}

TEST_F(OpSplitCopyTensorOutTest, DtypeMismatchDies) {
  GTEST_SKIP() << "ATen kernel can handle dtype mismatch";
  TensorFactory<ScalarType::Int> tf_int;
  TensorListFactory<ScalarType::Int> tlf_int;
  TensorListFactory<ScalarType::Float> tlf_float;

  Tensor input = tf_int.ones(/*sizes=*/{2, 2});

  // Use a split_size that produces a single output entry on success.
  constexpr int64_t split_size = 2;
  constexpr int64_t dim = 0;

  // Demonstrate that this setup works when the dtypes are the same.
  {
    TensorList out = tlf_int.zeros_like({input});
    op_split_copy_tensor_out(input, split_size, dim, out);
    EXPECT_TENSOR_LISTS_EQ(out, std::vector<Tensor>({input}));
  }

  // Dies with the same setup but the output dtype is different.
  {
    TensorList out = tlf_float.zeros_like({input});
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_split_copy_tensor_out(input, split_size, dim, out));
  }
}

TEST_F(OpSplitCopyTensorOutTest, WrongNumOutputEntriesDies) {
  TensorFactory<ScalarType::Int> tf;
  TensorListFactory<ScalarType::Int> tlf;

  Tensor input = tf.ones(/*sizes=*/{3});

  // Use a split_size that produces two output entries on success.
  constexpr int64_t split_size = 2;
  constexpr int64_t dim = 0;

  // Demonstrate that splitting the input should produce two output entries.
  {
    std::vector<Tensor> expected_out = {
        tf.ones(/*sizes=*/{2}),
        tf.ones(/*sizes=*/{1}),
    };
    TensorList out = tlf.zeros_like(expected_out);
    op_split_copy_tensor_out(input, split_size, dim, out);
    EXPECT_TENSOR_LISTS_EQ(out, expected_out);
  }

  // Dies with the same setup but the output has one fewer entry than it should.
  {
    std::vector<Tensor> incorrect_out = {
        tf.ones(/*sizes=*/{2}),
        // Missing second entry.
    };
    TensorList out = tlf.zeros_like(incorrect_out);
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_split_copy_tensor_out(input, split_size, dim, out));
  }

  // Dies with the same setup but the output has one more entry than it should.
  {
    std::vector<Tensor> incorrect_out = {
        tf.ones(/*sizes=*/{2}),
        tf.ones(/*sizes=*/{1}),
        tf.ones(/*sizes=*/{1}), // Extra entry.
    };
    TensorList out = tlf.zeros_like(incorrect_out);
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_split_copy_tensor_out(input, split_size, dim, out));
  }
}

TEST_F(OpSplitCopyTensorOutTest, WrongOutputShapeDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle wrong out shape";
  }
  TensorFactory<ScalarType::Int> tf;
  TensorListFactory<ScalarType::Int> tlf;

  Tensor input = tf.ones(/*sizes=*/{5, 3, 4});

  // Use a split_size that produces two output entries on success.
  constexpr int64_t split_size = 2;
  constexpr int64_t dim = 1;

  // Demonstrate the shapes that this split should produce.
  {
    std::vector<Tensor> expected_out = {
        tf.ones(/*sizes=*/{5, 2, 4}),
        tf.ones(/*sizes=*/{5, 1, 4}),
    };
    TensorList out = tlf.zeros_like(expected_out);
    op_split_copy_tensor_out(input, split_size, dim, out);
    EXPECT_TENSOR_LISTS_EQ(out, expected_out);
  }

  // Make each of the dimensions of the final element incorrect.
  {
    std::vector<Tensor> incorrect_out = {
        tf.ones(/*sizes=*/{5, 2, 4}),
        tf.ones(/*sizes=*/{5 + 1, 1, 4}), // Wrong size for dim 0.
    };
    TensorList out = tlf.zeros_like(incorrect_out);
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_split_copy_tensor_out(input, split_size, dim, out));
  }
  {
    std::vector<Tensor> incorrect_out = {
        tf.ones(/*sizes=*/{5, 2, 4}),
        tf.ones(/*sizes=*/{5, 1 + 1, 4}), // Wrong size for dim 1 (split dim).
    };
    TensorList out = tlf.zeros_like(incorrect_out);
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_split_copy_tensor_out(input, split_size, dim, out));
  }
  {
    std::vector<Tensor> incorrect_out = {
        tf.ones(/*sizes=*/{5, 2, 4}),
        tf.ones(/*sizes=*/{5, 1, 4 + 1}), // Wrong size for dim 2.
    };
    TensorList out = tlf.zeros_like(incorrect_out);
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_split_copy_tensor_out(input, split_size, dim, out));
  }

  // Wrong size of the split dimension for the non-last output element.
  {
    std::vector<Tensor> incorrect_out = {
        tf.ones(/*sizes=*/{5, 2 + 1, 4}), // Wrong size for dim 1 (split dim).
        tf.ones(/*sizes=*/{5, 1, 4}),
    };
    TensorList out = tlf.zeros_like(incorrect_out);
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_split_copy_tensor_out(input, split_size, dim, out));
  }

  // Wrong number of output dimensions.
  {
    std::vector<Tensor> incorrect_out = {
        tf.ones(/*sizes=*/{5, 2, 4}),
        tf.ones(/*sizes=*/{5, 1, 4, 2}), // Extra dimension
    };
    TensorList out = tlf.zeros_like(incorrect_out);
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_split_copy_tensor_out(input, split_size, dim, out));
  }
  {
    std::vector<Tensor> incorrect_out = {
        tf.ones(/*sizes=*/{5, 2, 4}),
        tf.ones(/*sizes=*/{5, 1}), // Missing dimension
    };
    TensorList out = tlf.zeros_like(incorrect_out);
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_split_copy_tensor_out(input, split_size, dim, out));
  }
}

TEST_F(OpSplitCopyTensorOutTest, DynamicShapeUpperBoundSameAsExpected) {
  test_dynamic_shape(
      {2, 3}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(OpSplitCopyTensorOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  GTEST_SKIP() << "Dynamic shape not supported";
  test_dynamic_shape(
      {10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(OpSplitCopyTensorOutTest, DynamicShapeUnbound) {
  GTEST_SKIP() << "Dynamic shape not supported";
  test_dynamic_shape(
      {1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
}
