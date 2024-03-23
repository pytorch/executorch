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
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpTrilTest : public OperatorTest {
 protected:
  Tensor& op_tril_out(const Tensor& self, int64_t diagonal, Tensor& out) {
    return torch::executor::aten::tril_outf(context_, self, diagonal, out);
  }

  // Assert `self` and `out` as zero tensors is a no-op.
  template <ScalarType DTYPE>
  void test_tril_out_zeros() {
    TensorFactory<DTYPE> tf;

    // clang-format off
  Tensor self = tf.make(
    /*sizes=*/{3, 3},
    /*data=*/
    {
        0,  0,  0, // tensor([[ 0,  0,  0],
        0,  0,  0, //         [ 0,  0,  0],
        0,  0,  0, //         [ 0,  0,  0]])
    }
  );
    // clang-format on

    Tensor out = tf.zeros({3, 3});

    op_tril_out(self, 0, out);

    // clang-format off
  Tensor result = tf.make(
    /*sizes=*/{3, 3},
    /*data=*/
    {
        0,  0,  0, // tensor([[ 0,  0,  0],
        0,  0,  0, //         [ 0,  0,  0],
        0,  0,  0, //         [ 0,  0,  0]])
    }
  );
    // clang-format on

    EXPECT_TENSOR_EQ(out, result);
  }

  // Assert `out` as a non-zero tensor yields correct results.
  template <ScalarType DTYPE>
  void test_tril_out_ones() {
    TensorFactory<DTYPE> tf;

    // clang-format off
  Tensor self = tf.make(
    /*sizes=*/{3, 3},
    /*data=*/
    {
        0,  0,  0, // tensor([[ 0,  0,  0],
        0,  0,  0, //         [ 0,  0,  0],
        0,  0,  0, //         [ 0,  0,  0]])
    }
  );
    // clang-format on

    Tensor out = tf.ones({3, 3});

    op_tril_out(self, 0, out);

    // clang-format off
  Tensor result = tf.make(
    /*sizes=*/{3, 3},
    /*data=*/
    {
        0,  0,  0, // tensor([[ 0,  0,  0],
        0,  0,  0, //         [ 0,  0,  0],
        0,  0,  0, //         [ 0,  0,  0]])
    }
  );
    // clang-format on

    EXPECT_TENSOR_EQ(out, result);
  }

  // Assert `tril` works with multiple empty dims.
  template <ScalarType DTYPE>
  void test_tril_out_empty_dims() {
    TensorFactory<DTYPE> tf;
    Tensor out = tf.zeros({1, 1, 1, 1});

    // tensor([[[[1]]]])
    Tensor self = tf.ones({1, 1, 1, 1});

    op_tril_out(self, 0, out);

    // tensor([[[[1]]]])
    Tensor result = tf.ones({1, 1, 1, 1});

    EXPECT_TENSOR_EQ(out, result);
  }

  // Assert `tril` works with a square tensor.
  template <ScalarType DTYPE>
  void test_tril_out_square() {
    TensorFactory<DTYPE> tf;

    // clang-format off
  Tensor self = tf.make(
    /*sizes=*/{3, 3},
    /*data=*/
    {
        1,  1,  1, // tensor([[ 1,  1,  1],
        1,  1,  1, //         [ 1,  1,  1],
        1,  1,  1, //         [ 1,  1,  1]])
    }
  );
    // clang-format on

    Tensor out = tf.zeros({3, 3});

    op_tril_out(self, 0, out);

    // clang-format off
  Tensor result = tf.make(
    /*sizes=*/{3, 3},
    /*data=*/
    {
        1,  0,  0, // tensor([[ 1,  0,  0],
        1,  1,  0, //         [ 1,  1,  0],
        1,  1,  1, //         [ 1,  1,  1]])
    }
  );
    // clang-format on

    EXPECT_TENSOR_EQ(out, result);
  }

  // Assert `tril` works with a rectangular tensor.
  template <ScalarType DTYPE>
  void test_tril_out_rectangle() {
    TensorFactory<DTYPE> tf;

    // clang-format off
  Tensor self = tf.make(
    /*sizes=*/{3, 5},
    /*data=*/
    {
        1,  1,  1,  1,  1, // tensor([[ 1,  1,  1,  1,  1],
        1,  1,  1,  1,  1, //         [ 1,  1,  1,  1,  1],
        1,  1,  1,  1,  1, //         [ 1,  1,  1,  1,  1]])
    }
  );
    // clang-format on

    Tensor out = tf.zeros({3, 5});

    op_tril_out(self, 0, out);

    // clang-format off
  Tensor result = tf.make(
    /*sizes=*/{3, 5},
    /*data=*/
    {
        1,  0,  0,  0,  0, // tensor([[ 1,  0,  0,  0,  0],
        1,  1,  0,  0,  0, //         [ 1,  1,  0,  0,  0],
        1,  1,  1,  0,  0, //         [ 1,  1,  1,  0,  0]])
    }
  );
    // clang-format on

    EXPECT_TENSOR_EQ(out, result);
  }

  // Assert `tril` works with a positive diagonal value.
  template <ScalarType DTYPE>
  void test_tril_out_pos_diag() {
    TensorFactory<DTYPE> tf;

    // clang-format off
  Tensor self = tf.make(
    /*sizes=*/{3, 3},
    /*data=*/
    {
        1,  1,  1, // tensor([[ 1,  1,  1],
        1,  1,  1, //         [ 1,  1,  1],
        1,  1,  1, //         [ 1,  1,  1]])
    }
  );
    // clang-format on

    Tensor out = tf.zeros({3, 3});

    op_tril_out(self, 1, out);

    // clang-format off
  Tensor result = tf.make(
    /*sizes=*/{3, 3},
    /*data=*/
    {
        1,  1,  0, // tensor([[ 1,  1,  0],
        1,  1,  1, //         [ 1,  1,  1],
        1,  1,  1, //         [ 1,  1,  1]])
    }
  );
    // clang-format on

    EXPECT_TENSOR_EQ(out, result);
  }

  // Assert `tril` works with a negative diagonal value.
  template <ScalarType DTYPE>
  void test_tril_out_neg_diag() {
    TensorFactory<DTYPE> tf;

    // clang-format off
  Tensor self = tf.make(
    /*sizes=*/{3, 3},
    /*data=*/
    {
        1,  1,  1, // tensor([[ 1,  1,  1],
        1,  1,  1, //         [ 1,  1,  1],
        1,  1,  1, //         [ 1,  1,  1]])
    }
  );
    // clang-format on

    Tensor out = tf.zeros({3, 3});

    op_tril_out(self, -1, out);

    // clang-format off
  Tensor result = tf.make(
    /*sizes=*/{3, 3},
    /*data=*/
    {
        0,  0,  0, // tensor([[ 0,  0,  0],
        1,  0,  0, //         [ 1,  0,  0],
        1,  1,  0, //         [ 1,  1,  0]])
    }
  );
    // clang-format on

    EXPECT_TENSOR_EQ(out, result);
  }

  // Assert `tril` works with a batch of tensors, where dims are equal.
  template <ScalarType DTYPE>
  void test_tril_out_multi_equal_dim() {
    TensorFactory<DTYPE> tf;

    // clang-format off
  Tensor self = tf.make(
    /*sizes=*/{3, 3, 3},
    /*data=*/
    {
        1,  1,  1, // tensor([[[ 1,  1,  1],
        1,  1,  1, //          [ 1,  1,  1],
        1,  1,  1, //          [ 1,  1,  1]],

        1,  1,  1, //         [[ 1,  1,  1],
        1,  1,  1, //          [ 1,  1,  1],
        1,  1,  1, //          [ 1,  1,  1]],

        1,  1,  1, //         [[ 1,  1,  1],
        1,  1,  1, //          [ 1,  1,  1],
        1,  1,  1, //          [ 1,  1,  1]]])
    }
  );
    // clang-format on

    Tensor out = tf.zeros({3, 3, 3});

    op_tril_out(self, 0, out);

    // clang-format off
  Tensor result = tf.make(
    /*sizes=*/{3, 3, 3},
    /*data=*/
    {
        1,  0,  0, // tensor([[[ 1,  0,  0],
        1,  1,  0, //          [ 1,  1,  0],
        1,  1,  1, //          [ 1,  1,  1]],

        1,  0,  0, //         [[ 1,  0,  0],
        1,  1,  0, //          [ 1,  1,  0],
        1,  1,  1, //          [ 1,  1,  1]],

        1,  0,  0, //         [[ 1,  0,  0],
        1,  1,  0, //          [ 1,  1,  0],
        1,  1,  1, //          [ 1,  1,  1]]])
    }
  );
    // clang-format on

    EXPECT_TENSOR_EQ(out, result);
  }

  // Assert `tril` works with a batch of tensors, where dims are unequal.
  template <ScalarType DTYPE>
  void test_tril_out_multi_unequal_dim() {
    TensorFactory<DTYPE> tf;

    // clang-format offF
    Tensor self = tf.make(
        /*sizes=*/{3, 2, 3},
        /*data=*/
        {
            1,
            1,
            1, // tensor([[[ 1,  1,  1],
            1,
            1,
            1, //          [ 1,  1,  1]],

            1,
            1,
            1, //         [[ 1,  1,  1],
            1,
            1,
            1, //          [ 1,  1,  1]],

            1,
            1,
            1, //         [[ 1,  1,  1],
            1,
            1,
            1, //          [ 1,  1,  1]]])
        });
    // clang-format on

    Tensor out = tf.zeros({3, 2, 3});

    op_tril_out(self, 0, out);

    // clang-format off
  Tensor result = tf.make(
    /*sizes=*/{3, 2, 3},
    /*data=*/
    {
        1,  0,  0, // tensor([[[ 1,  0,  0],
        1,  1,  0, //          [ 1,  1,  0]],

        1,  0,  0, //         [[ 1,  0,  0],
        1,  1,  0, //          [ 1,  1,  0]],

        1,  0,  0, //         [[ 1,  0,  0],
        1,  1,  0, //          [ 1,  1,  0]]])
    }
  );
    // clang-format on

    EXPECT_TENSOR_EQ(out, result);
  }

  // Assert `tril` works with non-0/1 values on regular diagonal.
  template <ScalarType DTYPE>
  void test_tril_out_arange_reg_diag() {
    TensorFactory<DTYPE> tf;

    // clang-format off
  Tensor self = tf.make(
    /*sizes=*/{3, 3},
    /*data=*/
    {
        1,  2,  3, // tensor([[ 1,  2,  3],
        4,  5,  6, //         [ 4,  5,  6],
        7,  8,  9, //         [ 7,  8,  9]])
    }
  );
    // clang-format on

    Tensor out = tf.zeros({3, 3});

    op_tril_out(self, 0, out);

    // clang-format off
  Tensor result = tf.make(
    /*sizes=*/{3, 3},
    /*data=*/
    {
        1,  0,  0, // tensor([[ 1,  0,  0],
        4,  5,  0, //         [ 4,  5,  0],
        7,  8,  9, //         [ 7,  8,  9]])
    }
  );
    // clang-format on

    EXPECT_TENSOR_EQ(out, result);
  }

  // Assert `tril` works with non-0/1 values on positive diagonal values.
  // An edge case with a far-out positive diagonal is also included.
  template <ScalarType DTYPE>
  void test_tril_out_arange_pos_diag() {
    TensorFactory<DTYPE> tf;

    // Case: diag = 1

    // clang-format off
  Tensor self = tf.make(
    /*sizes=*/{3, 3},
    /*data=*/
    {
        1,  2,  3, // tensor([[ 1,  2,  3],
        4,  5,  6, //         [ 4,  5,  6],
        7,  8,  9, //         [ 7,  8,  9]])
    }
  );
    // clang-format on

    Tensor out1 = tf.zeros({3, 3});

    op_tril_out(self, 1, out1);

    // clang-format off
  Tensor result1 = tf.make(
    /*sizes=*/{3, 3},
    /*data=*/
    {
        1,  2,  0, // tensor([[ 1,  2,  0],
        4,  5,  6, //         [ 4,  5,  6],
        7,  8,  9, //         [ 7,  8,  9]])
    }
  );
    // clang-format on

    EXPECT_TENSOR_EQ(out1, result1);

    // Case: diag = 2

    Tensor out2 = tf.zeros({3, 3});
    op_tril_out(self, 2, out2);
    EXPECT_TENSOR_EQ(out2, self);

    // Case: diag = 10

    Tensor out3 = tf.zeros({3, 3});
    op_tril_out(self, 10, out3);
    EXPECT_TENSOR_EQ(out3, self);
  }

  // Assert `tril` works with non-0/1 values on negative diagonal values.
  // An edge case with a far-out negative diagonal is also included.
  template <ScalarType DTYPE>
  void test_tril_out_arange_neg_diag() {
    TensorFactory<DTYPE> tf;

    // Case: diag = -1

    // clang-format off
  Tensor self = tf.make(
    /*sizes=*/{3, 3},
    /*data=*/
    {
        1,  2,  3, // tensor([[ 1,  2,  3],
        4,  5,  6, //         [ 4,  5,  6],
        7,  8,  9, //         [ 7,  8,  9]])
    }
  );
    // clang-format on

    Tensor out1 = tf.zeros({3, 3});

    op_tril_out(self, -1, out1);

    // clang-format off
  Tensor result1 = tf.make(
    /*sizes=*/{3, 3},
    /*data=*/
    {
        0,  0,  0, // tensor([[ 0,  0,  0],
        4,  0,  0, //         [ 4,  0,  0],
        7,  8,  0, //         [ 7,  8,  0]])
    }
  );
    // clang-format on

    EXPECT_TENSOR_EQ(out1, result1);

    // Case: diag = 2

    Tensor out2 = tf.zeros({3, 3});

    op_tril_out(self, -2, out2);

    // clang-format off
  Tensor result2 = tf.make(
    /*sizes=*/{3, 3},
    /*data=*/
    {
        0,  0,  0, // tensor([[ 0,  0,  0],
        0,  0,  0, //         [ 0,  0,  0],
        7,  0,  0, //         [ 7,  0,  0]])
    }
  );
    // clang-format on

    EXPECT_TENSOR_EQ(out2, result2);

    // Case: diag = 10

    Tensor out3 = tf.zeros({3, 3});

    op_tril_out(self, -10, out3);

    // clang-format off
  Tensor result3 = tf.make(
    /*sizes=*/{3, 3},
    /*data=*/
    {
        0,  0,  0, // tensor([[ 0,  0,  0],
        0,  0,  0, //         [ 0,  0,  0],
        0,  0,  0, //         [ 0,  0,  0]])
    }
  );
    // clang-format on

    EXPECT_TENSOR_EQ(out3, result3);
  }

  // Assert `tril` works on a batch of tensors with random integers, where dims
  // are equal.
  template <ScalarType DTYPE>
  void test_tril_out_randint_multi_equal() {
    TensorFactory<DTYPE> tf;

    // clang-format off
  Tensor self = tf.make(
    /*sizes=*/{3, 3, 3, 3},
    /*data=*/
    {
        9,  5,  4, // tensor([[[[ 9,  5,  4],
        3,  9,  6, //           [ 3,  9,  6],
        9,  9,  5, //           [ 9,  9,  5]],

        7,  2,  6, //          [[ 7,  2,  6],
        8,  5,  5, //           [ 8,  5,  5],
        9,  3,  9, //           [ 9,  3,  9]],

        1,  2,  1, //          [[ 1,  2,  1],
        6,  2,  6, //           [ 6,  2,  6],
        1,  1,  8, //           [ 1,  1,  8]]],

        3,  2,  5, //         [[[ 3,  2,  5],
        4,  4,  1, //           [ 4,  4,  1],
        7,  1,  1, //           [ 7,  1,  1]],

        5,  7,  8, //          [[ 5,  7,  8],
        1,  5,  7, //           [ 1,  5,  7],
        7,  6,  3, //           [ 7,  6,  3]]],

        3,  5,  9, //          [[ 3,  5,  9],
        4,  2,  2, //           [ 4,  2,  2],
        9,  5,  2, //           [ 9,  5,  2]]],

        8,  4,  7, //         [[[ 8,  4,  7],
        8,  7,  5, //           [ 8,  7,  5],
        7,  3,  8, //           [ 7,  3,  8]],

        9,  5,  5, //          [[ 9,  5,  5],
        6,  1,  8, //           [ 6,  1,  8],
        8,  9,  7, //           [ 8,  9,  7]]],

        1,  2,  3, //          [[ 1,  2,  3],
        7,  9,  1, //           [ 7,  9,  1],
        5,  2,  2, //           [ 5,  2,  2]]]])
    }
  );
    // clang-format on

    Tensor out = tf.zeros({3, 3, 3, 3});

    op_tril_out(self, 0, out);

    // clang-format off
  Tensor result = tf.make(
    /*sizes=*/{3, 3, 3, 3},
    /*data=*/
    {
        9,  0,  0, // tensor([[[[ 9,  0,  0],
        3,  9,  0, //           [ 3,  9,  0],
        9,  9,  5, //           [ 9,  9,  5]],

        7,  0,  0, //          [[ 7,  0,  0],
        8,  5,  0, //           [ 8,  5,  0],
        9,  3,  9, //           [ 9,  3,  9]],

        1,  0,  0, //          [[ 1,  0,  0],
        6,  2,  0, //           [ 6,  2,  0],
        1,  1,  8, //           [ 1,  1,  8]]],

        3,  0,  0, //         [[[ 3,  0,  0],
        4,  4,  0, //           [ 4,  4,  0],
        7,  1,  1, //           [ 7,  1,  1]],

        5,  0,  0, //          [[ 5,  0,  0],
        1,  5,  0, //           [ 1,  5,  0],
        7,  6,  3, //           [ 7,  6,  3]]],

        3,  0,  0, //          [[ 3,  0,  0],
        4,  2,  0, //           [ 4,  2,  0],
        9,  5,  2, //           [ 9,  5,  2]]],

        8,  0,  0, //         [[[ 8,  0,  0],
        8,  7,  0, //           [ 8,  7,  0],
        7,  3,  8, //           [ 7,  3,  8]],

        9,  0,  0, //          [[ 9,  0,  0],
        6,  1,  0, //           [ 6,  1,  0],
        8,  9,  7, //           [ 8,  9,  7]]],

        1,  0,  0, //          [[ 1,  0,  0],
        7,  9,  0, //           [ 7,  9,  0],
        5,  2,  2, //           [ 5,  2,  2]]]])
    }
  );
    // clang-format on

    EXPECT_TENSOR_EQ(out, result);
  }

  // Assert `tril` works on a batch of tensors with random integers, where dims
  // are unequal.
  template <ScalarType DTYPE>
  void test_tril_out_randint_multi_unequal() {
    TensorFactory<DTYPE> tf;

    // clang-format off
  Tensor self = tf.make(
    /*sizes=*/{3, 2, 3, 2},
    /*data=*/
    {
        1,  1, // tensor([[[[ 1,  1],
        1,  1, //           [ 1,  1],
        9,  1, //           [ 9,  1]],

        1,  6, //          [[ 1,  6],
        6,  2, //           [ 6,  2],
        7,  2, //           [ 7,  2]],

        2,  4, //         [[[ 2,  4],
        8,  3, //           [ 8,  3],
        4,  2, //           [ 4,  2]]],

        7,  6, //          [[ 7,  6],
        1,  8, //           [ 1,  8],
        4,  3, //           [ 4,  3]],

        2,  2, //         [[[ 2,  2],
        7,  4, //           [ 7,  4],
        3,  7, //           [ 3,  7]]],

        7,  8, //          [[ 7,  8],
        4,  9, //           [ 4,  9],
        1,  6, //           [ 1,  6]]]])
    }
  );
    // clang-format on

    Tensor out = tf.zeros({3, 2, 3, 2});

    op_tril_out(self, 0, out);

    // clang-format off
  Tensor result = tf.make(
    /*sizes=*/{3, 2, 3, 2},
    /*data=*/
    {
        1,  0, // tensor([[[[ 1,  0],
        1,  1, //           [ 1,  1],
        9,  1, //           [ 9,  1]],

        1,  0, //          [[ 1,  0],
        6,  2, //           [ 6,  2],
        7,  2, //           [ 7,  2]],

        2,  0, //         [[[ 2,  0],
        8,  3, //           [ 8,  3],
        4,  2, //           [ 4,  2]]],

        7,  0, //          [[ 7,  0],
        1,  8, //           [ 1,  8],
        4,  3, //           [ 4,  3]],

        2,  0, //         [[[ 2,  0],
        7,  4, //           [ 7,  4],
        3,  7, //           [ 3,  7]]],

        7,  0, //          [[ 7,  0],
        4,  9, //           [ 4,  9],
        1,  6, //           [ 1,  6]]]])
    }
  );
    // clang-format on

    EXPECT_TENSOR_EQ(out, result);
  }
};

// Create generic tests for all dtypes. Tensors contain 0s or 1s.
#define GENERATE_GENERIC_TEST(_, DTYPE)                   \
  TEST_F(OpTrilTest, DTYPE##GenericTest) {                \
    test_tril_out_zeros<ScalarType::DTYPE>();             \
    test_tril_out_ones<ScalarType::DTYPE>();              \
    test_tril_out_empty_dims<ScalarType::DTYPE>();        \
    test_tril_out_square<ScalarType::DTYPE>();            \
    test_tril_out_rectangle<ScalarType::DTYPE>();         \
    test_tril_out_pos_diag<ScalarType::DTYPE>();          \
    test_tril_out_neg_diag<ScalarType::DTYPE>();          \
    test_tril_out_multi_equal_dim<ScalarType::DTYPE>();   \
    test_tril_out_multi_unequal_dim<ScalarType::DTYPE>(); \
  }

ET_FORALL_REAL_TYPES_AND(Bool, GENERATE_GENERIC_TEST)

// Create generic tests for real dtypes. Tensors have diverse values.
#define GENERATE_REAL_TEST(_, DTYPE)                          \
  TEST_F(OpTrilTest, DTYPE##RealTest) {                       \
    test_tril_out_arange_pos_diag<ScalarType::DTYPE>();       \
    test_tril_out_arange_neg_diag<ScalarType::DTYPE>();       \
    test_tril_out_randint_multi_equal<ScalarType::DTYPE>();   \
    test_tril_out_randint_multi_unequal<ScalarType::DTYPE>(); \
  }

ET_FORALL_REAL_TYPES(GENERATE_REAL_TEST)

TEST_F(OpTrilTest, InvalidInputShapesDies) {
  TensorFactory<ScalarType::Int> tf;

  // `self` and `out` invalid shapes: ndims = 0 is <2.
  Tensor self1 = tf.zeros({});
  Tensor out1 = tf.zeros({});

  // Assert `out` can't be filled due to incompatible shapes.
  ET_EXPECT_KERNEL_FAILURE(context_, op_tril_out(self1, 0, out1));

  // `self` and `out` invalid shapes: ndims = 1 is <2.
  Tensor self2 = tf.zeros({1});
  Tensor out2 = tf.zeros({1});

  // Assert `out` can't be filled due to incompatible shapes.
  ET_EXPECT_KERNEL_FAILURE(context_, op_tril_out(self2, 0, out2));
}

TEST_F(OpTrilTest, MismatchedOutputShapesDies) {
  // Skip ATen test since it supports `self` and `out` having different shapes.
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched output shape";
  }

  TensorFactory<ScalarType::Int> tf;

  // `self` and `out` have different shapes but same dtype.
  Tensor self = tf.zeros({2, 1});
  Tensor out = tf.zeros({2, 2});

  // Assert `out` can't be filled due to incompatible shapes.
  ET_EXPECT_KERNEL_FAILURE(context_, op_tril_out(self, 0, out));
}

TEST_F(OpTrilTest, MismatchedOutputDtypeDies) {
  TensorFactory<ScalarType::Byte> tf_byte;
  TensorFactory<ScalarType::Float> tf_float;

  // `self` and `out` have different dtypes but same shape.
  Tensor self = tf_byte.zeros({2, 2});
  Tensor out = tf_float.zeros({2, 2});

  // Assert `out` can't be filled due to incompatible dtype.
  ET_EXPECT_KERNEL_FAILURE(context_, op_tril_out(self, 0, out));
}

TEST_F(OpTrilTest, InvalidTensorDims) {
  // Skip ATen test since it supports `self` and `out` having different shapes.
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched output shape";
  }

  TensorFactory<ScalarType::Int> tf;

  // Create `self` and `out` with 25 dims.
  std::vector<int32_t> sizes(25, 1);
  Tensor self = tf.zeros(sizes);
  Tensor out = tf.zeros(sizes);

  // Assert `out` can't be filled due to too many tensor dims.
  ET_EXPECT_KERNEL_FAILURE(context_, op_tril_out(self, 0, out));
}
