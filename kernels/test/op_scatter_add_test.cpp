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

class OpScatterAddOutTest : public OperatorTest {
 protected:
  Tensor& op_scatter_add_out(
      const Tensor& self,
      int64_t dim,
      const Tensor& index,
      const Tensor& src,
      Tensor& out) {
    return torch::executor::aten::scatter_add_outf(
        context_, self, dim, index, src, out);
  }

  // Common testing for the operator
  template <ScalarType DATA_DTYPE>
  void test_scatter_add_out() {
    TensorFactory<ScalarType::Long> tf_index;
    TensorFactory<DATA_DTYPE> tf_data;
    const std::vector<int32_t> sizes = {3, 5};
    // clang-format off
    Tensor src = tf_data.make(
      /*sizes=*/{2, 5},
      {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10
      });
    // clang-format on
    Tensor self = tf_data.zeros(sizes);
    Tensor out = tf_data.zeros(sizes);
    // clang-format off
    Tensor index = tf_index.make(
      /*sizes=*/{2, 3},
      {
        0, 1, 2,
        0, 1, 2
      });
    // clang-format on

    // Valid input should give the expected output
    op_scatter_add_out(self, 0, index, src, out);
    // clang-format off
    EXPECT_TENSOR_EQ(
        out, tf_data.make(
          sizes,
          {
            7, 0, 0,  0, 0,
            0, 9, 0,  0, 0,
            0, 0, 11, 0, 0
          }));
    // clang-format on

    // Valid input should give the expected output
    op_scatter_add_out(self, 1, index, src, out);
    // clang-format off
    EXPECT_TENSOR_EQ(
        out, tf_data.make(sizes,
        {
          1, 2, 3, 0, 0,
          6, 7, 8, 0, 0,
          0, 0, 0, 0, 0
        }));

    src = tf_data.make(
        /*sizes=*/{2, 3, 3},
        {
          // [0, :, :]
          1,  2,  3,
          4,  5,  6,
          7,  8,  9,

          // [1, :, :]
          10, 11, 12,
          13, 14, 15,
          16, 17, 18
        });
    // clang-format on
    self = tf_data.ones(/*sizes=*/{2, 3, 3});
    out = tf_data.zeros(/*sizes=*/{2, 3, 3});
    // clang-format off
    index = tf_index.make(
      /*sizes=*/{1, 3, 2},
      {
        0, 1,
        1, 2,
        0, 2
      });
    // clang-format on

    op_scatter_add_out(self, 1, index, src, out);
    // clang-format off
    EXPECT_TENSOR_EQ(
        out,
        tf_data.make(
            /*sizes=*/{2, 3, 3},
            {
              // [0, :, :]
              9, 1,  1,
              5, 3,  1,
              1, 14, 1,

              // [1, :, :]
              1, 1,  1,
              1, 1,  1,
              1, 1,  1
            }));
    // clang-format on

    out = tf_data.zeros(/*sizes=*/{2, 3, 3});
    op_scatter_add_out(self, 2, index, src, out);
    // clang-format off
    EXPECT_TENSOR_EQ(
        out,
        tf_data.make(
            /*sizes=*/{2, 3, 3},
            {
              // [0, :, :]
              2, 3, 1,
              1, 5, 6,
              8, 1, 9,

              // [1, :, :]
              1, 1, 1,
              1, 1, 1,
              1, 1, 1
            }));
    // clang-format on
  }

  // Invalid dimensions
  template <ScalarType DATA_DTYPE>
  void test_scatter_add_out_invalid_dim() {
    TensorFactory<ScalarType::Long> tf_index;
    TensorFactory<DATA_DTYPE> tf_data;
    const std::vector<int32_t> sizes = {3, 5};
    // clang-format off
    Tensor src = tf_data.make(/*sizes=*/{2, 5},
      {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10
      });
    Tensor index = tf_index.make(/*sizes=*/{2, 3},
      {
        0, 1, 2,
        0, 1, 2
      });
    // clang-format on
    Tensor self = tf_data.zeros(sizes);
    Tensor out = tf_data.zeros(sizes);

    // Invalid dim should die
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_scatter_add_out(self, -3, index, src, out));
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_scatter_add_out(self, 2, index, src, out));

    // Self, index and src hsould have same number of dimensions
    src = tf_data.zeros(/*sizes=*/{2, 2, 2});
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_scatter_add_out(self, 0, index, src, out));

    src = tf_data.zeros(/*sizes=*/{5, 5});
    index = tf_index.zeros(/*sizes=*/{2, 2, 2});
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_scatter_add_out(self, 0, index, src, out));

    // Size of dimension of index should be smaller than the size of that
    // dimension of src
    index = tf_index.zeros(/*sizes=*/{4, 6});
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_scatter_add_out(self, 0, index, src, out));

    // Size of dimension of index should be smaller than the size of that
    // dimension of self if dimension != dim
    index = tf_index.zeros(/*sizes=*/{4, 5});
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_scatter_add_out(self, 1, index, src, out));

    // Index out of bound for self in dim
    index = tf_index.make(/*sizes=*/{2, 3}, {0, 1, 3, 0, 1, 3});
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_scatter_add_out(self, 0, index, src, out));
  }

  // Mismatched shape
  template <ScalarType DATA_DTYPE>
  void test_scatter_add_out_mismatched_shape() {
    TensorFactory<ScalarType::Long> tf_index;
    TensorFactory<DATA_DTYPE> tf_data;

    // clang-format off
    Tensor src = tf_data.make(/*sizes=*/{2, 5},
      {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10
      });
    Tensor index = tf_index.make(/*sizes=*/{2, 3},
      {
        0, 1, 2,
        0, 1, 2
      });
    // clang-format on
    Tensor self = tf_data.zeros(/*sizes=*/{3, 5});
    Tensor out = tf_data.zeros(/*sizes=*/{2, 5});

    // self and out should be of the same shape
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_scatter_add_out(self, 0, index, src, out));
  }

  /* %python
  import torch
  torch.manual_seed(0)
  input_shape = (2, 3, 4)
  input = torch.randint(10, input_shape)
  dim = 2
  index = torch.randint(input.size(dim), input_shape)
  src = torch.randint(10, input_shape)
  expected = torch.scatter_add(input, dim, index, src)

  scatter_add_template = f"""
    {declare_tensor_factory("ScalarType::Int", "tf")}
    {declare_tensor_factory("ScalarType::Long", "tf_index")}

    {declare_tensor_make_t("input", "tf")}
    {declare_tensor_make_t("index", "tf_index")}
    {declare_tensor_make_t("src", "tf")}
    {declare_tensor_make_t("expected", "tf")}
    {declare_tensor_zeros("out_shape, dynamism", "tf", "out")}

    op_scatter_add_out(input, $dim$, index, src, out);
    EXPECT_TENSOR_EQ(out, expected);""" */

  void test_dynamic_shape(
      const std::vector<int32_t>& out_shape,
      enum torch::executor::TensorShapeDynamism dynamism) {
    /* %python
    %rewrite(scatter_add_template) */

    TensorFactory<ScalarType::Int> tf;
    TensorFactory<ScalarType::Long> tf_index;

    Tensor input = tf.make({2, 3, 4}, {4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6,
                                       6, 9, 8, 6, 6, 8, 4, 3, 6, 9, 1, 4});
    Tensor index =
        tf_index.make({2, 3, 4}, {0, 1, 1, 1, 1, 0, 1, 0, 3, 0, 3, 1,
                                  2, 3, 3, 0, 2, 3, 0, 1, 3, 1, 3, 3});
    Tensor src = tf.make({2, 3, 4}, {2, 1, 0, 9, 3, 1, 1, 0, 3, 6, 6, 7,
                                     9, 6, 3, 4, 5, 0, 8, 2, 8, 2, 7, 5});
    Tensor expected =
        tf.make({2, 3, 4}, {6,  19, 3,  0,  4,  13, 7, 3, 13, 10, 1, 15,
                            10, 9,  17, 15, 14, 10, 9, 3, 6,  11, 1, 24});
    Tensor out = tf.zeros(out_shape, dynamism);

    op_scatter_add_out(input, 2, index, src, out);
    EXPECT_TENSOR_EQ(out, expected);
  }
};

TEST_F(OpScatterAddOutTest, AllValidInputOutputSupport) {
#define TEST_ENTRY(CTYPE, DTYPE) test_scatter_add_out<ScalarType::DTYPE>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpScatterAddOutTest, InfinityAndNANTest) {
  TensorFactory<ScalarType::Long> tf_index;
  TensorFactory<ScalarType::Float> tf_data;
  const std::vector<int32_t> sizes = {3, 5};

  // clang-format off
  Tensor src = tf_data.make(
      /*sizes=*/{2, 5},
      {
        INFINITY, -INFINITY, NAN,       2.33, 3.14,
        NAN,      INFINITY,  -INFINITY, 3.14, 2.33
      });
  // clang-format on
  Tensor self = tf_data.ones(sizes);
  Tensor out = tf_data.zeros(sizes);
  Tensor index = tf_index.make(/*sizes=*/{2, 3}, {0, 1, 2, 0, 1, 2});

  // Valid input should give the expected output
  op_scatter_add_out(self, 0, index, src, out);
  // clang-format off
  EXPECT_TENSOR_CLOSE(
      out,
      tf_data.make(sizes,
      {
        NAN, 1,   1,   1, 1,
        1,   NAN, 1,   1, 1,
        1,   1,   NAN, 1, 1
      }));
  // clang-format on
}

TEST_F(OpScatterAddOutTest, InvalidDimensionsDies) {
#define TEST_ENTRY(CTYPE, DTYPE) \
  test_scatter_add_out_invalid_dim<ScalarType::DTYPE>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpScatterAddOutTest, MismatchedShapeDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched shape";
  }
#define TEST_ENTRY(CTYPE, DTYPE) \
  test_scatter_add_out_mismatched_shape<ScalarType::DTYPE>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpScatterAddOutTest, MismatchedInputDtypesDies) {
  TensorFactory<ScalarType::Byte> tf_byte;
  TensorFactory<ScalarType::Char> tf_char;
  TensorFactory<ScalarType::Long> tf_long;
  const std::vector<int32_t> sizes = {3, 5};
  // clang-format off
  Tensor src = tf_char.make(/*sizes=*/{2, 5},
    {
      1, 2, 3, 4, 5,
      6, 7, 8, 9, 10
    });
  Tensor index = tf_byte.make(/*sizes=*/{2, 3},
    {
      0, 1, 2,
      0, 1, 2
    });
  // clang-format on
  Tensor self = tf_char.zeros(sizes);
  Tensor out = tf_char.zeros(sizes);

  // Types other than long for index should die
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_scatter_add_out(self, 0, index, src, out));

  // Mismatched dtype of src and self should die
  // clang-format off
  src = tf_char.make(/*sizes=*/{2, 5},
    {
      1, 2, 3, 4, 5,
      6, 7, 8, 9, 10
    });
  // clang-format on
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_scatter_add_out(self, 0, index, src, out));
  // clang-format off
  src = tf_byte.make(/*sizes=*/{2, 5},
    {
      1, 2, 3, 4, 5,
      6, 7, 8, 9, 10
    });
  // clang-format on
  self = tf_byte.zeros(sizes);
  out = tf_char.zeros(sizes);

  // Mismatched dtype of self and out should die
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_scatter_add_out(self, 0, index, src, out));
}

TEST_F(OpScatterAddOutTest, DynamicShapeUpperBoundSameAsExpected) {
  test_dynamic_shape(
      {2, 3, 4}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(OpScatterAddOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  test_dynamic_shape(
      {10, 10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(OpScatterAddOutTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  test_dynamic_shape(
      {1, 1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
}
