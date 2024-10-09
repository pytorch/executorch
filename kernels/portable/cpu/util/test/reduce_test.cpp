/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/reduce_util.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/test/utils/DeathTest.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ArrayRef;
using exec_aten::optional;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using executorch::runtime::testing::TensorFactory;
using torch::executor::apply_over_dim;
using torch::executor::apply_over_dim_list;
using torch::executor::get_out_numel;

void _apply_over_dim(const Tensor& in, const optional<int64_t>& dim) {
  int64_t* in_data = in.mutable_data_ptr<int64_t>();
  for (size_t out_ix = 0; out_ix < get_out_numel(in, dim); ++out_ix) {
    apply_over_dim(
        [in_data, out_ix](size_t in_ix, size_t _) { in_data[in_ix] = out_ix; },
        in,
        dim,
        out_ix);
  }
}

void _apply_over_dim_list(
    const Tensor& in,
    const optional<ArrayRef<int64_t>>& dim_list) {
  int64_t* in_data = in.mutable_data_ptr<int64_t>();
  for (size_t out_ix = 0; out_ix < get_out_numel(in, dim_list); ++out_ix) {
    apply_over_dim_list(
        [in_data, out_ix](size_t in_ix) { in_data[in_ix] = out_ix; },
        in,
        dim_list,
        out_ix);
  }
}

TEST(ReduceUtilTest, ApplyOverDim) {
  TensorFactory<ScalarType::Long> tf;
  optional<ArrayRef<int64_t>> dim_list;

  Tensor in = tf.zeros({2, 4, 5, 3});
  _apply_over_dim(in, 0);
  // clang-format off
  EXPECT_TENSOR_EQ(in, tf.make({2, 4, 5, 3}, {
     0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
    15, 16, 17,   18, 19, 20,   21, 22, 23,   24, 25, 26,   27, 28, 29,
    30, 31, 32,   33, 34, 35,   36, 37, 38,   39, 40, 41,   42, 43, 44,
    45, 46, 47,   48, 49, 50,   51, 52, 53,   54, 55, 56,   57, 58, 59,

     0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
    15, 16, 17,   18, 19, 20,   21, 22, 23,   24, 25, 26,   27, 28, 29,
    30, 31, 32,   33, 34, 35,   36, 37, 38,   39, 40, 41,   42, 43, 44,
    45, 46, 47,   48, 49, 50,   51, 52, 53,   54, 55, 56,   57, 58, 59,
  }));
  // clang-format on

  in = tf.zeros({2, 4, 5, 3});
  _apply_over_dim(in, 1);
  // clang-format off
  EXPECT_TENSOR_EQ(in, tf.make({2, 4, 5, 3}, {
     0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
     0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
     0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
     0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,

    15, 16, 17,   18, 19, 20,   21, 22, 23,   24, 25, 26,   27, 28, 29,
    15, 16, 17,   18, 19, 20,   21, 22, 23,   24, 25, 26,   27, 28, 29,
    15, 16, 17,   18, 19, 20,   21, 22, 23,   24, 25, 26,   27, 28, 29,
    15, 16, 17,   18, 19, 20,   21, 22, 23,   24, 25, 26,   27, 28, 29,
  }));
  // clang-format on

  in = tf.zeros({2, 4, 5, 3});
  int64_t dim_array_2[1] = {2};
  dim_list = optional<ArrayRef<int64_t>>(ArrayRef<int64_t>{dim_array_2, 1});
  _apply_over_dim(in, 2);
  // clang-format off
  EXPECT_TENSOR_EQ(in, tf.make({2, 4, 5, 3}, {
     0,  1,  2,    0,  1,  2,    0,  1,  2,    0,  1,  2,    0,  1,  2,
     3,  4,  5,    3,  4,  5,    3,  4,  5,    3,  4,  5,    3,  4,  5,
     6,  7,  8,    6,  7,  8,    6,  7,  8,    6,  7,  8,    6,  7,  8,
     9, 10, 11,    9, 10, 11,    9, 10, 11,    9, 10, 11,    9, 10, 11,

    12, 13, 14,   12, 13, 14,   12, 13, 14,   12, 13, 14,   12, 13, 14,
    15, 16, 17,   15, 16, 17,   15, 16, 17,   15, 16, 17,   15, 16, 17,
    18, 19, 20,   18, 19, 20,   18, 19, 20,   18, 19, 20,   18, 19, 20,
    21, 22, 23,   21, 22, 23,   21, 22, 23,   21, 22, 23,   21, 22, 23,
  }));
  // clang-format on

  in = tf.zeros({2, 4, 5, 3});
  int64_t dim_array_3[1] = {3};
  dim_list = optional<ArrayRef<int64_t>>(ArrayRef<int64_t>{dim_array_3, 1});
  _apply_over_dim(in, 3);
  // clang-format off
  EXPECT_TENSOR_EQ(in, tf.make({2, 4, 5, 3}, {
     0,  0,  0,    1,  1,  1,    2,  2,  2,    3,  3,  3,    4,  4,  4,
     5,  5,  5,    6,  6,  6,    7,  7,  7,    8,  8,  8,    9,  9,  9,
    10, 10, 10,   11, 11, 11,   12, 12, 12,   13, 13, 13,   14, 14, 14,
    15, 15, 15,   16, 16, 16,   17, 17, 17,   18, 18, 18,   19, 19, 19,

    20, 20, 20,   21, 21, 21,   22, 22, 22,   23, 23, 23,   24, 24, 24,
    25, 25, 25,   26, 26, 26,   27, 27, 27,   28, 28, 28,   29, 29, 29,
    30, 30, 30,   31, 31, 31,   32, 32, 32,   33, 33, 33,   34, 34, 34,
    35, 35, 35,   36, 36, 36,   37, 37, 37,   38, 38, 38,   39, 39, 39,
  }));
  // clang-format on
}

TEST(ReduceUtilTest, ApplyOverDimListNull) {
  TensorFactory<ScalarType::Long> tf;
  optional<ArrayRef<int64_t>> null_dim_list;

  Tensor in = tf.ones({2, 4, 5, 3});
  _apply_over_dim_list(in, null_dim_list);
  EXPECT_TENSOR_EQ(in, tf.zeros({2, 4, 5, 3}));
}

TEST(ReduceUtilTest, ApplyOverZeroDimListEmpty) {
  TensorFactory<ScalarType::Long> tf;
  optional<ArrayRef<int64_t>> null_dim_list;

  Tensor in = tf.ones({});
  _apply_over_dim_list(in, null_dim_list);
  EXPECT_TENSOR_EQ(in, tf.zeros({}));
}

TEST(ReduceUtilTest, ApplyOverZeroDim) {
  TensorFactory<ScalarType::Long> tf;
  optional<ArrayRef<int64_t>> dim_list;
  int64_t dim_array_0[1] = {0};
  dim_list = optional<ArrayRef<int64_t>>(ArrayRef<int64_t>{dim_array_0, 1});

  Tensor in = tf.ones({});
  _apply_over_dim_list(in, dim_list);
  EXPECT_TENSOR_EQ(in, tf.zeros({}));
}

TEST(ReduceUtilTest, ApplyOverDimListEmpty) {
  TensorFactory<ScalarType::Long> tf;
  optional<ArrayRef<int64_t>> empty_dim_list{ArrayRef<int64_t>{}};

  Tensor in = tf.ones({2, 4, 5, 3});
  _apply_over_dim_list(in, empty_dim_list);
  EXPECT_TENSOR_EQ(in, tf.zeros({2, 4, 5, 3}));
}

TEST(ReduceUtilTest, ApplyOverDimListLength1) {
  TensorFactory<ScalarType::Long> tf;
  optional<ArrayRef<int64_t>> dim_list;

  Tensor in = tf.zeros({2, 4, 5, 3});
  int64_t dim_array_0[1] = {0};
  dim_list = optional<ArrayRef<int64_t>>(ArrayRef<int64_t>{dim_array_0, 1});
  _apply_over_dim_list(in, dim_list);
  // clang-format off
  EXPECT_TENSOR_EQ(in, tf.make({2, 4, 5, 3}, {
     0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
    15, 16, 17,   18, 19, 20,   21, 22, 23,   24, 25, 26,   27, 28, 29,
    30, 31, 32,   33, 34, 35,   36, 37, 38,   39, 40, 41,   42, 43, 44,
    45, 46, 47,   48, 49, 50,   51, 52, 53,   54, 55, 56,   57, 58, 59,

     0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
    15, 16, 17,   18, 19, 20,   21, 22, 23,   24, 25, 26,   27, 28, 29,
    30, 31, 32,   33, 34, 35,   36, 37, 38,   39, 40, 41,   42, 43, 44,
    45, 46, 47,   48, 49, 50,   51, 52, 53,   54, 55, 56,   57, 58, 59,
  }));
  // clang-format on

  in = tf.zeros({2, 4, 5, 3});
  int64_t dim_array_1[1] = {1};
  dim_list = optional<ArrayRef<int64_t>>(ArrayRef<int64_t>{dim_array_1, 1});
  _apply_over_dim_list(in, dim_list);
  // clang-format off
  EXPECT_TENSOR_EQ(in, tf.make({2, 4, 5, 3}, {
     0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
     0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
     0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
     0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,

    15, 16, 17,   18, 19, 20,   21, 22, 23,   24, 25, 26,   27, 28, 29,
    15, 16, 17,   18, 19, 20,   21, 22, 23,   24, 25, 26,   27, 28, 29,
    15, 16, 17,   18, 19, 20,   21, 22, 23,   24, 25, 26,   27, 28, 29,
    15, 16, 17,   18, 19, 20,   21, 22, 23,   24, 25, 26,   27, 28, 29,
  }));
  // clang-format on

  in = tf.zeros({2, 4, 5, 3});
  int64_t dim_array_2[1] = {2};
  dim_list = optional<ArrayRef<int64_t>>(ArrayRef<int64_t>{dim_array_2, 1});
  _apply_over_dim_list(in, dim_list);
  // clang-format off
  EXPECT_TENSOR_EQ(in, tf.make({2, 4, 5, 3}, {
     0,  1,  2,    0,  1,  2,    0,  1,  2,    0,  1,  2,    0,  1,  2,
     3,  4,  5,    3,  4,  5,    3,  4,  5,    3,  4,  5,    3,  4,  5,
     6,  7,  8,    6,  7,  8,    6,  7,  8,    6,  7,  8,    6,  7,  8,
     9, 10, 11,    9, 10, 11,    9, 10, 11,    9, 10, 11,    9, 10, 11,

    12, 13, 14,   12, 13, 14,   12, 13, 14,   12, 13, 14,   12, 13, 14,
    15, 16, 17,   15, 16, 17,   15, 16, 17,   15, 16, 17,   15, 16, 17,
    18, 19, 20,   18, 19, 20,   18, 19, 20,   18, 19, 20,   18, 19, 20,
    21, 22, 23,   21, 22, 23,   21, 22, 23,   21, 22, 23,   21, 22, 23,
  }));
  // clang-format on

  in = tf.zeros({2, 4, 5, 3});
  int64_t dim_array_3[1] = {3};
  dim_list = optional<ArrayRef<int64_t>>(ArrayRef<int64_t>{dim_array_3, 1});
  _apply_over_dim_list(in, dim_list);
  // clang-format off
  EXPECT_TENSOR_EQ(in, tf.make({2, 4, 5, 3}, {
     0,  0,  0,    1,  1,  1,    2,  2,  2,    3,  3,  3,    4,  4,  4,
     5,  5,  5,    6,  6,  6,    7,  7,  7,    8,  8,  8,    9,  9,  9,
    10, 10, 10,   11, 11, 11,   12, 12, 12,   13, 13, 13,   14, 14, 14,
    15, 15, 15,   16, 16, 16,   17, 17, 17,   18, 18, 18,   19, 19, 19,

    20, 20, 20,   21, 21, 21,   22, 22, 22,   23, 23, 23,   24, 24, 24,
    25, 25, 25,   26, 26, 26,   27, 27, 27,   28, 28, 28,   29, 29, 29,
    30, 30, 30,   31, 31, 31,   32, 32, 32,   33, 33, 33,   34, 34, 34,
    35, 35, 35,   36, 36, 36,   37, 37, 37,   38, 38, 38,   39, 39, 39,
  }));
  // clang-format on
}

TEST(ReduceUtilTest, ApplyOverDimListLength2) {
  TensorFactory<ScalarType::Long> tf;
  optional<ArrayRef<int64_t>> dim_list;

  Tensor in = tf.zeros({2, 4, 5, 3});
  int64_t dim_array_01[2] = {0, 1};
  dim_list = optional<ArrayRef<int64_t>>(ArrayRef<int64_t>{dim_array_01, 2});
  _apply_over_dim_list(in, dim_list);
  // clang-format off
  EXPECT_TENSOR_EQ(in, tf.make({2, 4, 5, 3}, {
     0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
     0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
     0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
     0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,

     0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
     0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
     0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
     0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
  }));
  // clang-format on

  in = tf.zeros({2, 4, 5, 3});
  int64_t dim_array_02[2] = {0, 2};
  dim_list = optional<ArrayRef<int64_t>>(ArrayRef<int64_t>{dim_array_02, 2});
  _apply_over_dim_list(in, dim_list);
  // clang-format off
  EXPECT_TENSOR_EQ(in, tf.make({2, 4, 5, 3}, {
     0,  1,  2,    0,  1,  2,    0,  1,  2,    0,  1,  2,    0,  1,  2,
     3,  4,  5,    3,  4,  5,    3,  4,  5,    3,  4,  5,    3,  4,  5,
     6,  7,  8,    6,  7,  8,    6,  7,  8,    6,  7,  8,    6,  7,  8,
     9, 10, 11,    9, 10, 11,    9, 10, 11,    9, 10, 11,    9, 10, 11,

     0,  1,  2,    0,  1,  2,    0,  1,  2,    0,  1,  2,    0,  1,  2,
     3,  4,  5,    3,  4,  5,    3,  4,  5,    3,  4,  5,    3,  4,  5,
     6,  7,  8,    6,  7,  8,    6,  7,  8,    6,  7,  8,    6,  7,  8,
     9, 10, 11,    9, 10, 11,    9, 10, 11,    9, 10, 11,    9, 10, 11,
  }));
  // clang-format on

  in = tf.zeros({2, 4, 5, 3});
  int64_t dim_array_03[2] = {0, 3};
  dim_list = optional<ArrayRef<int64_t>>(ArrayRef<int64_t>{dim_array_03, 2});
  _apply_over_dim_list(in, dim_list);
  // clang-format off
  EXPECT_TENSOR_EQ(in, tf.make({2, 4, 5, 3}, {
     0,  0,  0,    1,  1,  1,    2,  2,  2,    3,  3,  3,    4,  4,  4,
     5,  5,  5,    6,  6,  6,    7,  7,  7,    8,  8,  8,    9,  9,  9,
    10, 10, 10,   11, 11, 11,   12, 12, 12,   13, 13, 13,   14, 14, 14,
    15, 15, 15,   16, 16, 16,   17, 17, 17,   18, 18, 18,   19, 19, 19,

     0,  0,  0,    1,  1,  1,    2,  2,  2,    3,  3,  3,    4,  4,  4,
     5,  5,  5,    6,  6,  6,    7,  7,  7,    8,  8,  8,    9,  9,  9,
    10, 10, 10,   11, 11, 11,   12, 12, 12,   13, 13, 13,   14, 14, 14,
    15, 15, 15,   16, 16, 16,   17, 17, 17,   18, 18, 18,   19, 19, 19,
  }));
  // clang-format on

  in = tf.zeros({2, 4, 5, 3});
  int64_t dim_array_12[2] = {1, 2};
  dim_list = optional<ArrayRef<int64_t>>(ArrayRef<int64_t>{dim_array_12, 2});
  _apply_over_dim_list(in, dim_list);
  // clang-format off
  EXPECT_TENSOR_EQ(in, tf.make({2, 4, 5, 3}, {
     0,  1,  2,    0,  1,  2,    0,  1,  2,    0,  1,  2,    0,  1,  2,
     0,  1,  2,    0,  1,  2,    0,  1,  2,    0,  1,  2,    0,  1,  2,
     0,  1,  2,    0,  1,  2,    0,  1,  2,    0,  1,  2,    0,  1,  2,
     0,  1,  2,    0,  1,  2,    0,  1,  2,    0,  1,  2,    0,  1,  2,

     3,  4,  5,    3,  4,  5,    3,  4,  5,    3,  4,  5,    3,  4,  5,
     3,  4,  5,    3,  4,  5,    3,  4,  5,    3,  4,  5,    3,  4,  5,
     3,  4,  5,    3,  4,  5,    3,  4,  5,    3,  4,  5,    3,  4,  5,
     3,  4,  5,    3,  4,  5,    3,  4,  5,    3,  4,  5,    3,  4,  5,
  }));
  // clang-format on

  in = tf.zeros({2, 4, 5, 3});
  int64_t dim_array_13[2] = {1, 3};
  dim_list = optional<ArrayRef<int64_t>>(ArrayRef<int64_t>{dim_array_13, 2});
  _apply_over_dim_list(in, dim_list);
  // clang-format off
  EXPECT_TENSOR_EQ(in, tf.make({2, 4, 5, 3}, {
     0,  0,  0,    1,  1,  1,    2,  2,  2,    3,  3,  3,    4,  4,  4,
     0,  0,  0,    1,  1,  1,    2,  2,  2,    3,  3,  3,    4,  4,  4,
     0,  0,  0,    1,  1,  1,    2,  2,  2,    3,  3,  3,    4,  4,  4,
     0,  0,  0,    1,  1,  1,    2,  2,  2,    3,  3,  3,    4,  4,  4,

     5,  5,  5,    6,  6,  6,    7,  7,  7,    8,  8,  8,    9,  9,  9,
     5,  5,  5,    6,  6,  6,    7,  7,  7,    8,  8,  8,    9,  9,  9,
     5,  5,  5,    6,  6,  6,    7,  7,  7,    8,  8,  8,    9,  9,  9,
     5,  5,  5,    6,  6,  6,    7,  7,  7,    8,  8,  8,    9,  9,  9,
  }));
  // clang-format on

  in = tf.zeros({2, 4, 5, 3});
  int64_t dim_array_23[2] = {2, 3};
  dim_list = optional<ArrayRef<int64_t>>(ArrayRef<int64_t>{dim_array_23, 2});
  _apply_over_dim_list(in, dim_list);
  // clang-format off
  EXPECT_TENSOR_EQ(in, tf.make({2, 4, 5, 3}, {
    0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0,
    1, 1, 1,   1, 1, 1,   1, 1, 1,   1, 1, 1,   1, 1, 1,
    2, 2, 2,   2, 2, 2,   2, 2, 2,   2, 2, 2,   2, 2, 2,
    3, 3, 3,   3, 3, 3,   3, 3, 3,   3, 3, 3,   3, 3, 3,

    4, 4, 4,   4, 4, 4,   4, 4, 4,   4, 4, 4,   4, 4, 4,
    5, 5, 5,   5, 5, 5,   5, 5, 5,   5, 5, 5,   5, 5, 5,
    6, 6, 6,   6, 6, 6,   6, 6, 6,   6, 6, 6,   6, 6, 6,
    7, 7, 7,   7, 7, 7,   7, 7, 7,   7, 7, 7,   7, 7, 7,
  }));
  // clang-format on
}

TEST(ReduceUtilTest, ApplyOverDimListLength3) {
  TensorFactory<ScalarType::Long> tf;
  optional<ArrayRef<int64_t>> dim_list;

  Tensor in = tf.zeros({2, 4, 5, 3});
  int64_t dim_array_012[3] = {0, 1, 2};
  dim_list = optional<ArrayRef<int64_t>>(ArrayRef<int64_t>{dim_array_012, 3});
  _apply_over_dim_list(in, dim_list);
  // clang-format off
  EXPECT_TENSOR_EQ(in, tf.make({2, 4, 5, 3}, {
    0, 1, 2,   0, 1, 2,   0, 1, 2,   0, 1, 2,   0, 1, 2,
    0, 1, 2,   0, 1, 2,   0, 1, 2,   0, 1, 2,   0, 1, 2,
    0, 1, 2,   0, 1, 2,   0, 1, 2,   0, 1, 2,   0, 1, 2,
    0, 1, 2,   0, 1, 2,   0, 1, 2,   0, 1, 2,   0, 1, 2,

    0, 1, 2,   0, 1, 2,   0, 1, 2,   0, 1, 2,   0, 1, 2,
    0, 1, 2,   0, 1, 2,   0, 1, 2,   0, 1, 2,   0, 1, 2,
    0, 1, 2,   0, 1, 2,   0, 1, 2,   0, 1, 2,   0, 1, 2,
    0, 1, 2,   0, 1, 2,   0, 1, 2,   0, 1, 2,   0, 1, 2,
  }));
  // clang-format on

  in = tf.zeros({2, 4, 5, 3});
  int64_t dim_array_013[3] = {0, 1, 3};
  dim_list = optional<ArrayRef<int64_t>>(ArrayRef<int64_t>{dim_array_013, 3});
  _apply_over_dim_list(in, dim_list);
  // clang-format off
  EXPECT_TENSOR_EQ(in, tf.make({2, 4, 5, 3}, {
    0, 0, 0,   1, 1, 1,   2, 2, 2,   3, 3, 3,   4, 4, 4,
    0, 0, 0,   1, 1, 1,   2, 2, 2,   3, 3, 3,   4, 4, 4,
    0, 0, 0,   1, 1, 1,   2, 2, 2,   3, 3, 3,   4, 4, 4,
    0, 0, 0,   1, 1, 1,   2, 2, 2,   3, 3, 3,   4, 4, 4,

    0, 0, 0,   1, 1, 1,   2, 2, 2,   3, 3, 3,   4, 4, 4,
    0, 0, 0,   1, 1, 1,   2, 2, 2,   3, 3, 3,   4, 4, 4,
    0, 0, 0,   1, 1, 1,   2, 2, 2,   3, 3, 3,   4, 4, 4,
    0, 0, 0,   1, 1, 1,   2, 2, 2,   3, 3, 3,   4, 4, 4,
  }));
  // clang-format on

  in = tf.zeros({2, 4, 5, 3});
  int64_t dim_array_023[3] = {0, 2, 3};
  dim_list = optional<ArrayRef<int64_t>>(ArrayRef<int64_t>{dim_array_023, 3});
  _apply_over_dim_list(in, dim_list);
  // clang-format off
  EXPECT_TENSOR_EQ(in, tf.make({2, 4, 5, 3}, {
    0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0,
    1, 1, 1,   1, 1, 1,   1, 1, 1,   1, 1, 1,   1, 1, 1,
    2, 2, 2,   2, 2, 2,   2, 2, 2,   2, 2, 2,   2, 2, 2,
    3, 3, 3,   3, 3, 3,   3, 3, 3,   3, 3, 3,   3, 3, 3,

    0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0,
    1, 1, 1,   1, 1, 1,   1, 1, 1,   1, 1, 1,   1, 1, 1,
    2, 2, 2,   2, 2, 2,   2, 2, 2,   2, 2, 2,   2, 2, 2,
    3, 3, 3,   3, 3, 3,   3, 3, 3,   3, 3, 3,   3, 3, 3,
  }));
  // clang-format on

  in = tf.zeros({2, 4, 5, 3});
  int64_t dim_array_123[3] = {1, 2, 3};
  dim_list = optional<ArrayRef<int64_t>>(ArrayRef<int64_t>{dim_array_123, 3});
  _apply_over_dim_list(in, dim_list);
  // clang-format off
  EXPECT_TENSOR_EQ(in, tf.make({2, 4, 5, 3}, {
    0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0,
    0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0,
    0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0,
    0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0,

    1, 1, 1,   1, 1, 1,   1, 1, 1,   1, 1, 1,   1, 1, 1,
    1, 1, 1,   1, 1, 1,   1, 1, 1,   1, 1, 1,   1, 1, 1,
    1, 1, 1,   1, 1, 1,   1, 1, 1,   1, 1, 1,   1, 1, 1,
    1, 1, 1,   1, 1, 1,   1, 1, 1,   1, 1, 1,   1, 1, 1,
  }));
  // clang-format on
}

TEST(ReduceUtilTest, ApplyOverDimListLength4) {
  TensorFactory<ScalarType::Long> tf;
  optional<ArrayRef<int64_t>> dim_list;

  Tensor in = tf.ones({2, 4, 5, 3});
  int64_t dim_array_0123[4] = {0, 1, 2, 3};
  dim_list = optional<ArrayRef<int64_t>>(ArrayRef<int64_t>{dim_array_0123, 4});
  _apply_over_dim_list(in, dim_list);
  EXPECT_TENSOR_EQ(in, tf.zeros({2, 4, 5, 3}));
}

TEST(ReduceUtilTest, ApplyOnZeroDimTensorOverDim) {
  TensorFactory<ScalarType::Long> tf;

  Tensor in = tf.ones({});
  _apply_over_dim(in, 0);
  EXPECT_TENSOR_EQ(in, tf.make({}, {0}));
}

TEST(ReduceUtilTest, ApplyOnZeroDimTensorOverDimListNull) {
  TensorFactory<ScalarType::Long> tf;
  optional<ArrayRef<int64_t>> null_dim_list;

  Tensor in = tf.ones({});
  _apply_over_dim_list(in, null_dim_list);
  EXPECT_TENSOR_EQ(in, tf.make({}, {0}));
}

TEST(ReduceUtilTest, ApplyOnZeroDimTensorOverDimListEmpty) {
  TensorFactory<ScalarType::Long> tf;
  optional<ArrayRef<int64_t>> empty_dim_list{ArrayRef<int64_t>{}};

  Tensor in = tf.ones({});
  _apply_over_dim_list(in, empty_dim_list);
  EXPECT_TENSOR_EQ(in, tf.make({}, {0}));
}

TEST(ReduceUtilTest, ApplyOnZeroDimTensorOverDimListNonEmpty) {
  TensorFactory<ScalarType::Long> tf;
  int64_t dim_array_0[1] = {0};
  optional<ArrayRef<int64_t>> dim_list =
      optional<ArrayRef<int64_t>>(ArrayRef<int64_t>{dim_array_0, 1});

  Tensor in = tf.ones({});
  _apply_over_dim_list(in, dim_list), "";
  EXPECT_TENSOR_EQ(in, tf.make({}, {0}));
}

TEST(ReduceUtilTest, ApplyOnEmptyTensorOverDim) {
  TensorFactory<ScalarType::Long> tf;
  optional<ArrayRef<int64_t>> dim_list;

  Tensor in = tf.zeros({2, 0, 5, 3});
  Tensor out = tf.zeros({2, 5, 3});

  // dim = 1
  int64_t dim = 1;
  EXPECT_TRUE(in.numel() == 0);
  EXPECT_TRUE(out.numel() == 30 && out.numel() == get_out_numel(in, dim));

  int64_t* in_data = in.mutable_data_ptr<int64_t>();
  int64_t* out_data = out.mutable_data_ptr<int64_t>();
  for (size_t out_ix = 0; out_ix < get_out_numel(in, dim); ++out_ix) {
    out_data[out_ix] = 1;
    apply_over_dim(
        [in_data, out_data, out_ix](size_t in_ix, size_t _) {
          in_data[in_ix] = out_ix; // Should be ignored.
          out_data[out_ix] = 2; // Should be ignored.
        },
        in,
        dim,
        out_ix);
  }
  EXPECT_TENSOR_EQ(out, tf.ones({2, 5, 3}));

  // dim = 0
  dim = 0;
  EXPECT_TRUE(in.numel() == 0);
  EXPECT_TRUE(get_out_numel(in, dim) == 0);
  // Should die if called on empty tensor with dim that also produces
  // empty tensor, because out_ix will be out of bounds
  ET_EXPECT_DEATH(
      apply_over_dim([](size_t in_ix, size_t _) { return; }, in, dim, 0), "");
}

TEST(ReduceUtilTest, ApplyOnEmptyTensorOverDimList) {
  TensorFactory<ScalarType::Long> tf;
  optional<ArrayRef<int64_t>> dim_list;

  Tensor in = tf.zeros({2, 0, 5, 3});
  Tensor out = tf.zeros({5, 3});

  // dim list = {0, 1}
  int64_t dim_array_01[2] = {0, 1};
  dim_list = optional<ArrayRef<int64_t>>(ArrayRef<int64_t>{dim_array_01, 2});

  EXPECT_TRUE(in.numel() == 0);
  EXPECT_TRUE(out.numel() == 15 && out.numel() == get_out_numel(in, dim_list));

  int64_t* in_data = in.mutable_data_ptr<int64_t>();
  int64_t* out_data = out.mutable_data_ptr<int64_t>();
  for (size_t out_ix = 0; out_ix < get_out_numel(in, dim_list); ++out_ix) {
    out_data[out_ix] = 1;
    apply_over_dim_list(
        [in_data, out_data, out_ix](size_t in_ix) {
          in_data[in_ix] = out_ix; // Should be ignored.
          out_data[out_ix] = 2; // Should be ignored.
        },
        in,
        dim_list,
        out_ix);
  }
  EXPECT_TENSOR_EQ(out, tf.ones({5, 3}));

  // dim list = {0, 2}
  int64_t dim_array_02[2] = {0, 2};
  dim_list = optional<ArrayRef<int64_t>>(ArrayRef<int64_t>{dim_array_02, 2});

  EXPECT_TRUE(in.numel() == 0);
  EXPECT_TRUE(get_out_numel(in, dim_list) == 0);
  // Should die if called on empty tensor with dim list that also produces
  // empty tensor, because out_ix will be out of bounds
  ET_EXPECT_DEATH(
      apply_over_dim_list([](size_t in_ix) { return; }, in, dim_list, 0), "");
}

TEST(ReduceUtilTest, ApplyOverDimListInvalid) {
  TensorFactory<ScalarType::Long> tf;
  optional<ArrayRef<int64_t>> dim_list;

  Tensor in = tf.zeros({2, 4, 5, 3});
  int64_t dim_array_09[2] = {0, 9};
  dim_list = optional<ArrayRef<int64_t>>(ArrayRef<int64_t>{dim_array_09, 2});

  ET_EXPECT_DEATH(
      apply_over_dim_list([](size_t in_ix) { return; }, in, dim_list, 0), "");

  int64_t dim_array_neg[3] = {0, -5, 3};
  dim_list = optional<ArrayRef<int64_t>>(ArrayRef<int64_t>{dim_array_neg, 3});

  ET_EXPECT_DEATH(
      apply_over_dim_list([](size_t in_ix) { return; }, in, dim_list, 0), "");

  int64_t dim_array_011[3] = {0, 1, 1};
  dim_list = optional<ArrayRef<int64_t>>(ArrayRef<int64_t>{dim_array_011, 3});

  ET_EXPECT_DEATH(
      apply_over_dim_list([](size_t in_ix) { return; }, in, dim_list, 0), "");

  int64_t dim_array_1_3[2] = {1, -3};
  dim_list = optional<ArrayRef<int64_t>>(ArrayRef<int64_t>{dim_array_1_3, 2});

  ET_EXPECT_DEATH(
      apply_over_dim_list([](size_t in_ix) { return; }, in, dim_list, 0), "");
}
