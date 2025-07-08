/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/broadcast_indexes_range.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>

#include <gtest/gtest.h>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::testing::TensorFactory;
using torch::executor::BroadcastIndexesRange;
using torch::executor::delinearize_index;
using torch::executor::linearize_access_indexes;

namespace {
template <typename Range>
auto range_to_vec(const Range& rng) {
  return std::vector<typename Range::iterator::value_type>(
      rng.begin(), rng.end());
}
} // namespace
TEST(BroadcastIndexesRangeTest, Empty) {
  TensorFactory<ScalarType::Int> tf;

  Tensor a = tf.make({0}, {});
  ASSERT_EQ(a.numel(), 0);
  bool loop_entered = false;
  for (auto _ : BroadcastIndexesRange<1>(a, a)) {
    loop_entered = true;
  }
  EXPECT_FALSE(loop_entered);
}

// [W] -> [W]
TEST(BroadcastIndexesRangeTest, OneDNotBroadcasted) {
  TensorFactory<ScalarType::Int> tf;

  Tensor out = tf.zeros({5});
  int idx = 0;
  const auto range = BroadcastIndexesRange<1>(out, out);
  for (const auto& elem : range_to_vec(range)) {
    EXPECT_EQ(*(range.begin() + idx), elem);
    EXPECT_EQ(elem[0], idx++);
    EXPECT_EQ(elem[0], elem[1]);
  }
}

template <typename Range>
void test_operator_plus(const Range& range) {
  size_t idx = 0;
  for (const auto& indexes : range) {
    EXPECT_EQ(*(range.begin() + idx), indexes);
    idx++;
  }
}

// [1] -> [H, W]
// [W] -> [H, W]
// [1, 1] -> [H, W]
// [1, W] -> [H, W]
// [H, 1] -> [H, W]
// [H, W] -> [H, W]
// Cover all these at the same time to also exercise multiple input tensors.
TEST(BroadcastIndexesRangeTest, OneAndTwoDExhaustive) {
  TensorFactory<ScalarType::Int> tf;
  Tensor out = tf.zeros({3, 4});
  Tensor in_0d_scalar = tf.zeros({});
  Tensor in_1d_scalar = tf.zeros({1});
  Tensor in_2d_scalar = tf.zeros({1, 1});

  Tensor in_row = tf.zeros({4});
  Tensor in_col = tf.zeros({3, 1});

  Tensor in_not_broadcast = tf.zeros({3, 4});

  const auto range = BroadcastIndexesRange<6>(
      out,
      in_0d_scalar,
      in_1d_scalar,
      in_2d_scalar,
      in_row,
      in_col,
      in_not_broadcast);
  auto actual = range_to_vec(range);
  decltype(actual) expected = {
      {0, 0, 0, 0, 0, 0, 0},
      {1, 0, 0, 0, 1, 0, 1},
      {2, 0, 0, 0, 2, 0, 2},
      {3, 0, 0, 0, 3, 0, 3},
      {4, 0, 0, 0, 0, 1, 4},
      {5, 0, 0, 0, 1, 1, 5},
      {6, 0, 0, 0, 2, 1, 6},
      {7, 0, 0, 0, 3, 1, 7},
      {8, 0, 0, 0, 0, 2, 8},
      {9, 0, 0, 0, 1, 2, 9},
      {10, 0, 0, 0, 2, 2, 10},
      {11, 0, 0, 0, 3, 2, 11},
  };
  EXPECT_EQ(expected, actual);

  test_operator_plus(range);
}

// Make sure nothing is thrown off by a size-1 dim in the output:
// [] -> [1, W]
// [] -> [H, 1]
// [1] -> [1, W]
// [1] -> [H, 1]
// [W] -> [1, W]
// [1, 1] -> [1, W]
// [1, 1] -> [H, 1]
// [1, W] -> [1, W]
// [H, 1] -> [H, 1]
TEST(BroadcastIndexesRangeTest, OneAndTwoDWith1InOutputShapeExhaustive) {
  TensorFactory<ScalarType::Int> tf;
  constexpr auto H = 2;
  constexpr auto W = 3;
  Tensor out_row = tf.zeros({1, W});
  Tensor out_col = tf.zeros({H, 1});
  Tensor in_0d_scalar = tf.zeros({});
  Tensor in_1d_scalar = tf.zeros({1});
  Tensor in_2d_scalar = tf.zeros({1, 1});

  Tensor in_row = tf.zeros({W});
  Tensor in_leading_one_row = tf.zeros({1, W});

  Tensor in_col = tf.zeros({H, 1});

  size_t idx = 0;
  const auto range_row = BroadcastIndexesRange<5>(
      out_row,
      in_0d_scalar,
      in_1d_scalar,
      in_2d_scalar,
      in_row,
      in_leading_one_row);
  for (const auto
       [out_idx,
        in_0d_idx,
        in_1d_idx,
        in_2d_idx,
        in_row_idx,
        in_leading_one_row_idx] : range_row) {
    EXPECT_EQ(out_idx, idx++);
    EXPECT_EQ(in_0d_idx, 0);
    EXPECT_EQ(in_1d_idx, 0);
    EXPECT_EQ(in_2d_idx, 0);
    EXPECT_EQ(in_row_idx, out_idx);
    EXPECT_EQ(in_leading_one_row_idx, out_idx);
  }

  test_operator_plus(range_row);

  idx = 0;
  const auto range_col = BroadcastIndexesRange<4>(
      out_col, in_0d_scalar, in_1d_scalar, in_2d_scalar, in_col);
  for (const auto [out_idx, in_0d_idx, in_1d_idx, in_2d_idx, in_col_idx] :
       range_col) {
    EXPECT_EQ(out_idx, idx++);
    EXPECT_EQ(in_0d_idx, 0);
    EXPECT_EQ(in_1d_idx, 0);
    EXPECT_EQ(in_2d_idx, 0);
    EXPECT_EQ(in_col_idx, out_idx);
  }

  test_operator_plus(range_col);
}

// [1, 1, 1] -> [C, H, W]
// [C, H, 1] -> [C, H, W]
// [C, 1, W] -> [C, H, W]
// [1, H, W] -> [C, H, W]
// [C, 1, 1] -> [C, H, W]
// [1, H, 1] -> [C, H, W]
// [1, 1, W] -> [C, H, W]
// [C, H, W] -> [C, H, W]
TEST(BroadcastIndexesRangeTest, ThreeDBroadcasting) {
  TensorFactory<ScalarType::Int> tf;
  Tensor out = tf.zeros({2, 3, 4});
  std::array<Tensor, 8> input_tensors = {
      tf.zeros({2, 3, 1}),
      tf.zeros({2, 1, 4}),
      tf.zeros({1, 3, 4}),
      tf.zeros({2, 1, 1}),
      tf.zeros({1, 3, 1}),
      tf.zeros({1, 1, 4}),
      tf.zeros({1, 1, 1}),
      tf.zeros({2, 3, 4}),
  };
  // Writing out all the indexes would be too cumbersome, so here we
  // take the opportunity to mutation test against delinearize_index
  // and linearize_access_indexes.
  int idx = 0;
  const auto range = BroadcastIndexesRange<8>(
      out,
      input_tensors[0],
      input_tensors[1],
      input_tensors[2],
      input_tensors[3],
      input_tensors[4],
      input_tensors[5],
      input_tensors[6],
      input_tensors[7]);
  for (const auto indexes : range) {
    const auto out_idx = indexes[0];
    EXPECT_EQ(out_idx, idx++);
    size_t out_indexes[executorch::runtime::kTensorDimensionLimit];
    delinearize_index(
        out_idx, out, out_indexes, executorch::runtime::kTensorDimensionLimit);
    for (const auto tensor_idx : c10::irange(0, input_tensors.size())) {
      EXPECT_EQ(
          indexes[tensor_idx + 1],
          linearize_access_indexes(
              out_indexes, out.dim(), input_tensors[tensor_idx]));
    }
  }
  test_operator_plus(range);
}

// 4-D should generalize, but we will go ahead and test:
// [N, 1, H, 1] -> [N, C, H, W]
// [1, C, 1, W] -> [N, C, H, W]
template <size_t N, size_t C, size_t H, size_t W>
void four_d_broadcasting_test() {
  TensorFactory<ScalarType::Int> tf;
  Tensor out = tf.zeros({N, C, H, W});
  Tensor in_broadcast_cw = tf.zeros({N, 1, H, 1});
  Tensor in_broadcast_nh = tf.zeros({1, C, 1, W});

  // Writing out all the indexes would be too cumbersome, so here we
  // take the opportunity to mutation test against delinearize_index
  // and linearize_access_indexes.
  int idx = 0;
  const auto range =
      BroadcastIndexesRange<2>(out, in_broadcast_cw, in_broadcast_nh);
  for (const auto [out_idx, in_cw_idx, in_nh_idx] : range) {
    EXPECT_EQ(out_idx, idx++);
    size_t out_indexes[executorch::runtime::kTensorDimensionLimit];
    delinearize_index(
        out_idx, out, out_indexes, executorch::runtime::kTensorDimensionLimit);
    EXPECT_EQ(
        in_cw_idx,
        linearize_access_indexes(out_indexes, out.dim(), in_broadcast_cw));
    EXPECT_EQ(
        in_nh_idx,
        linearize_access_indexes(out_indexes, out.dim(), in_broadcast_nh));
  }

  test_operator_plus(range);
}

TEST(BroadcastIndexesRangeTest, FourDBroadcasting) {
  four_d_broadcasting_test<2, 3, 4, 5>();
}

TEST(BroadcastIndexesRangeTest, FourDBroadcastingWithOneDimsInOutput) {
  four_d_broadcasting_test<2, 3, 1, 5>();
  four_d_broadcasting_test<2, 1, 3, 1>();
}
