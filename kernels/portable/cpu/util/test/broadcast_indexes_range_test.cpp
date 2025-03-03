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
  for (const auto& elem : range_to_vec(BroadcastIndexesRange<1>(out, out))) {
    EXPECT_EQ(elem[0], idx++);
    EXPECT_EQ(elem[0], elem[1]);
  }
}

// [1] -> [W]
TEST(BroadcastIndexesRangeTest, ScalarBroadcastToOneD) {
  TensorFactory<ScalarType::Int> tf;

  Tensor out = tf.zeros({5});
  Tensor in = tf.zeros({1});

  auto actual = range_to_vec(BroadcastIndexesRange<1>(out, in));
  decltype(actual) expected = {
      {0, 0},
      {1, 0},
      {2, 0},
      {3, 0},
      {4, 0},
  };
  EXPECT_EQ(expected, actual);
}

// [1] -> [H, W]
// [1, W] -> [H, W]
// [H, 1] -> [H, W]
// [H, W] -> [H, W]
// Cover all these at the same time to also exercise multiple input tensors.
TEST(BroadcastIndexesRangeTest, TwoDExhaustive) {
  TensorFactory<ScalarType::Int> tf;
  Tensor out = tf.zeros({3, 4});
  Tensor in_0d_scalar = tf.zeros({});
  Tensor in_1d_scalar = tf.zeros({1});
  Tensor in_2d_scalar = tf.zeros({1, 1});

  Tensor in_row = tf.zeros({4});
  Tensor in_col = tf.zeros({3, 1});

  Tensor in_not_broadcast = tf.zeros({3, 4});

  auto actual = range_to_vec(BroadcastIndexesRange<6>(
      out,
      in_0d_scalar,
      in_1d_scalar,
      in_2d_scalar,
      in_row,
      in_col,
      in_not_broadcast));
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
}

// Here we assume that the previous tests established that padding
// with leading 1s is working, and test:
// [C, H, 1] -> [C, H, W]
// [C, 1, W] -> [C, H, W]
// [C, 1, 1] -> [C, H, W]
// [1, H, 1] -> [C, H, W]
// [C, H, W] -> [C, H, W]
TEST(BroadcastIndexesRangeTest, ThreeDBroadcasting) {
  TensorFactory<ScalarType::Int> tf;
  Tensor out = tf.zeros({2, 3, 4});
  Tensor in_broadcast_w = tf.zeros({2, 3, 1});
  Tensor in_broadcast_h = tf.zeros({2, 1, 4});
  Tensor in_broadcast_hw = tf.zeros({2, 1, 1});
  Tensor in_broadcast_cw = tf.zeros({1, 3, 1});
  Tensor in_not_broadcast = tf.zeros({2, 3, 4});
  auto actual = range_to_vec(BroadcastIndexesRange<5>(
      out,
      in_broadcast_w,
      in_broadcast_h,
      in_broadcast_hw,
      in_broadcast_cw,
      in_not_broadcast));
  decltype(actual) expected = {
      {0, 0, 0, 0, 0, 0},   {1, 0, 1, 0, 0, 1},   {2, 0, 2, 0, 0, 2},
      {3, 0, 3, 0, 0, 3},   {4, 1, 0, 0, 1, 4},   {5, 1, 1, 0, 1, 5},
      {6, 1, 2, 0, 1, 6},   {7, 1, 3, 0, 1, 7},   {8, 2, 0, 0, 2, 8},
      {9, 2, 1, 0, 2, 9},   {10, 2, 2, 0, 2, 10}, {11, 2, 3, 0, 2, 11},
      {12, 3, 4, 1, 0, 12}, {13, 3, 5, 1, 0, 13}, {14, 3, 6, 1, 0, 14},
      {15, 3, 7, 1, 0, 15}, {16, 4, 4, 1, 1, 16}, {17, 4, 5, 1, 1, 17},
      {18, 4, 6, 1, 1, 18}, {19, 4, 7, 1, 1, 19}, {20, 5, 4, 1, 2, 20},
      {21, 5, 5, 1, 2, 21}, {22, 5, 6, 1, 2, 22}, {23, 5, 7, 1, 2, 23},
  };
  EXPECT_EQ(expected, actual);
}

// 4-D should generalize, but we will go ahead and test:
// [N, 1, H, 1] -> [N, C, H, W]
// [1, C, 1, W] -> [N, C, H, W]
TEST(BroadcastIndexesRangeTest, FourDBroadcasting) {
  TensorFactory<ScalarType::Int> tf;
  Tensor out = tf.zeros({2, 3, 4, 5});
  Tensor in_broadcast_cw = tf.zeros({2, 1, 4, 1});
  Tensor in_broadcast_nh = tf.zeros({1, 3, 1, 5});

  int idx = 0;
  // Writing out all the indices would be too cumbersome, so here we
  // take the opportunity to mutation test against delinearize_index
  // and linearize_access_indexes.
  for (const auto [out_idx, in_cw_idx, in_nh_idx] :
       BroadcastIndexesRange<2>(out, in_broadcast_cw, in_broadcast_nh)) {
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
}
