/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/delinearized_indexes_range.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>

#include <gtest/gtest.h>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::testing::TensorFactory;
using torch::executor::DelinearizedIndexesRange;

TEST(DelinearizedIndexesRangeTest, Empty) {
  TensorFactory<ScalarType::Int> tf;

  Tensor a = tf.make({0}, {});
  ASSERT_EQ(a.numel(), 0);
  bool loop_entered = false;
  for (auto _ : DelinearizedIndexesRange(a)) {
    loop_entered = true;
  }
  EXPECT_FALSE(loop_entered);
}

TEST(DelinearizedIndexesRangeTest, OneD) {
  TensorFactory<ScalarType::Int> tf;

  Tensor a = tf.zeros({5});
  DelinearizedIndexesRange r(a);
  std::vector<typename DelinearizedIndexesRange::iterator::value_type> v(r.begin(), r.end());
  int idx = 0;
  for (const auto& elem: v) {
    EXPECT_EQ(elem[0], idx++);
  }
}

TEST(DelinearizedIndexesRangeTest, ThreeD) {
  TensorFactory<ScalarType::Int> tf;
  Tensor a = tf.zeros({3, 2, 3});
  DelinearizedIndexesRange r(a);
  std::vector<typename DelinearizedIndexesRange::iterator::value_type> v(r.begin(), r.end());
  std::vector<typename DelinearizedIndexesRange::iterator::value_type> expected = {
    {0, 0, 0},
    {0, 0, 1},
    {0, 0, 2},
    {0, 1, 0},
    {0, 1, 1},
    {0, 1, 2},
    {1, 0, 0},
    {1, 0, 1},
    {1, 0, 2},
    {1, 1, 0},
    {1, 1, 1},
    {1, 1, 2},
    {2, 0, 0},
    {2, 0, 1},
    {2, 0, 2},
    {2, 1, 0},
    {2, 1, 1},
    {2, 1, 2},
  };
  EXPECT_EQ(v, expected);
}
