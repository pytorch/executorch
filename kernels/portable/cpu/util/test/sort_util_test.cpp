/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/sort_util.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/test/utils/DeathTest.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::ArrayRef;
using torch::executor::testing::TensorFactory;

TEST(SortUtilTest, SortTensorTest) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Long> lf;

  Tensor a = tf.make({4}, {3, 2, 1, 4});
  Tensor b = tf.zeros({4});
  Tensor c = lf.zeros({4});

  // Ascending order sort test
  sort_tensor(a, b, c);

  Tensor expected = tf.make({4}, {1, 2, 3, 4});
  Tensor expected_indices = lf.make({4}, {2, 1, 0, 3});
  EXPECT_TENSOR_EQ(b, expected);
  EXPECT_TENSOR_EQ(c, expected_indices);

  // Descending order sort test
  sort_tensor(a, b, c, true);
  expected = tf.make({4}, {4, 3, 2, 1});
  expected_indices = lf.make({4}, {3, 0, 1, 2});
  EXPECT_TENSOR_EQ(b, expected);
  EXPECT_TENSOR_EQ(c, expected_indices);
}
