/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/test/utils/DeathTest.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using executorch::runtime::ArrayRef;
using executorch::runtime::testing::TensorFactory;
using torch::executor::broadcast_tensor;
using torch::executor::delinearize_index;
using torch::executor::get_broadcast_target_size;
using torch::executor::linearize_access_indexes;
using torch::executor::tensor_is_broadcastable_to;
using torch::executor::tensors_are_broadcastable_between;

TEST(BroadcastUtilTest, BroadcastTensor) {
  TensorFactory<ScalarType::Int> tf;

  Tensor a = tf.make({1}, {2});
  Tensor b = tf.make({2, 2}, {2, 2, 2, 2});
  Tensor c = tf.zeros({2, 2});

  Tensor d = torch::executor::broadcast_tensor(a, c);
  EXPECT_TENSOR_DATA_EQ(d, tf.make({2, 2}, {2, 2, 2, 2}));
  torch::executor::free_broadcast_tensor(d);

  d = torch::executor::broadcast_tensor(b, c);
  EXPECT_TENSOR_DATA_EQ(d, tf.make({2, 2}, {2, 2, 2, 2}));
  torch::executor::free_broadcast_tensor(d);
}

TEST(BroadcastUtilTest, BroadcastableBetween) {
  TensorFactory<ScalarType::Int> tf;

  std::vector<Tensor> tensor_list = {
      tf.zeros({1, 2}), tf.zeros({2, 1}), tf.zeros({1}), tf.zeros({2, 2})};

  for (int i = 0; i < 4; i++) {
    for (int j = i + 1; j < 4; j++) {
      EXPECT_TRUE(
          tensors_are_broadcastable_between(tensor_list[i], tensor_list[j]));
    }
  }
}

TEST(BroadcastUtilTest, BroadcastableToFrom) {
  TensorFactory<ScalarType::Int> tf;

  Tensor a = tf.make({1, 2}, {2, 2});
  Tensor b = tf.make({2, 1}, {2, 2});
  Tensor c = tf.zeros({2, 2});

  ASSERT_TRUE(tensor_is_broadcastable_to(a, c));
  Tensor d = torch::executor::broadcast_tensor(a, c);
  EXPECT_TENSOR_DATA_EQ(d, tf.make({2, 2}, {2, 2, 2, 2}));
  torch::executor::free_broadcast_tensor(d);

  ASSERT_TRUE(tensor_is_broadcastable_to(b, c));
  d = torch::executor::broadcast_tensor(b, c);
  EXPECT_TENSOR_DATA_EQ(d, tf.make({2, 2}, {2, 2, 2, 2}));
  torch::executor::free_broadcast_tensor(d);
}

TEST(BroadcastUtilTest, NotBroadcastableTo) {
  TensorFactory<ScalarType::Int> tf;

  // Tensor a is broadcastable to tensor b means when tracing their sizes from
  // back to front, each pair of corresponding dimensions should meet one of the
  // following conditions:
  // 1. the two dimensions are equal;
  // 2. a's dimension is 1;
  // 3. one of the dimensions does not exist.
  Tensor a = tf.make({3}, {2, 2, 2});
  Tensor b = tf.zeros({2, 1});
  Tensor c = tf.zeros({1, 2});

  ASSERT_FALSE(tensor_is_broadcastable_to(a, b));
  ET_EXPECT_DEATH(broadcast_tensor(a, b), "");

  // Can not broadcast from b to c, though they are broadcastable.
  // When broadcasting, b and c should be broadcasted to a new size (2, 2).
  // Neither of them can be broadcasted to each other's size.
  ASSERT_FALSE(tensor_is_broadcastable_to(b, c));
  ET_EXPECT_DEATH(broadcast_tensor(b, c), "");
}

TEST(BroadcastUtilTest, NotBroadcastableBetween) {
  TensorFactory<ScalarType::Int> tf;

  Tensor a = tf.make({3}, {2, 2, 2});
  Tensor b = tf.zeros({2, 1});

  EXPECT_FALSE(tensor_is_broadcastable_to(a, b));
}

TEST(BroadcastUtilTest, GetBroadcastTargetSize) {
  TensorFactory<ScalarType::Int> tf;
  Tensor::SizesType
      expected_output_size[torch::executor::kTensorDimensionLimit] = {};
  size_t expected_output_dim = 0;

  Tensor a = tf.zeros({2, 1});
  Tensor b = tf.zeros({5, 1, 2});

  executorch::runtime::Error err = get_broadcast_target_size(
      a,
      b,
      expected_output_size,
      torch::executor::kTensorDimensionLimit,
      &expected_output_dim);
  EXPECT_EQ(err, torch::executor::Error::Ok);

  EXPECT_TRUE(
      ArrayRef<Tensor::SizesType>(expected_output_size, expected_output_dim)
          .equals(ArrayRef<Tensor::SizesType>({5, 2, 2})));
}

size_t linearize_indexes(size_t* indexes, size_t indexes_len, const Tensor& t) {
  size_t linear_index = 0;
  size_t acc_loop_counts = 1;
  for (ssize_t i = indexes_len - 1; i >= 0; --i) {
    linear_index += indexes[i] * acc_loop_counts;
    acc_loop_counts *= (size_t)t.sizes()[i];
  }
  return linear_index;
}

TEST(BroadcastUtilTest, DelinearizeIndex) {
  TensorFactory<ScalarType::Int> tf;

  const size_t DIMS = 3;
  Tensor t = tf.zeros({4, 3, 5});
  auto sizes = t.sizes();

  for (size_t i0 = 0; i0 < (size_t)sizes[0]; ++i0) {
    for (size_t i1 = 0; i1 < (size_t)sizes[1]; ++i1) {
      for (size_t i2 = 0; i2 < (size_t)sizes[2]; ++i2) {
        size_t indexes[DIMS] = {i0, i1, i2};
        auto linear_index = linearize_indexes(indexes, DIMS, t);

        size_t out_indexes[DIMS];
        delinearize_index(linear_index, t, out_indexes, DIMS);

        EXPECT_EQ(linear_index, linearize_indexes(out_indexes, DIMS, t));
      }
    }
  }
}

TEST(BroadcastUtilTest, LinearizeIndex) {
  TensorFactory<ScalarType::Int> tf;

  Tensor broadcast_from = tf.zeros({2, 1, 3, 1});
  Tensor broadcast_to = tf.zeros({2, 2, 3, 4});

  // The linear index for brodcast_from should be the same in
  // the brocasted dimension.
  for (size_t i = 0; i < 3; ++i) {
    size_t test_indexes[] = {0, 0, 0, i};
    ArrayRef<size_t> broadcast_to_indexes(test_indexes);
    size_t linear_index = linearize_access_indexes(
        broadcast_to_indexes, broadcast_to.dim(), broadcast_from);
    EXPECT_EQ(linear_index, 0);
  }

  // The linear index for brodcast_from should be the same.
  // the brocasted dimension.
  for (size_t i = 0; i <= 2; ++i) {
    size_t test_indexes[] = {0, i, 2, 3};
    ArrayRef<size_t> broadcast_to_indexes(test_indexes);
    size_t linear_index = linearize_access_indexes(
        broadcast_to_indexes, broadcast_to.dim(), broadcast_from);
    EXPECT_EQ(linear_index, 2);
  }
}
