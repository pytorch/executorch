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
using executorch::aten::DimOrderType;
using executorch::aten::IntArrayRef;
using executorch::aten::OptionalArrayRef;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using std::optional;
using torch::executor::testing::TensorFactory;

class OpEmptyDimOrderOutTest : public OperatorTest {
 protected:
  Tensor& op_empty_dim_order_out(
      IntArrayRef size,
      OptionalArrayRef<int64_t> dim_order,
      Tensor& out) {
    return torch::executor::dim_order_ops::_empty_dim_order_outf(
        context_, size, dim_order, out);
  }

  template <ScalarType DTYPE>
  void test_op_empty_dim_order_out(std::vector<int32_t>&& size_int32_t) {
    TensorFactory<DTYPE> tf;
    std::vector<int64_t> sizes(size_int32_t.begin(), size_int32_t.end());
    auto aref = executorch::aten::ArrayRef<int64_t>(sizes.data(), sizes.size());
    OptionalArrayRef<int64_t> dim_order;
    Tensor out = tf.ones(size_int32_t);

    op_empty_dim_order_out(aref, dim_order, out);
  }

  void too_short_dim_order_die() {
    TensorFactory<ScalarType::Float> tf;

    int64_t sizes[3] = {3, 2, 4};
    auto sizes_aref = executorch::aten::ArrayRef<int64_t>(sizes);

    int64_t raw_dim_order[2] = {0, 1};
    auto dim_order = OptionalArrayRef<int64_t>(raw_dim_order);
    Tensor out =
        tf.ones({3, 2, 4}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_empty_dim_order_out(sizes_aref, dim_order, out));
  }

  void illegal_dim_order_die() {
    TensorFactory<ScalarType::Float> tf;

    int64_t sizes[2] = {3, 2};
    auto sizes_aref = executorch::aten::ArrayRef<int64_t>(sizes);

    int64_t raw_dim_order[2] = {1, 2};
    auto dim_order = OptionalArrayRef<int64_t>(raw_dim_order);
    Tensor out =
        tf.ones({3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_empty_dim_order_out(sizes_aref, dim_order, out));
  }

  void wrong_dim_order_die() {
    TensorFactory<ScalarType::Float> tf;

    int64_t sizes[4] = {3, 2, 4, 5};
    auto sizes_aref = executorch::aten::ArrayRef<int64_t>(sizes);

    // should be {0, 2, 3, 1}
    int64_t raw_dim_order[4] = {0, 1, 2, 3};
    auto dim_order = OptionalArrayRef<int64_t>(raw_dim_order);
    Tensor out = tf.full_channels_last(
        {3, 2, 4, 5}, 1, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_empty_dim_order_out(sizes_aref, dim_order, out));
  }
};

#define GENERATE_TEST(_, DTYPE)                                \
  TEST_F(OpEmptyDimOrderOutTest, DTYPE##Tensors) {             \
    test_op_empty_dim_order_out<ScalarType::DTYPE>({2, 3, 4}); \
    test_op_empty_dim_order_out<ScalarType::DTYPE>({2, 0, 4}); \
    test_op_empty_dim_order_out<ScalarType::DTYPE>({});        \
  }

ET_FORALL_REAL_TYPES_AND(Bool, GENERATE_TEST)

TEST_F(OpEmptyDimOrderOutTest, DynamicShapeUpperBoundSameAsExpected) {
  TensorFactory<ScalarType::Float> tf;

  int64_t sizes[2] = {3, 2};
  auto sizes_aref = executorch::aten::ArrayRef<int64_t>(sizes);
  OptionalArrayRef<int64_t> dim_order;
  Tensor out =
      tf.ones({3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_empty_dim_order_out(sizes_aref, dim_order, out);
}

TEST_F(OpEmptyDimOrderOutTest, ContiguousDimOrderSuccees) {
  TensorFactory<ScalarType::Float> tf;

  int64_t sizes[2] = {3, 2};
  auto sizes_aref = executorch::aten::ArrayRef<int64_t>(sizes);

  int64_t raw_dim_order[2] = {0, 1};
  auto dim_order = OptionalArrayRef<int64_t>(raw_dim_order);
  Tensor out =
      tf.ones({3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_empty_dim_order_out(sizes_aref, dim_order, out);
}

TEST_F(OpEmptyDimOrderOutTest, ChannelsLastsDimOrderSuccees) {
  TensorFactory<ScalarType::Float> tf;

  int64_t sizes[4] = {3, 2, 4, 5};
  auto sizes_aref = executorch::aten::ArrayRef<int64_t>(sizes);

  int64_t raw_dim_order[4] = {0, 2, 3, 1};
  auto dim_order = OptionalArrayRef<int64_t>(raw_dim_order);
  Tensor out = tf.full_channels_last(
      {3, 2, 4, 5}, 1, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_empty_dim_order_out(sizes_aref, dim_order, out);
}

TEST_F(OpEmptyDimOrderOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  TensorFactory<ScalarType::Float> tf;

  int64_t sizes[2] = {3, 2};
  auto sizes_aref = executorch::aten::ArrayRef<int64_t>(sizes);
  OptionalArrayRef<int64_t> dim_order;
  Tensor out =
      tf.ones({10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_empty_dim_order_out(sizes_aref, dim_order, out);
}

TEST_F(OpEmptyDimOrderOutTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape unbound not supported";
  }
  TensorFactory<ScalarType::Float> tf;

  int64_t sizes[2] = {3, 2};
  auto sizes_aref = executorch::aten::ArrayRef<int64_t>(sizes);
  OptionalArrayRef<int64_t> dim_order;
  Tensor out =
      tf.ones({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  op_empty_dim_order_out(sizes_aref, dim_order, out);
}
