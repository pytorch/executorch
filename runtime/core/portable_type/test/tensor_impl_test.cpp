/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/portable_type/tensor_impl.h>

#include <gtest/gtest.h>
#include <random>

#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/DeathTest.h>

using namespace ::testing;

using executorch::runtime::ArrayRef;
using executorch::runtime::Error;
using executorch::runtime::TensorShapeDynamism;
using executorch::runtime::etensor::ScalarType;
using executorch::runtime::etensor::TensorImpl;
using SizesType = TensorImpl::SizesType;
using DimOrderType = TensorImpl::DimOrderType;
using StridesType = TensorImpl::StridesType;
using executorch::runtime::internal::resize_tensor_impl;

class TensorImplTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    executorch::runtime::runtime_init();
  }
};

TEST_F(TensorImplTest, TestCtorAndGetters) {
  SizesType sizes[2] = {3, 2};
  DimOrderType dim_order[2] = {0, 1};
  StridesType strides[2] = {2, 1};
  float data[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  TensorImpl t(ScalarType::Float, 2, sizes, data, dim_order, strides);

  EXPECT_EQ(t.numel(), 6);
  EXPECT_EQ(t.nbytes(), 6 * 4); // 6 4 byte floats
  EXPECT_EQ(t.dim(), 2);
  EXPECT_EQ(t.scalar_type(), ScalarType::Float);
  EXPECT_EQ(t.element_size(), 4);
  EXPECT_EQ(t.data(), data);
  EXPECT_EQ(t.mutable_data(), data);
  EXPECT_EQ(t.sizes().data(), sizes);
  EXPECT_EQ(t.sizes().size(), 2);
  EXPECT_EQ(t.strides().data(), strides);
  EXPECT_EQ(t.strides().size(), 2);
}

// Verify that contig means stride[0] >= stride[1] >= ... stride[size-1] == 1
TEST_F(TensorImplTest, TestSetSizesContigContract) {
  const int RANK = 5;
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(1, 100);
  SizesType sizes[RANK] = {100, 100, 100, 100, 100};
  DimOrderType dim_order[RANK] = {0, 1, 2, 3, 4};
  StridesType strides[RANK] = {100000000, 1000000, 10000, 100, 1};
  float* data = nullptr;
  TensorImpl t(
      ScalarType::Float,
      RANK,
      sizes,
      data,
      dim_order,
      strides,
      TensorShapeDynamism::DYNAMIC_BOUND);

  SizesType new_sizes[RANK] = {0, 0, 0, 0, 0};
  // assign random sizes between 1 and 100
  for (int i = 0; i < RANK; i++) {
    new_sizes[i] = distribution(generator);
  }
  Error err = resize_tensor_impl(&t, {new_sizes, RANK});
  EXPECT_EQ(err, Error::Ok);

  auto strides_ref = t.strides();
  StridesType prev = strides_ref[0];
  for (auto stride : strides_ref) {
    EXPECT_LE(stride, prev);
  }
  EXPECT_EQ(t.strides()[strides_ref.size() - 1], 1);
}

TEST_F(TensorImplTest, TestSetSizesContigZeroSizes) {
  SizesType sizes[3] = {2, 0, 3};
  DimOrderType dim_order[3] = {0, 1, 2};
  StridesType strides[3] = {3, 3, 1};
  float* data = nullptr;
  TensorImpl t(
      ScalarType::Float,
      3,
      sizes,
      data,
      dim_order,
      strides,
      TensorShapeDynamism::DYNAMIC_BOUND);

  SizesType new_sizes_1[3] = {1, 0, 2};
  Error err = resize_tensor_impl(&t, {new_sizes_1, 3});
  EXPECT_EQ(err, Error::Ok);
  EXPECT_EQ(t.size(1), 0);

  // Treat 0 dimensions as size 1 for stride calculation as thats what aten does
  auto strides_ref = t.strides();
  EXPECT_EQ(strides_ref[0], 2);
  EXPECT_EQ(strides_ref[1], 2);
  EXPECT_EQ(strides_ref[2], 1);

  // Numel is 0 for tensors with a 0 sized dimension
  EXPECT_EQ(t.numel(), 0);
}

TEST_F(TensorImplTest, TestSetSizesContigStatic) {
  SizesType sizes[2] = {3, 2};
  DimOrderType dim_order[2] = {0, 1};
  StridesType strides[2] = {2, 1};
  float data[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  TensorImpl t(ScalarType::Float, 2, sizes, data, dim_order, strides);

  SizesType new_sizes_1[2] = {3, 2};
  Error err = resize_tensor_impl(&t, {new_sizes_1, 2});
  EXPECT_EQ(err, Error::Ok);
  EXPECT_EQ(t.size(1), 2);

  // strides shouldnt change
  auto strides_ref = t.strides();
  EXPECT_EQ(strides_ref[0], 2);
  EXPECT_EQ(strides_ref[1], 1);

  SizesType new_sizes_2[2] = {2, 2};
  // Can't change size of a StaticShape Tensor
  err = resize_tensor_impl(&t, {new_sizes_2, 2});
  EXPECT_NE(err, Error::Ok);

  SizesType new_sizes_3[1] = {2};
  // Can't change rank of any Tensor
  err = resize_tensor_impl(&t, {new_sizes_3, 1});
  EXPECT_NE(err, Error::Ok);
}

TEST_F(TensorImplTest, TestSetSizesContigUpperBounded) {
  SizesType sizes[2] = {3, 2};
  DimOrderType dim_order[2] = {0, 1};
  StridesType strides[2] = {2, 1};
  float data[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  TensorImpl t(
      ScalarType::Float,
      2,
      sizes,
      data,
      dim_order,
      strides,
      TensorShapeDynamism::DYNAMIC_BOUND);

  SizesType new_sizes_1[2] = {1, 1};
  // Can resize down
  Error err = resize_tensor_impl(&t, {new_sizes_1, 2});
  EXPECT_EQ(err, Error::Ok);
  EXPECT_EQ(t.size(1), 1);

  // strides contiguous
  auto strides_ref = t.strides();
  EXPECT_EQ(strides_ref[0], 1);
  EXPECT_EQ(strides_ref[1], 1);

  SizesType new_sizes_2[2] = {3, 2};
  // Can resize back up
  err = resize_tensor_impl(&t, {new_sizes_2, 2});
  EXPECT_EQ(err, Error::Ok);
  EXPECT_EQ(t.size(1), 2);

  // Back to original strides
  strides_ref = t.strides();
  EXPECT_EQ(strides_ref[0], 2);
  EXPECT_EQ(strides_ref[1], 1);

  SizesType new_sizes_3[2] = {4, 2};
  // Can't execeed capacity of UpperBounded Tensor
  err = resize_tensor_impl(&t, {new_sizes_3, 2});
  EXPECT_NE(err, Error::Ok);

  SizesType new_sizes_4[1] = {4};
  // Can't change rank of any Tensor
  err = resize_tensor_impl(&t, {new_sizes_4, 1});
  EXPECT_NE(err, Error::Ok);
}

TEST_F(TensorImplTest, TestZeroDimSetEmptySizesContig) {
  SizesType sizes[0] = {};
  DimOrderType dim_order[0] = {};
  StridesType strides[0] = {};
  float data[1] = {1.0};
  TensorImpl t(
      ScalarType::Float,
      0,
      sizes,
      data,
      dim_order,
      strides,
      TensorShapeDynamism::DYNAMIC_BOUND);

  ArrayRef<SizesType> new_sizes_empty{};
  // Can resize with empty sizes
  Error err = resize_tensor_impl(&t, new_sizes_empty);
  EXPECT_EQ(err, Error::Ok);
  EXPECT_EQ(t.dim(), 0);

  SizesType new_sizes_1[1] = {1};
  // Can't change rank of tensor
  err = resize_tensor_impl(&t, {new_sizes_1, 1});
  EXPECT_NE(err, Error::Ok);
}

TEST_F(TensorImplTest, TestSetSizesContigUnbounded) {
  SizesType sizes[2] = {3, 2};
  DimOrderType dim_order[2] = {0, 1};
  StridesType strides[2] = {2, 1};
  float data[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  TensorImpl t(
      ScalarType::Float,
      2,
      sizes,
      data,
      dim_order,
      strides,
      TensorShapeDynamism::DYNAMIC_UNBOUND);

  SizesType new_sizes_1[2] = {1, 1};
  // Can resize down
  Error err = resize_tensor_impl(&t, {new_sizes_1, 2});
  EXPECT_EQ(err, Error::Ok);
  EXPECT_EQ(t.size(1), 1);

  // strides contiguous
  auto strides_ref = t.strides();
  EXPECT_EQ(strides_ref[0], 1);
  EXPECT_EQ(strides_ref[1], 1);

  SizesType new_sizes_2[2] = {3, 2};
  // Can resize back up
  err = resize_tensor_impl(&t, {new_sizes_2, 2});
  EXPECT_EQ(err, Error::Ok);
  EXPECT_EQ(t.size(1), 2);

  // Back to original strides
  strides_ref = t.strides();
  EXPECT_EQ(strides_ref[0], 2);
  EXPECT_EQ(strides_ref[1], 1);

  SizesType new_sizes_4[1] = {4};
  // Can't change rank of any Tensor
  err = resize_tensor_impl(&t, {new_sizes_4, 1});
  EXPECT_NE(err, Error::Ok);

  // TODO(T175194371): For now, we can't resize past the original capacity.
  // Once we can, this test should check for that ability.

  SizesType new_sizes_3[2] = {4, 2};
  // Can't execeed original capacity.
  err = resize_tensor_impl(&t, {new_sizes_3, 2});
  EXPECT_NE(err, Error::Ok);
}

TEST_F(TensorImplTest, TestDynamicTensorNoStridesDimOrder) {
  SizesType sizes[3] = {2, 3, 4};
  float data[24] = {0};
  TensorImpl t(
      ScalarType::Float,
      3,
      sizes,
      data,
      nullptr,
      nullptr,
      TensorShapeDynamism::DYNAMIC_BOUND);

  EXPECT_EQ(t.dim(), 3);
  EXPECT_EQ(t.numel(), 24);
  EXPECT_EQ(t.nbytes(), 24 * sizeof(float));

  SizesType new_sizes[3] = {3, 2, 4};
  Error err = resize_tensor_impl(&t, {new_sizes, 3});
  EXPECT_EQ(err, Error::Ok);
  EXPECT_EQ(t.dim(), 3);
  EXPECT_EQ(t.size(0), 3);
  EXPECT_EQ(t.size(1), 2);
  EXPECT_EQ(t.size(2), 4);
  EXPECT_EQ(t.numel(), 3 * 2 * 4);

  const float* y = t.data<float>();
  EXPECT_EQ(y, data);
}

TEST_F(TensorImplTest, TestDynamicTensorNoStridesDimOrderResizeDown) {
  SizesType sizes[3] = {4, 4, 4};
  float data[64] = {0};
  TensorImpl t(
      ScalarType::Float,
      3,
      sizes,
      data,
      nullptr,
      nullptr,
      TensorShapeDynamism::DYNAMIC_BOUND);

  EXPECT_EQ(t.dim(), 3);
  EXPECT_EQ(t.numel(), 64);
  EXPECT_EQ(t.nbytes(), 64 * sizeof(float));

  SizesType new_sizes[3] = {2, 2, 2};
  Error err = resize_tensor_impl(&t, {new_sizes, 3});
  EXPECT_EQ(err, Error::Ok);
  EXPECT_EQ(t.dim(), 3);
  EXPECT_EQ(t.size(0), 2);
  EXPECT_EQ(t.size(1), 2);
  EXPECT_EQ(t.size(2), 2);
  EXPECT_EQ(t.numel(), 2 * 2 * 2);

  const float* y = t.data<float>();
  EXPECT_EQ(y, data);
}

TEST_F(TensorImplTest, TestDynamicTensorNoStridesDimOrderResizeZeroDim) {
  SizesType sizes[3] = {4, 4, 4};
  float data[64] = {0};
  TensorImpl t(
      ScalarType::Float,
      3,
      sizes,
      data,
      nullptr,
      nullptr,
      TensorShapeDynamism::DYNAMIC_BOUND);

  EXPECT_EQ(t.dim(), 3);
  EXPECT_EQ(t.numel(), 64);
  EXPECT_EQ(t.nbytes(), 64 * sizeof(float));

  SizesType new_sizes[3] = {0, 4, 4};
  Error err = resize_tensor_impl(&t, {new_sizes, 3});
  EXPECT_EQ(err, Error::Ok);
  EXPECT_EQ(t.dim(), 3);
  EXPECT_EQ(t.size(0), 0);
  EXPECT_EQ(t.size(1), 4);
  EXPECT_EQ(t.size(2), 4);
  EXPECT_EQ(t.numel(), 0);

  const float* y = t.data<float>();
  EXPECT_EQ(y, data);
}

TEST_F(TensorImplTest, TestWriteRead) {
  SizesType sizes[1] = {1};
  DimOrderType dim_order[1] = {0};
  StridesType strides[1] = {1};
  float data[1] = {1.0};
  TensorImpl t(ScalarType::Float, 1, sizes, data, dim_order, strides);

  const float* y = t.data<float>();
  EXPECT_EQ(y[0], 1.0);

  float* x = t.mutable_data<float>();
  x[0] = 22.0;

  EXPECT_EQ(y[0], 22.0);
}

TEST_F(TensorImplTest, TestInvalidScalarType) {
  SizesType sizes[2] = {3, 2};
  ET_EXPECT_DEATH(TensorImpl t(static_cast<ScalarType>(-1), 2, sizes), "");
}

TEST_F(TensorImplTest, TestNegativeDimension) {
  SizesType sizes[2] = {3, 2};
  ET_EXPECT_DEATH(TensorImpl t(ScalarType::Float, -1, sizes), "");
}

TEST_F(TensorImplTest, TestNullSizesNonZeroDim) {
  ET_EXPECT_DEATH(TensorImpl t(ScalarType::Float, 2, nullptr), "");
}

TEST_F(TensorImplTest, TestNonNegativeSizes) {
  SizesType sizes[2] = {3, -2};
  ET_EXPECT_DEATH(TensorImpl t(ScalarType::Float, 2, sizes), "");
}

TEST_F(TensorImplTest, TestEmptyTensor) {
  SizesType sizes[2] = {0, 0};
  TensorImpl t(ScalarType::Float, 2, sizes);
  EXPECT_EQ(t.numel(), 0);
  EXPECT_EQ(t.data(), nullptr);
}

TEST_F(TensorImplTest, TestTensorWithNoElementsButAllocatedMemory) {
  SizesType sizes[2] = {0, 0};
  float data[1] = {1.0};
  TensorImpl t(ScalarType::Float, 2, sizes, data);
  EXPECT_EQ(t.numel(), 0);
  EXPECT_EQ(t.data(), data);
}

TEST_F(TensorImplTest, TestTensorWithShapeButNoMemory) {
  SizesType sizes[2] = {3, 2};
  TensorImpl t(ScalarType::Float, 2, sizes);
  EXPECT_GT(t.numel(), 0);
  EXPECT_EQ(t.data(), nullptr);
}

TEST_F(TensorImplTest, TestNormalTensor) {
  SizesType sizes[2] = {3, 2};
  float data[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  TensorImpl t(ScalarType::Float, 2, sizes, data);
  EXPECT_GT(t.numel(), 0);
  EXPECT_EQ(t.data(), data);
}

TEST_F(TensorImplTest, TestResizingTensorToZeroAndBack) {
  SizesType sizes[2] = {3, 2};
  TensorImpl t(
      ScalarType::Float,
      2,
      sizes,
      nullptr,
      nullptr,
      nullptr,
      TensorShapeDynamism::DYNAMIC_BOUND);

  float data[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  t.set_data(data);
  EXPECT_GT(t.numel(), 0);
  EXPECT_EQ(t.data(), data);

  SizesType zero_sizes[2] = {0, 0};
  t.set_sizes_contiguous({zero_sizes, 2});
  EXPECT_EQ(t.numel(), 0);
  EXPECT_EQ(t.data(), data);

  SizesType new_sizes[2] = {3, 2};
  t.set_sizes_contiguous({new_sizes, 2});
  EXPECT_GT(t.numel(), 0);
  EXPECT_EQ(t.data(), data);
}
