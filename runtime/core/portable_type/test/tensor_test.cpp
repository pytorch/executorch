/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/portable_type/tensor.h>

#include <gtest/gtest.h>

#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/DeathTest.h>

using executorch::runtime::etensor::ScalarType;
using executorch::runtime::etensor::Tensor;
using executorch::runtime::etensor::TensorImpl;

class TensorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    executorch::runtime::runtime_init();
  }
};

TEST_F(TensorTest, InvalidScalarType) {
  TensorImpl::SizesType sizes[1] = {1};

  // Undefined, which is sort of a special case since it's not part of the
  // iteration macros but is still a part of the enum.
  ET_EXPECT_DEATH({ TensorImpl y(ScalarType::Undefined, 1, sizes); }, "");

  // Some out-of-range types, also demonstrating that NumOptions is not really a
  // scalar type.
  ET_EXPECT_DEATH({ TensorImpl y(ScalarType::NumOptions, 1, sizes); }, "");
  ET_EXPECT_DEATH(
      { TensorImpl y(static_cast<ScalarType>(127), 1, sizes); }, "");
  ET_EXPECT_DEATH({ TensorImpl y(static_cast<ScalarType>(-1), 1, sizes); }, "");
}

TEST_F(TensorTest, SetData) {
  TensorImpl::SizesType sizes[1] = {5};
  TensorImpl::DimOrderType dim_order[1] = {0};
  int32_t data[5] = {0, 0, 1, 0, 0};
  auto a_impl = TensorImpl(ScalarType::Int, 1, sizes, data, dim_order, nullptr);
  auto a = Tensor(&a_impl);
  EXPECT_EQ(a.const_data_ptr(), data);
  a.set_data(nullptr);
  EXPECT_EQ(a.const_data_ptr(), nullptr);
}

TEST_F(TensorTest, Strides) {
  TensorImpl::SizesType sizes[2] = {2, 2};
  TensorImpl::DimOrderType dim_order[2] = {0, 1};
  int32_t data[4] = {0, 0, 1, 1};
  TensorImpl::StridesType strides[2] = {2, 1};
  auto a_impl = TensorImpl(ScalarType::Int, 2, sizes, data, dim_order, strides);
  Tensor a(&a_impl);

  EXPECT_EQ(a_impl.scalar_type(), ScalarType::Int);
  EXPECT_EQ(a.scalar_type(), ScalarType::Int);
  EXPECT_EQ(a.const_data_ptr<int32_t>()[0], 0);
  EXPECT_EQ(a.const_data_ptr<int32_t>()[0 + a.strides()[0]], 1);
}

TEST_F(TensorTest, ModifyDataOfConstTensor) {
  TensorImpl::SizesType sizes[1] = {1};
  TensorImpl::DimOrderType dim_order[2] = {0};
  int32_t data[1] = {1};
  auto a_impl = TensorImpl(ScalarType::Int, 1, sizes, data, dim_order);
  const Tensor a(&a_impl);
  a.mutable_data_ptr<int32_t>()[0] = 0;

  EXPECT_EQ(a_impl.scalar_type(), ScalarType::Int);
  EXPECT_EQ(a.scalar_type(), ScalarType::Int);
  EXPECT_EQ(a.const_data_ptr<int32_t>()[0], 0);
}
