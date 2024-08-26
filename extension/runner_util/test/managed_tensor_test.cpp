/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/runner_util/managed_tensor.h>

#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::DimOrderType;
using exec_aten::ScalarType;
using exec_aten::SizesType;
using exec_aten::StridesType;
using executorch::extension::ManagedTensor;
using executorch::runtime::ArrayRef;

class ManagedTensorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    executorch::runtime::runtime_init();

    data_ = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    sizes_ = {2, 3, 4};
    expected_strides_ = {12, 4, 1};
    managed_tensor_ =
        std::make_unique<ManagedTensor>(data_.data(), sizes_, ScalarType::Long);
  }

 protected:
  std::vector<int64_t> data_;
  std::vector<SizesType> sizes_;
  std::vector<int> expected_strides_;
  std::unique_ptr<ManagedTensor> managed_tensor_;
};

TEST_F(ManagedTensorTest, Smoke) {
  const auto tensor = managed_tensor_->get_aliasing_tensor();

  EXPECT_EQ(tensor.sizes(), ArrayRef<SizesType>(sizes_.data(), sizes_.size()));
  EXPECT_EQ(tensor.scalar_type(), ScalarType::Long);
  EXPECT_EQ(tensor.const_data_ptr(), data_.data());
  for (size_t i = 0; i < expected_strides_.size(); ++i) {
    EXPECT_EQ(tensor.strides()[i], expected_strides_[i]);
  }
}

TEST_F(ManagedTensorTest, ResizeWithUpdatedRank) {
  // gtest death test doesn't work on iOS:
  // https://github.com/google/googletest/issues/2834
#if !GTEST_OS_IOS
  EXPECT_EXIT(
      managed_tensor_->resize(std::vector<SizesType>{2, 3, 4, 5}),
      ::testing::KilledBySignal(SIGABRT),
      "");
#endif
}

TEST_F(ManagedTensorTest, ResizeShrink) {
  managed_tensor_->resize(std::vector<SizesType>{2, 2, 2});
  const auto tensor = managed_tensor_->get_aliasing_tensor();

  std::vector<SizesType> expected_sizes = {2, 2, 2};
  EXPECT_EQ(
      tensor.sizes(),
      ArrayRef<SizesType>(expected_sizes.data(), expected_sizes.size()));
  EXPECT_EQ(tensor.scalar_type(), ScalarType::Long);
  EXPECT_EQ(tensor.const_data_ptr(), data_.data());
}

TEST_F(ManagedTensorTest, Resize) {
  managed_tensor_->resize(std::vector<SizesType>{4, 3, 2});
  const auto tensor = managed_tensor_->get_aliasing_tensor();

  std::vector<SizesType> expected_sizes = {4, 3, 2};
  EXPECT_EQ(
      tensor.sizes(),
      ArrayRef<SizesType>(expected_sizes.data(), expected_sizes.size()));
  EXPECT_EQ(tensor.scalar_type(), ScalarType::Long);
  EXPECT_EQ(tensor.const_data_ptr(), data_.data());
}
