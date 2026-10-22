/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <vector>

using executorch::runtime::Error;
using executorch::runtime::etensor::DeviceType;
using executorch::runtime::etensor::ScalarType;
using executorch::runtime::etensor::Tensor;
using executorch::runtime::etensor::TensorImpl;
using executorch::runtime::internal::copy_tensor_data;

// Copying into a memory-planned tensor uses a host memcpy. A tensor that lives
// on an accelerator cannot be filled that way, and doing it anyway crashed the
// process with no message, so the runtime reports it instead.
class CopyTensorDataDeviceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    executorch::runtime::runtime_init();
  }

  // Each tensor gets its own storage, so a host-to-host copy has a distinct
  // destination to land in and the data can be checked afterwards. Sharing one
  // buffer would make the copy a self-memcpy, which is undefined and which no
  // assertion could distinguish from the copy being skipped.
  Tensor make(
      DeviceType device,
      std::array<float, 4> values = {0.0f, 0.0f, 0.0f, 0.0f}) {
    storages_.push_back(std::make_unique<std::array<float, 4>>(values));
    impls_.push_back(std::make_unique<TensorImpl>(
        ScalarType::Float,
        static_cast<ssize_t>(sizes_.size()),
        sizes_.data(),
        storages_.back()->data(),
        dim_order_.data(),
        strides_.data(),
        executorch::runtime::TensorShapeDynamism::STATIC,
        device));
    return Tensor(impls_.back().get());
  }

  std::array<executorch::aten::SizesType, 1> sizes_{4};
  std::array<executorch::aten::DimOrderType, 1> dim_order_{0};
  std::array<executorch::aten::StridesType, 1> strides_{1};
  std::vector<std::unique_ptr<std::array<float, 4>>> storages_;
  std::vector<std::unique_ptr<TensorImpl>> impls_;
};

TEST_F(CopyTensorDataDeviceTest, HostToHostIsCopied) {
  Tensor destination = make(DeviceType::CPU);
  Tensor source = make(DeviceType::CPU, {1.0f, 2.0f, 3.0f, 4.0f});
  EXPECT_EQ(copy_tensor_data(destination, source), Error::Ok);
  // Assert the data actually moved. Checking only the returned Error would
  // still pass if the copy were removed entirely.
  const float* copied = destination.const_data_ptr<float>();
  ASSERT_NE(copied, nullptr);
  EXPECT_EQ(copied[0], 1.0f);
  EXPECT_EQ(copied[1], 2.0f);
  EXPECT_EQ(copied[2], 3.0f);
  EXPECT_EQ(copied[3], 4.0f);
}

TEST_F(CopyTensorDataDeviceTest, ADeviceDestinationIsRefused) {
  // This is the case that used to crash: a planned buffer on an accelerator,
  // filled by a host memcpy.
  Tensor destination = make(DeviceType::CUDA);
  Tensor source = make(DeviceType::CPU);
  EXPECT_EQ(copy_tensor_data(destination, source), Error::NotSupported);
}

TEST_F(CopyTensorDataDeviceTest, ADeviceSourceIsRefused) {
  Tensor destination = make(DeviceType::CPU);
  Tensor source = make(DeviceType::CUDA);
  EXPECT_EQ(copy_tensor_data(destination, source), Error::NotSupported);
}
