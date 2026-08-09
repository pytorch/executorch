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
using executorch::runtime::internal::copy_tensor_data;
using executorch::runtime::etensor::DeviceType;
using executorch::runtime::etensor::ScalarType;
using executorch::runtime::etensor::Tensor;
using executorch::runtime::etensor::TensorImpl;

// Copying into a memory-planned tensor uses a host memcpy. A tensor that lives on an accelerator
// cannot be filled that way, and doing it anyway crashed the process with no message, so the runtime
// reports it instead.
class CopyTensorDataDeviceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    executorch::runtime::runtime_init();
  }

  // The data pointer is never dereferenced in the refusal cases below, because the copy is rejected
  // before it happens. It has to be non-null so the earlier argument checks pass.
  Tensor make(DeviceType device) {
    impls_.push_back(std::make_unique<TensorImpl>(
        ScalarType::Float,
        static_cast<ssize_t>(sizes_.size()),
        sizes_.data(),
        storage_.data(),
        dim_order_.data(),
        strides_.data(),
        executorch::runtime::etensor::TensorShapeDynamism::STATIC,
        device));
    return Tensor(impls_.back().get());
  }

  std::array<executorch::aten::SizesType, 1> sizes_{4};
  std::array<executorch::aten::DimOrderType, 1> dim_order_{0};
  std::array<executorch::aten::StridesType, 1> strides_{1};
  std::array<float, 4> storage_{1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<std::unique_ptr<TensorImpl>> impls_;
};

TEST_F(CopyTensorDataDeviceTest, HostToHostIsCopied) {
  Tensor destination = make(DeviceType::CPU);
  Tensor source = make(DeviceType::CPU);
  EXPECT_EQ(copy_tensor_data(destination, source), Error::Ok);
}

TEST_F(CopyTensorDataDeviceTest, ADeviceDestinationIsRefused) {
  // This is the case that used to crash: a planned buffer on an accelerator, filled by a host memcpy.
  Tensor destination = make(DeviceType::CUDA);
  Tensor source = make(DeviceType::CPU);
  EXPECT_EQ(copy_tensor_data(destination, source), Error::NotSupported);
}

TEST_F(CopyTensorDataDeviceTest, ADeviceSourceIsRefused) {
  Tensor destination = make(DeviceType::CPU);
  Tensor source = make(DeviceType::CUDA);
  EXPECT_EQ(copy_tensor_data(destination, source), Error::NotSupported);
}
