/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/portable_type/device.h>

#include <gtest/gtest.h>

using executorch::runtime::etensor::Device;
using executorch::runtime::etensor::DeviceIndex;
using executorch::runtime::etensor::DeviceType;
using executorch::runtime::etensor::kNumDeviceTypes;

// --- DeviceType enum ---

TEST(DeviceTypeTest, EnumValues) {
  EXPECT_EQ(static_cast<int8_t>(DeviceType::CPU), 0);
  EXPECT_EQ(static_cast<int8_t>(DeviceType::CUDA), 1);
}

TEST(DeviceTypeTest, NumDeviceTypesCoversAllEnums) {
  // kNumDeviceTypes must be large enough to index all defined device types.
  EXPECT_GT(kNumDeviceTypes, static_cast<size_t>(DeviceType::CPU));
  EXPECT_GT(kNumDeviceTypes, static_cast<size_t>(DeviceType::CUDA));
}

// --- Device: CPU ---

TEST(DeviceTest, CpuDefaultIndex) {
  Device d(DeviceType::CPU);
  EXPECT_TRUE(d.is_cpu());
  EXPECT_EQ(d.type(), DeviceType::CPU);
  EXPECT_EQ(d.index(), -1);
}

TEST(DeviceTest, CpuExplicitIndex) {
  Device d(DeviceType::CPU, 0);
  EXPECT_TRUE(d.is_cpu());
  EXPECT_EQ(d.index(), 0);
}

// --- Device: CUDA ---

TEST(DeviceTest, CudaDefaultIndex) {
  Device d(DeviceType::CUDA);
  EXPECT_FALSE(d.is_cpu());
  EXPECT_EQ(d.type(), DeviceType::CUDA);
  EXPECT_EQ(d.index(), -1);
}

TEST(DeviceTest, CudaExplicitIndex) {
  Device d(DeviceType::CUDA, 0);
  EXPECT_EQ(d.index(), 0);
}

// --- Device: equality ---

TEST(DeviceTest, EqualitySameTypeAndIndex) {
  EXPECT_EQ(Device(DeviceType::CPU, 0), Device(DeviceType::CPU, 0));
  EXPECT_EQ(Device(DeviceType::CUDA, 1), Device(DeviceType::CUDA, 1));
}

TEST(DeviceTest, InequalityDifferentType) {
  EXPECT_NE(Device(DeviceType::CPU, 0), Device(DeviceType::CUDA, 0));
}

TEST(DeviceTest, InequalityDifferentIndex) {
  EXPECT_NE(Device(DeviceType::CUDA, 0), Device(DeviceType::CUDA, 1));
}

TEST(DeviceTest, EqualityDefaultIndices) {
  EXPECT_EQ(Device(DeviceType::CPU), Device(DeviceType::CPU));
  EXPECT_EQ(Device(DeviceType::CUDA), Device(DeviceType::CUDA));
  EXPECT_NE(Device(DeviceType::CPU), Device(DeviceType::CUDA));
}

// --- Device: implicit construction ---

TEST(DeviceTest, ImplicitConstructionFromDeviceType) {
  // Device constructor is implicit, allowing DeviceType → Device conversion.
  Device d = DeviceType::CUDA;
  EXPECT_EQ(d.index(), -1);
}

// --- Deprecated namespace aliases ---

TEST(DeviceTest, DeprecatedNamespaceAliases) {
  // Verify the torch::executor aliases still work.
  torch::executor::Device d(torch::executor::DeviceType::CUDA, 0);
  EXPECT_EQ(d.index(), 0);
}
