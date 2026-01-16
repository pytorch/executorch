/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/aoti/slim/c10/core/Device.h>
#include <executorch/backends/aoti/slim/c10/core/DeviceType.h>

#include <gtest/gtest.h>

using namespace executorch::backends::aoti::slim::c10;

class DeviceTypeTest : public ::testing::Test {};

TEST_F(DeviceTypeTest, CPUEnumValue) {
  // Verify CPU has the correct enum value (0)
  EXPECT_EQ(static_cast<int>(DeviceType::CPU), 0);
}

TEST_F(DeviceTypeTest, DeviceTypeName) {
  // Verify DeviceTypeName returns correct strings
  EXPECT_EQ(DeviceTypeName(DeviceType::CPU, false), "CPU");
  EXPECT_EQ(DeviceTypeName(DeviceType::CPU, true), "cpu");
}

TEST_F(DeviceTypeTest, IsValidDeviceType) {
  // Verify isValidDeviceType works correctly
  EXPECT_TRUE(isValidDeviceType(DeviceType::CPU));
}

TEST_F(DeviceTypeTest, KCPUConstant) {
  // Verify kCPU constant
  EXPECT_EQ(kCPU, DeviceType::CPU);
}

class DeviceTest : public ::testing::Test {};

TEST_F(DeviceTest, ConstructFromDeviceType) {
  // Construct Device from DeviceType
  Device cpu_device(DeviceType::CPU);

  EXPECT_TRUE(cpu_device.is_cpu());
  EXPECT_EQ(cpu_device.type(), DeviceType::CPU);
  EXPECT_EQ(cpu_device.index(), -1); // Default index
  EXPECT_FALSE(cpu_device.has_index());
}

TEST_F(DeviceTest, ConstructWithIndex) {
  // Construct Device with explicit index
  Device cpu_device(DeviceType::CPU, 0);

  EXPECT_TRUE(cpu_device.is_cpu());
  EXPECT_EQ(cpu_device.type(), DeviceType::CPU);
  EXPECT_EQ(cpu_device.index(), 0);
  EXPECT_TRUE(cpu_device.has_index());
}

TEST_F(DeviceTest, ConstructFromString) {
  // Construct Device from string
  Device cpu1("cpu");
  EXPECT_TRUE(cpu1.is_cpu());
  EXPECT_EQ(cpu1.index(), -1);

  Device cpu2("CPU");
  EXPECT_TRUE(cpu2.is_cpu());
  EXPECT_EQ(cpu2.index(), -1);

  Device cpu3("cpu:0");
  EXPECT_TRUE(cpu3.is_cpu());
  EXPECT_EQ(cpu3.index(), 0);
}

TEST_F(DeviceTest, Equality) {
  Device cpu1(DeviceType::CPU, 0);
  Device cpu2(DeviceType::CPU, 0);
  Device cpu3(DeviceType::CPU, -1);

  EXPECT_EQ(cpu1, cpu2);
  EXPECT_NE(cpu1, cpu3);
}

TEST_F(DeviceTest, Str) {
  Device cpu1(DeviceType::CPU);
  EXPECT_EQ(cpu1.str(), "cpu");

  Device cpu2(DeviceType::CPU, 0);
  EXPECT_EQ(cpu2.str(), "cpu:0");
}

TEST_F(DeviceTest, SetIndex) {
  Device cpu(DeviceType::CPU);
  EXPECT_EQ(cpu.index(), -1);

  cpu.set_index(0);
  EXPECT_EQ(cpu.index(), 0);
  EXPECT_TRUE(cpu.has_index());
}

TEST_F(DeviceTest, Hash) {
  // Verify Device can be hashed (for use in unordered containers)
  Device cpu1(DeviceType::CPU, 0);
  Device cpu2(DeviceType::CPU, 0);
  Device cpu3(DeviceType::CPU, -1);

  std::hash<Device> hasher;
  EXPECT_EQ(hasher(cpu1), hasher(cpu2));
  EXPECT_NE(hasher(cpu1), hasher(cpu3));
}
