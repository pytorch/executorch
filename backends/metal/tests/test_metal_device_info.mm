/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 */
#import <Metal/Metal.h>
#include <gtest/gtest.h>
#include <executorch/backends/metal/core/MetalDeviceInfo.h>

using executorch::backends::metal_v2::MetalDeviceInfo;
using executorch::backends::metal_v2::DeviceTier;

namespace {
class MetalDeviceInfoTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!MTLCreateSystemDefaultDevice()) GTEST_SKIP() << "no Metal device";
  }
};

TEST_F(MetalDeviceInfoTest, IsAvailableTrueOnMacWithGPU) {
  EXPECT_TRUE(MetalDeviceInfo::isAvailable());
}

TEST_F(MetalDeviceInfoTest, DeviceReturnsNonNil) {
  EXPECT_NE(MetalDeviceInfo::device(), nil);
}

TEST_F(MetalDeviceInfoTest, DeviceIsCachedAcrossCalls) {
  id<MTLDevice> a = MetalDeviceInfo::device();
  id<MTLDevice> b = MetalDeviceInfo::device();
  EXPECT_EQ(a, b);
}

TEST_F(MetalDeviceInfoTest, TierIsValidEnum) {
  DeviceTier t = MetalDeviceInfo::tier();
  EXPECT_TRUE(t == DeviceTier::Phone ||
              t == DeviceTier::MacBase ||
              t == DeviceTier::MacUltra);
}

TEST_F(MetalDeviceInfoTest, SupportsFamilyApple7Cached) {
  // Apple7 is a baseline (M1/A14+); supportsFamily() should return
  // a stable value and be cached. We just verify the call is safe.
  bool a = MetalDeviceInfo::supportsFamily(MTLGPUFamilyApple7);
  bool b = MetalDeviceInfo::supportsFamily(MTLGPUFamilyApple7);
  EXPECT_EQ(a, b);
}

TEST_F(MetalDeviceInfoTest, IsNaxAvailableConsistent) {
  // Just verify the call is safe and stable across calls.
  bool a = MetalDeviceInfo::isNaxAvailable();
  bool b = MetalDeviceInfo::isNaxAvailable();
  EXPECT_EQ(a, b);
}

TEST_F(MetalDeviceInfoTest, ResetForTestingDropsCachedDevice) {
  // Pre-seed the cache.
  id<MTLDevice> first = MetalDeviceInfo::device();
  ASSERT_NE(first, nil);
  MetalDeviceInfo::resetForTesting();
  // After reset(), the next device() call re-probes via
  // MTLCreateSystemDefaultDevice. The OS caches that itself so the
  // returned identity is the same MTLDevice — verify it's non-nil and
  // points at the same singleton.
  id<MTLDevice> second = MetalDeviceInfo::device();
  EXPECT_NE(second, nil) << "device() should re-probe after reset()";
  EXPECT_EQ(second, first) << "OS-level MTLCreateSystemDefaultDevice should "
                              "return the same identity";
}

}  // namespace
