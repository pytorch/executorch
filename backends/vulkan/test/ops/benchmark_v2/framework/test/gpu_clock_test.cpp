// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/GpuClock.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <random>

namespace vkcompute::shader_profiling {
namespace {

class GpuClockTest : public ::testing::Test {
 protected:
  std::filesystem::path root_;

  void SetUp() override {
    do {
      root_ = std::filesystem::temp_directory_path() /
          ("benchmark_gpu_clock_" + std::to_string(std::random_device{}()));
    } while (!std::filesystem::create_directory(root_));
  }

  void TearDown() override {
    std::filesystem::remove_all(root_);
  }

  void write(const std::string& path, const std::string& value) {
    const auto file = root_ / path;
    std::filesystem::create_directories(file.parent_path());
    std::ofstream output(file);
    ASSERT_TRUE(output.good());
    output << value;
  }
};

TEST_F(GpuClockTest, ReadsChangingAdrenoFrequencyInHz) {
  const std::string path = "class/devfreq/3d00000.qcom,kgsl-3d0/cur_freq";
  write(path, "903000000\n");
  GpuClock clock(root_.string());
  EXPECT_EQ(clock.read_mhz(), 903.0);
  write(path, "366000000\n");
  EXPECT_EQ(clock.read_mhz(), 366.0);
}

TEST_F(GpuClockTest, ReadsClassicKgslFrequencyInHz) {
  write("class/kgsl/kgsl-3d0/gpuclk", "578000000\n");
  EXPECT_EQ(GpuClock(root_.string()).read_mhz(), 578.0);
}

TEST_F(GpuClockTest, ReadsPixelMaliFrequencyInKHz) {
  write("devices/platform/1f000000.mali/cur_freq", "150000\n");
  EXPECT_EQ(GpuClock(root_.string()).read_mhz(), 150.0);
}

TEST_F(GpuClockTest, ReadsLegacyMaliClockInMHz) {
  write("devices/platform/18500000.mali/clock", "702\n");
  EXPECT_EQ(GpuClock(root_.string()).read_mhz(), 702.0);
}

TEST_F(GpuClockTest, ReadsExynosSgpuFrequencyInKHz) {
  write("class/devfreq/22200000.sgpu/cur_freq", "1000000\n");
  EXPECT_EQ(GpuClock(root_.string()).read_mhz(), 1000.0);
}

TEST_F(GpuClockTest, KernelFallbackUsesDriverSpecificUnits) {
  write("class/devfreq/22200000.sgpu/name", "sgpu\n");
  write("kernel/gpu/gpu_clock", "800000\n");
  EXPECT_EQ(GpuClock(root_.string()).read_mhz(), 800.0);
  std::filesystem::remove_all(root_ / "class/devfreq/22200000.sgpu");
  write("class/devfreq/3d00000.qcom,kgsl-3d0/name", "kgsl\n");
  write("kernel/gpu/gpu_clock", "903\n");
  EXPECT_EQ(GpuClock(root_.string()).read_mhz(), 903.0);
}

TEST_F(GpuClockTest, IgnoresBusAndNonGpuDevfreqDevices) {
  write("class/devfreq/kgsl-busmon/cur_freq", "999000000\n");
  write("class/devfreq/gpu-bw/cur_freq", "999000000\n");
  write("class/devfreq/17000010.devfreq_mif/cur_freq", "999000000\n");
  write("class/devfreq/1c000000.gpu/cur_freq", "450000000\n");
  EXPECT_EQ(GpuClock(root_.string()).read_mhz(), 450.0);
}

TEST_F(GpuClockTest, MissingAndInvalidSourcesAreUnavailable) {
  EXPECT_EQ(GpuClock(root_.string()).read_mhz(), std::nullopt);
  write("class/kgsl/kgsl-3d0/gpuclk", "invalid");
  EXPECT_EQ(GpuClock(root_.string()).read_mhz(), std::nullopt);
  write("class/devfreq/3d00000.qcom,kgsl-3d0/cur_freq", "903000000\n");
  EXPECT_EQ(GpuClock(root_.string()).read_mhz(), 903.0);
}

TEST_F(GpuClockTest, FailedSamplesAreUnavailableAndZeroIsValid) {
  const std::string path = "class/kgsl/kgsl-3d0/gpuclk";
  write(path, "903000000\n");
  GpuClock clock(root_.string());
  for (const char* value : {"", "bad", "-1", "nan", "inf", "100 200"}) {
    write(path, value);
    EXPECT_EQ(clock.read_mhz(), std::nullopt);
  }
  write(path, "0\n");
  EXPECT_EQ(clock.read_mhz(), 0.0);
  std::filesystem::remove(root_ / path);
  EXPECT_EQ(clock.read_mhz(), std::nullopt);
}

} // namespace
} // namespace vkcompute::shader_profiling
