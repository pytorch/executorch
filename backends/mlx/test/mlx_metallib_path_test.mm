// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "SwiftPMMetallibPath.h"

#include <TargetConditionals.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace executorch::backends::mlx {
namespace {

const char* expected_metallib_filename() {
#if TARGET_OS_SIMULATOR
  return "mlx-ios-simulator.metallib";
#elif TARGET_OS_IOS
  return "mlx-ios.metallib";
#else
  return "mlx-macos.metallib";
#endif
}

class MLXMetallibPathTest : public ::testing::Test {
 protected:
  void SetUp() override {
    root_ = std::filesystem::temp_directory_path() /
        ("executorch_mlx_metallib_path_" +
         std::to_string(reinterpret_cast<uintptr_t>(this)));
    ASSERT_TRUE(std::filesystem::create_directories(root_));
  }

  void TearDown() override {
    std::error_code error;
    std::filesystem::remove_all(root_, error);
  }

  std::filesystem::path root_;
};

TEST_F(MLXMetallibPathTest, MissingBundleReturnsNoPath) {
  EXPECT_FALSE(find_swiftpm_metallib_path({root_.string()}).has_value());
}

TEST_F(MLXMetallibPathTest, ProcessWithoutSwiftPMBundleReturnsNoPath) {
  EXPECT_FALSE(resolve_swiftpm_metallib_path().has_value());
}

TEST_F(MLXMetallibPathTest, FindsCurrentPlatformSlice) {
  const auto bundle = root_ / "executorch_backend_mlx_resources.bundle";
  ASSERT_TRUE(std::filesystem::create_directory(bundle));
  const auto metallib = bundle / expected_metallib_filename();
  std::ofstream(metallib) << "fixture";

  EXPECT_EQ(find_swiftpm_metallib_path({root_.string()}), metallib.string());
  EXPECT_EQ(find_swiftpm_metallib_path({bundle.string()}), metallib.string());
}

TEST_F(MLXMetallibPathTest, IgnoresWrongPlatformSlice) {
  const auto bundle = root_ / "executorch_backend_mlx_resources.bundle";
  ASSERT_TRUE(std::filesystem::create_directory(bundle));
#if TARGET_OS_OSX
  const auto wrong_metallib = bundle / "mlx-ios.metallib";
#else
  const auto wrong_metallib = bundle / "mlx-macos.metallib";
#endif
  std::ofstream(wrong_metallib) << "fixture";

  EXPECT_FALSE(find_swiftpm_metallib_path({root_.string()}).has_value());
}

} // namespace
} // namespace executorch::backends::mlx
