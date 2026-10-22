// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#import <Foundation/Foundation.h>
#import <TargetConditionals.h>

#include "SwiftPMMetallibPath.h"

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

const char* wrong_metallib_filename() {
#if TARGET_OS_OSX
  return "mlx-ios.metallib";
#else
  return "mlx-macos.metallib";
#endif
}

class MLXMetallibPathTest : public ::testing::Test {
 protected:
  void SetUp() override {
    NSString* name = [NSString
        stringWithFormat:@"executorch_mlx_metallib_path_%@",
                         NSUUID.UUID.UUIDString];
    root_ = std::filesystem::temp_directory_path() /
        std::string(name.fileSystemRepresentation);
    ASSERT_TRUE(std::filesystem::create_directories(root_));
  }

  void TearDown() override {
    std::error_code error;
    std::filesystem::remove_all(root_, error);
  }

  std::filesystem::path create_bundle(bool deep) {
    const auto bundle = root_ / "executorch_backend_mlx_resources.bundle";
    const auto contents = deep ? bundle / "Contents" : bundle;
    const auto resources = deep ? contents / "Resources" : bundle;
    EXPECT_TRUE(std::filesystem::create_directories(resources));

    std::ofstream(contents / "Info.plist")
        << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        << "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" "
           "\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
        << "<plist version=\"1.0\"><dict>"
        << "<key>CFBundleIdentifier</key>"
        << "<string>org.pytorch.executorch.mlx-test-resources</string>"
        << "<key>CFBundlePackageType</key><string>BNDL</string>"
        << "</dict></plist>\n";
    return resources;
  }

  std::filesystem::path root_;
};

TEST_F(MLXMetallibPathTest, MissingBundleReturnsNoPath) {
  EXPECT_FALSE(find_swiftpm_metallib_path({root_.string()}).has_value());
}

TEST_F(MLXMetallibPathTest, ProcessWithoutSwiftPMBundleReturnsNoPath) {
  EXPECT_FALSE(resolve_swiftpm_metallib_path().has_value());
}

TEST_F(MLXMetallibPathTest, FindsFlatBundleResource) {
  const auto resources = create_bundle(/*deep=*/false);
  const auto metallib = resources / expected_metallib_filename();
  std::ofstream(metallib) << "fixture";

  EXPECT_EQ(find_swiftpm_metallib_path({root_.string()}), metallib.string());
}

TEST_F(MLXMetallibPathTest, FindsMacOSDeepBundleResource) {
  const auto resources = create_bundle(/*deep=*/true);
  const auto metallib = resources / expected_metallib_filename();
  std::ofstream(metallib) << "fixture";

  EXPECT_EQ(find_swiftpm_metallib_path({root_.string()}), metallib.string());
  EXPECT_EQ(
      find_swiftpm_metallib_path(
          {(root_ / "executorch_backend_mlx_resources.bundle").string()}),
      metallib.string());
}

TEST_F(MLXMetallibPathTest, SelectsCurrentPlatformSlice) {
  const auto resources = create_bundle(/*deep=*/true);
  const auto expected = resources / expected_metallib_filename();
  const auto wrong = resources / wrong_metallib_filename();
  std::ofstream(expected) << "expected";
  std::ofstream(wrong) << "wrong";

  EXPECT_EQ(find_swiftpm_metallib_path({root_.string()}), expected.string());
}

TEST_F(MLXMetallibPathTest, IgnoresWrongPlatformSlice) {
  const auto resources = create_bundle(/*deep=*/true);
  std::ofstream(resources / wrong_metallib_filename()) << "fixture";

  EXPECT_FALSE(find_swiftpm_metallib_path({root_.string()}).has_value());
}

} // namespace
} // namespace executorch::backends::mlx
