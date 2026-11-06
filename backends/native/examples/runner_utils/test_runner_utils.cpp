// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/examples/runner_utils/runner_utils.h>

#include <bit>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#if !defined(_WIN32)
#include <sys/resource.h>
#endif

namespace ptn::runner_utils {
namespace {

class RunnerOutputTest : public ::testing::Test {
 protected:
  void SetUp() override {
    directory_ = std::filesystem::path(::testing::TempDir()) /
        ("runner_output_" + std::to_string(std::random_device{}()));
    owns_directory_ = std::filesystem::create_directory(directory_);
    ASSERT_TRUE(owns_directory_);
    path_ = (directory_ / "output.bin").string();
  }

  void TearDown() override {
    if (owns_directory_) {
      std::filesystem::remove_all(directory_);
    }
  }

  std::filesystem::path directory_;
  std::string path_;
  bool owns_directory_ = false;
};

// cppcheck-suppress-begin syntaxError
TEST_F(RunnerOutputTest, WriteFloatsPreservesAllBits) {
  const std::vector<uint32_t> bits{
      0x00000000, 0x80000000, 0x3f800000, 0xc0400000, 0x7f800000, 0x7fc12345};
  std::vector<float> values;
  values.reserve(bits.size());
  for (const uint32_t value : bits) {
    values.push_back(std::bit_cast<float>(value));
  }

  write_floats(path_, values);

  EXPECT_EQ(std::filesystem::file_size(path_), bits.size() * sizeof(float));
  std::vector<uint32_t> actual;
  for (const float value : read_floats(path_)) {
    actual.push_back(std::bit_cast<uint32_t>(value));
  }
  EXPECT_EQ(actual, bits);
}

TEST_F(RunnerOutputTest, WriteFloatsDoesNotOverwriteExistingFile) {
  const std::vector<float> original{1.0f, -3.0f};
  write_floats(path_, original);

  EXPECT_THROW(
      write_floats(path_, std::vector<float>{4.0f}), std::runtime_error);

  EXPECT_EQ(read_floats(path_), original);
}

TEST_F(RunnerOutputTest, WriteFloatsCreatesEmptyOutput) {
  write_floats(path_, {});

  EXPECT_EQ(std::filesystem::file_size(path_), 0);
  EXPECT_TRUE(read_floats(path_).empty());
}

TEST_F(RunnerOutputTest, WriteFloatsReportsInvalidOutputPath) {
  const std::string missing = (directory_ / "missing" / "output.bin").string();

  EXPECT_THROW(
      write_floats(missing, std::vector<float>{1.0f}), std::runtime_error);
}

#if !defined(_WIN32)
TEST_F(RunnerOutputTest, WriteFloatsRemovesIncompleteOutput) {
  const std::vector<float> values(1024, 1.0f);
  rlimit original{};
  ASSERT_EQ(getrlimit(RLIMIT_FSIZE, &original), 0);
  rlimit limited = original;
  limited.rlim_cur = sizeof(float);
  const auto previous_handler = std::signal(SIGXFSZ, SIG_IGN);
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &limited), 0);

  bool threw = false;
  try {
    write_floats(path_, values);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  setrlimit(RLIMIT_FSIZE, &original);
  std::signal(SIGXFSZ, previous_handler);

  EXPECT_TRUE(threw);
  EXPECT_FALSE(std::filesystem::exists(path_));
}
#endif

TEST(RunnerUtilsTest, CanCompareFloatsChecksSizesAndTolerances) {
  EXPECT_TRUE(can_compare_floats(2, 2, 1e-3, 0.0));
  EXPECT_FALSE(can_compare_floats(1, 2, 1e-3, 1e-3));
  EXPECT_FALSE(can_compare_floats(1, 1, -1e-3, 1e-3));
}

TEST(RunnerUtilsTest, CompareFloatsUsesAbsoluteAndRelativeTolerance) {
  EXPECT_TRUE(compare_floats({1.0f, 100.0f}, {1.0005f, 100.05f}, 1e-3, 1e-3));
  EXPECT_FALSE(compare_floats({1.0f, 100.0f}, {1.1f, 100.0f}, 1e-3, 1e-3));
  EXPECT_FALSE(compare_floats({1.0f}, {1.0f, 2.0f}, 1e-3, 1e-3));
}

TEST(RunnerUtilsTest, CompareFloatsRejectsNonFiniteValues) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float infinity = std::numeric_limits<float>::infinity();

  EXPECT_FALSE(compare_floats({nan}, {1.0f}, 1e-3, 1e-3));
  EXPECT_FALSE(compare_floats({1.0f}, {nan}, 1e-3, 1e-3));
  EXPECT_FALSE(compare_floats({infinity}, {infinity}, 1e-3, 1e-3));
}

TEST(RunnerUtilsTest, CompareFloatsRejectsInvalidTolerances) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();

  EXPECT_FALSE(compare_floats({1.0f}, {1.0f}, nan, 1e-3));
  EXPECT_FALSE(compare_floats({1.0f}, {1.0f}, 1e-3, infinity));
  EXPECT_FALSE(compare_floats({1.0f}, {1.0f}, -1e-3, 1e-3));
}

TEST(RunnerUtilsTest, TopKReturnsEmptyForEmptyInput) {
  EXPECT_TRUE(top_k({}, 5).empty());
  EXPECT_TRUE(top_k({1.0f, 2.0f}, -1).empty());
}

TEST(RunnerUtilsTest, TopKReturnsIndicesByDescendingValue) {
  EXPECT_EQ(top_k({0.1f, 0.9f, 0.5f, 0.7f}, 3), (std::vector<int>{1, 3, 2}));
  EXPECT_EQ(top_k({2.0f, 3.0f}, 5), (std::vector<int>{1, 0}));
}
// cppcheck-suppress-end syntaxError

} // namespace
} // namespace ptn::runner_utils
