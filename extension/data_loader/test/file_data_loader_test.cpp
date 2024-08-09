/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/data_loader/file_data_loader.h>

#include <cstring>

#include <gtest/gtest.h>

#include <executorch/extension/testing_util/temp_file.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/alignment.h>

using namespace ::testing;
using executorch::extension::FileDataLoader;
using executorch::extension::testing::TempFile;
using executorch::runtime::DataLoader;
using executorch::runtime::Error;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Result;

class FileDataLoaderTest : public ::testing::TestWithParam<size_t> {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    executorch::runtime::runtime_init();
  }

  // The alignment in bytes that tests should use. The values are set by the
  // list in the INSTANTIATE_TEST_SUITE_P call below.
  size_t alignment() const {
    return GetParam();
  }
};

TEST_P(FileDataLoaderTest, InBoundsLoadsSucceed) {
  // Write some heterogeneous data to a file.
  uint8_t data[256];
  for (int i = 0; i < sizeof(data); ++i) {
    data[i] = i;
  }
  TempFile tf(data, sizeof(data));

  // Wrap it in a loader.
  Result<FileDataLoader> fdl =
      FileDataLoader::from(tf.path().c_str(), alignment());
  ASSERT_EQ(fdl.error(), Error::Ok);

  // size() should succeed and reflect the total size.
  Result<size_t> size = fdl->size();
  ASSERT_EQ(size.error(), Error::Ok);
  EXPECT_EQ(*size, sizeof(data));

  // Load the first bytes of the data.
  {
    Result<FreeableBuffer> fb = fdl->load(
        /*offset=*/0,
        /*size=*/8,
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    ASSERT_EQ(fb.error(), Error::Ok);
    EXPECT_ALIGNED(fb->data(), alignment());
    EXPECT_EQ(fb->size(), 8);
    EXPECT_EQ(
        0,
        std::memcmp(
            fb->data(),
            "\x00\x01\x02\x03"
            "\x04\x05\x06\x07",
            fb->size()));

    // Freeing should release the buffer and clear out the segment.
    fb->Free();
    EXPECT_EQ(fb->size(), 0);
    EXPECT_EQ(fb->data(), nullptr);

    // Safe to call multiple times.
    fb->Free();
  }

  // Load the last few bytes of the data, a different size than the first time.
  {
    Result<FreeableBuffer> fb = fdl->load(
        /*offset=*/sizeof(data) - 3,
        /*size=*/3,
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    ASSERT_EQ(fb.error(), Error::Ok);
    EXPECT_ALIGNED(fb->data(), alignment());
    EXPECT_EQ(fb->size(), 3);
    EXPECT_EQ(0, std::memcmp(fb->data(), "\xfd\xfe\xff", fb->size()));
  }

  // Loading all of the data succeeds.
  {
    Result<FreeableBuffer> fb = fdl->load(
        /*offset=*/0,
        /*size=*/sizeof(data),
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    ASSERT_EQ(fb.error(), Error::Ok);
    EXPECT_ALIGNED(fb->data(), alignment());
    EXPECT_EQ(fb->size(), sizeof(data));
    EXPECT_EQ(0, std::memcmp(fb->data(), data, fb->size()));
  }

  // Loading zero-sized data succeeds, even at the end of the data.
  {
    Result<FreeableBuffer> fb = fdl->load(
        /*offset=*/sizeof(data),
        /*size=*/0,
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    ASSERT_EQ(fb.error(), Error::Ok);
    EXPECT_EQ(fb->size(), 0);
  }
}

TEST_P(FileDataLoaderTest, OutOfBoundsLoadFails) {
  // Create a temp file; contents don't matter.
  uint8_t data[256] = {};
  TempFile tf(data, sizeof(data));

  Result<FileDataLoader> fdl =
      FileDataLoader::from(tf.path().c_str(), alignment());
  ASSERT_EQ(fdl.error(), Error::Ok);

  // Loading beyond the end of the data should fail.
  {
    Result<FreeableBuffer> fb = fdl->load(
        /*offset=*/0,
        /*size=*/sizeof(data) + 1,
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    EXPECT_NE(fb.error(), Error::Ok);
  }

  // Loading zero bytes still fails if it's past the end of the data.
  {
    Result<FreeableBuffer> fb = fdl->load(
        /*offset=*/sizeof(data) + 1,
        /*size=*/0,
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    EXPECT_NE(fb.error(), Error::Ok);
  }
}

TEST_P(FileDataLoaderTest, FromMissingFileFails) {
  // Wrapping a file that doesn't exist should fail.
  Result<FileDataLoader> fdl = FileDataLoader::from(
      "/tmp/FILE_DOES_NOT_EXIST_EXECUTORCH_MMAP_LOADER_TEST");
  EXPECT_NE(fdl.error(), Error::Ok);
}

TEST_P(FileDataLoaderTest, BadAlignmentFails) {
  // Create a temp file; contents don't matter.
  uint8_t data[256] = {};
  TempFile tf(data, sizeof(data));

  // Creating a loader with default alignment works fine.
  {
    Result<FileDataLoader> fdl = FileDataLoader::from(tf.path().c_str());
    ASSERT_EQ(fdl.error(), Error::Ok);
  }

  // Bad alignments fail.
  const std::vector<size_t> bad_alignments = {0, 3, 5, 17};
  for (size_t bad_alignment : bad_alignments) {
    Result<FileDataLoader> fdl =
        FileDataLoader::from(tf.path().c_str(), bad_alignment);
    ASSERT_EQ(fdl.error(), Error::InvalidArgument);
  }
}

// Tests that the move ctor works.
TEST_P(FileDataLoaderTest, MoveCtor) {
  // Create a loader.
  std::string contents = "FILE_CONTENTS";
  TempFile tf(contents);
  Result<FileDataLoader> fdl =
      FileDataLoader::from(tf.path().c_str(), alignment());
  ASSERT_EQ(fdl.error(), Error::Ok);
  EXPECT_EQ(fdl->size().get(), contents.size());

  // Move it into another instance.
  FileDataLoader fdl2(std::move(*fdl));

  // Old loader should now be invalid.
  EXPECT_EQ(
      fdl->load(
             0,
             0,
             DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program))
          .error(),
      Error::InvalidState);
  EXPECT_EQ(fdl->size().error(), Error::InvalidState);

  // New loader should point to the file.
  EXPECT_EQ(fdl2.size().get(), contents.size());
  Result<FreeableBuffer> fb = fdl2.load(
      /*offset=*/0,
      contents.size(),
      DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
  ASSERT_EQ(fb.error(), Error::Ok);
  EXPECT_ALIGNED(fb->data(), alignment());
  ASSERT_EQ(fb->size(), contents.size());
  EXPECT_EQ(0, std::memcmp(fb->data(), contents.data(), fb->size()));
}

// Test that the deprecated From method (capital 'F') still works.
TEST_P(FileDataLoaderTest, DEPRECATEDFrom) {
  // Write some heterogeneous data to a file.
  uint8_t data[256];
  for (int i = 0; i < sizeof(data); ++i) {
    data[i] = i;
  }
  TempFile tf(data, sizeof(data));

  // Wrap it in a loader.
  Result<FileDataLoader> fdl =
      FileDataLoader::From(tf.path().c_str(), alignment());
  ASSERT_EQ(fdl.error(), Error::Ok);

  // size() should succeed and reflect the total size.
  Result<size_t> size = fdl->size();
  ASSERT_EQ(size.error(), Error::Ok);
  EXPECT_EQ(*size, sizeof(data));
}

// Run all FileDataLoaderTests multiple times, varying the return value of
// `GetParam()` based on the `testing::Values` list. The tests will interpret
// the value as "alignment".
INSTANTIATE_TEST_SUITE_P(
    VariedSegments,
    FileDataLoaderTest,
    testing::Values(
        1,
        4,
        alignof(std::max_align_t),
        2 * alignof(std::max_align_t),
        128,
        1024));
