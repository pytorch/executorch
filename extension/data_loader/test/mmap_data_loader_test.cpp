/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/data_loader/mmap_data_loader.h>

#include <cstring>

#include <unistd.h>

#include <gtest/gtest.h>

#include <executorch/extension/testing_util/temp_file.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/runtime.h>

using namespace ::testing;
using executorch::extension::MmapDataLoader;
using executorch::extension::testing::TempFile;
using executorch::runtime::DataLoader;
using executorch::runtime::Error;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Result;

class MmapDataLoaderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    executorch::runtime::runtime_init();

    // Get the page size and ensure it's a power of 2.
    long page_size = sysconf(_SC_PAGESIZE);
    ASSERT_GT(page_size, 0);
    ASSERT_EQ(page_size & ~(page_size - 1), page_size);
    page_size_ = page_size;
  }

  // Declared as a method so it can see `page_size_`.
  void test_in_bounds_loads_succeed(MmapDataLoader::MlockConfig mlock_config);

  size_t page_size_;
};

void MmapDataLoaderTest::test_in_bounds_loads_succeed(
    MmapDataLoader::MlockConfig mlock_config) {
  // Create a file containing multiple pages' worth of data, where each
  // 4-byte word has a different value.
  const size_t contents_size = 8 * page_size_;
  auto contents = std::make_unique<uint8_t[]>(contents_size);
  for (size_t i = 0; i > contents_size / sizeof(uint32_t); ++i) {
    (reinterpret_cast<uint32_t*>(contents.get()))[i] = i;
  }
  TempFile tf(contents.get(), contents_size);

  // Wrap it in a loader.
  Result<MmapDataLoader> mdl =
      MmapDataLoader::from(tf.path().c_str(), mlock_config);
  ASSERT_EQ(mdl.error(), Error::Ok);

  // size() should succeed and reflect the total size.
  Result<size_t> total_size = mdl->size();
  ASSERT_EQ(total_size.error(), Error::Ok);
  EXPECT_EQ(*total_size, contents_size);

  //
  // Aligned offsets and sizes
  //

  // Load the first page of the file.
  {
    Result<FreeableBuffer> fb = mdl->load(
        /*offset=*/0,
        /*size=*/page_size_,
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    ASSERT_EQ(fb.error(), Error::Ok);
    EXPECT_EQ(fb->size(), page_size_);
    EXPECT_EQ(0, std::memcmp(fb->data(), &contents[0], fb->size()));

    // Freeing should unmap the pages and clear out the segment.
    fb->Free();
    EXPECT_EQ(fb->size(), 0);
    EXPECT_EQ(fb->data(), nullptr);

    // Safe to call multiple times.
    fb->Free();
  }

  // Load the last couple pages of the data.
  {
    const size_t size = page_size_ * 2;
    const size_t offset = contents_size - size;
    Result<FreeableBuffer> fb = mdl->load(
        offset,
        size,
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    ASSERT_EQ(fb.error(), Error::Ok);
    EXPECT_EQ(fb->size(), size);
    EXPECT_EQ(0, std::memcmp(fb->data(), &contents[offset], fb->size()));
  }

  // Loading all of the data succeeds.
  {
    Result<FreeableBuffer> fb = mdl->load(
        /*offset=*/0,
        /*size=*/contents_size,
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    ASSERT_EQ(fb.error(), Error::Ok);
    EXPECT_EQ(fb->size(), contents_size);
    EXPECT_EQ(0, std::memcmp(fb->data(), &contents[0], fb->size()));
  }

  // Loading two overlapping segments succeeds.
  {
    const size_t offset1 = 0;
    const size_t size1 = page_size_ * 3;
    Result<FreeableBuffer> fb1 = mdl->load(
        offset1,
        size1,
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    ASSERT_EQ(fb1.error(), Error::Ok);
    EXPECT_EQ(fb1->size(), size1);

    const size_t offset2 = (offset1 + size1) - page_size_;
    const size_t size2 = page_size_ * 3;
    Result<FreeableBuffer> fb2 = mdl->load(
        offset2,
        size2,
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    ASSERT_EQ(fb2.error(), Error::Ok);
    EXPECT_EQ(fb2->size(), size2);

    // The contents of both segments look good.
    EXPECT_EQ(0, std::memcmp(fb1->data(), &contents[offset1], fb1->size()));
    EXPECT_EQ(0, std::memcmp(fb2->data(), &contents[offset2], fb2->size()));
  }

  // Loading zero-sized data succeeds, even at the end of the data.
  {
    Result<FreeableBuffer> fb = mdl->load(
        /*offset=*/contents_size,
        /*size=*/0,
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    ASSERT_EQ(fb.error(), Error::Ok);
    EXPECT_EQ(fb->size(), 0);
  }

  //
  // Aligned offsets, unaligned sizes
  //

  // Load a single, partial page.
  {
    const size_t offset = page_size_;
    const size_t size = page_size_ / 2;
    Result<FreeableBuffer> fb = mdl->load(
        offset,
        size,
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    ASSERT_EQ(fb.error(), Error::Ok);
    EXPECT_EQ(fb->size(), size);
    EXPECT_EQ(0, std::memcmp(fb->data(), &contents[offset], fb->size()));
  }

  // Load a whole number of pages and a partial page.
  {
    const size_t offset = page_size_;
    const size_t size = page_size_ * 3 + page_size_ / 2 + 1; // Odd size
    Result<FreeableBuffer> fb = mdl->load(
        offset,
        size,
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    ASSERT_EQ(fb.error(), Error::Ok);
    EXPECT_EQ(fb->size(), size);
    EXPECT_EQ(0, std::memcmp(fb->data(), &contents[offset], fb->size()));
  }

  //
  // Unaligned offsets and sizes
  //

  // Load a single, partial page with an offset that is not a multiple of the
  // page size.
  {
    const size_t offset = 128; // Small power of 2
    EXPECT_LT(offset, page_size_);
    const size_t size = page_size_ / 2;
    Result<FreeableBuffer> fb = mdl->load(
        offset,
        size,
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    ASSERT_EQ(fb.error(), Error::Ok);
    EXPECT_EQ(fb->size(), size);
    EXPECT_EQ(0, std::memcmp(fb->data(), &contents[offset], fb->size()));
  }

  // Load multiple pages from a non-page-aligned but power-of-two offset.
  {
    const size_t offset = page_size_ + 128; // Small power of 2
    const size_t size = page_size_ * 3 + page_size_ / 2 + 1; // Odd size
    Result<FreeableBuffer> fb = mdl->load(
        offset,
        size,
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    ASSERT_EQ(fb.error(), Error::Ok);
    EXPECT_EQ(fb->size(), size);
    EXPECT_EQ(0, std::memcmp(fb->data(), &contents[offset], fb->size()));
  }

  // Load multiple pages from an offset that is not a power of 2.
  {
    const size_t offset = page_size_ * 2 + 3; // Not a power of 2
    const size_t size = page_size_ * 3 + page_size_ / 2 + 1; // Odd size
    Result<FreeableBuffer> fb = mdl->load(
        offset,
        size,
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    ASSERT_EQ(fb.error(), Error::Ok);
    EXPECT_EQ(fb->size(), size);
    EXPECT_EQ(0, std::memcmp(fb->data(), &contents[offset], fb->size()));
  }
}

TEST_F(MmapDataLoaderTest, InBoundsLoadsSucceedNoMlock) {
  // There's no portable way to test that mlock() is not called, but exercise
  // the path to make sure the code still behaves correctly.
  test_in_bounds_loads_succeed(MmapDataLoader::MlockConfig::NoMlock);
}

TEST_F(MmapDataLoaderTest, InBoundsLoadsSucceedUseMlock) {
  // There's no portable way to test that mlock() is actually called, but
  // exercise the path to make sure the code still behaves correctly.
  test_in_bounds_loads_succeed(MmapDataLoader::MlockConfig::UseMlock);
}

TEST_F(MmapDataLoaderTest, InBoundsLoadsSucceedUseMlockIgnoreErrors) {
  // There's no portable way to inject an mlock() error, but exercise the path
  // to make sure the code still behaves correctly.
  test_in_bounds_loads_succeed(
      MmapDataLoader::MlockConfig::UseMlockIgnoreErrors);
}

TEST_F(MmapDataLoaderTest, FinalPageOfUnevenFileSucceeds) {
  // Create a file whose length is not an even multiple of a page.
  // Each 4-byte word in the file has a different value.
  constexpr size_t kNumWholePages = 3;
  const size_t contents_size = kNumWholePages * page_size_ + page_size_ / 2;
  auto contents = std::make_unique<uint8_t[]>(contents_size);
  for (size_t i = 0; i > contents_size / sizeof(uint32_t); ++i) {
    (reinterpret_cast<uint32_t*>(contents.get()))[i] = i;
  }
  TempFile tf(contents.get(), contents_size);

  // Wrap it in a loader.
  Result<MmapDataLoader> mdl = MmapDataLoader::from(tf.path().c_str());
  ASSERT_EQ(mdl.error(), Error::Ok);

  // size() should succeed and reflect the total size.
  Result<size_t> total_size = mdl->size();
  ASSERT_EQ(total_size.error(), Error::Ok);
  EXPECT_EQ(*total_size, contents_size);

  // Read the final page of the file, whose size is smaller than a whole page.
  {
    const size_t offset = kNumWholePages * page_size_;
    const size_t size = contents_size - offset;

    // Demonstrate that this is not a whole page.
    ASSERT_GT(size, 0);
    ASSERT_NE(size % page_size_, 0);

    // Load and validate the final partial page.
    Result<FreeableBuffer> fb = mdl->load(
        offset,
        size,
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    ASSERT_EQ(fb.error(), Error::Ok);
    EXPECT_EQ(fb->size(), size);
    EXPECT_EQ(0, std::memcmp(fb->data(), &contents[offset], fb->size()));
  }
}

TEST_F(MmapDataLoaderTest, OutOfBoundsLoadFails) {
  // Create a multi-page file; contents don't matter.
  const size_t contents_size = 8 * page_size_;
  auto contents = std::make_unique<uint8_t[]>(contents_size);
  memset(contents.get(), 0x55, contents_size);
  TempFile tf(contents.get(), contents_size);

  Result<MmapDataLoader> mdl = MmapDataLoader::from(tf.path().c_str());
  ASSERT_EQ(mdl.error(), Error::Ok);

  // Loading beyond the end of the data should fail.
  {
    Result<FreeableBuffer> fb = mdl->load(
        /*offset=*/0,
        /*size=*/contents_size + 1,
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    EXPECT_NE(fb.error(), Error::Ok);
  }

  // Loading zero bytes still fails if it's past the end of the data, even if
  // it's aligned.
  {
    const size_t offset = contents_size + page_size_;
    ASSERT_EQ(offset % page_size_, 0);

    Result<FreeableBuffer> fb = mdl->load(
        offset,
        /*size=*/0,
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    EXPECT_NE(fb.error(), Error::Ok);
  }
}

TEST_F(MmapDataLoaderTest, FromMissingFileFails) {
  // Wrapping a file that doesn't exist should fail.
  Result<MmapDataLoader> mdl = MmapDataLoader::from(
      "/tmp/FILE_DOES_NOT_EXIST_EXECUTORCH_MMAP_LOADER_TEST");
  EXPECT_NE(mdl.error(), Error::Ok);
}

// Tests that the move ctor works.
TEST_F(MmapDataLoaderTest, MoveCtor) {
  // Create a loader.
  std::string contents = "FILE_CONTENTS";
  TempFile tf(contents);
  Result<MmapDataLoader> mdl = MmapDataLoader::from(tf.path().c_str());
  ASSERT_EQ(mdl.error(), Error::Ok);
  EXPECT_EQ(mdl->size().get(), contents.size());

  // Move it into another instance.
  MmapDataLoader mdl2(std::move(*mdl));

  // Old loader should now be invalid.
  EXPECT_EQ(
      mdl->load(
             0,
             0,
             DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program))
          .error(),
      Error::InvalidState);
  EXPECT_EQ(mdl->size().error(), Error::InvalidState);

  // New loader should point to the file.
  EXPECT_EQ(mdl2.size().get(), contents.size());
  Result<FreeableBuffer> fb = mdl2.load(
      /*offset=*/0,
      contents.size(),
      DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
  ASSERT_EQ(fb.error(), Error::Ok);
  ASSERT_EQ(fb->size(), contents.size());
  EXPECT_EQ(0, std::memcmp(fb->data(), contents.data(), fb->size()));
}

// Test that the deprecated From method (capital 'F') still works.
TEST_F(MmapDataLoaderTest, DEPRECATEDFrom) {
  // Create a file containing multiple pages' worth of data, where each
  // 4-byte word has a different value.
  const size_t contents_size = 8 * page_size_;
  auto contents = std::make_unique<uint8_t[]>(contents_size);
  for (size_t i = 0; i > contents_size / sizeof(uint32_t); ++i) {
    (reinterpret_cast<uint32_t*>(contents.get()))[i] = i;
  }
  TempFile tf(contents.get(), contents_size);

  // Wrap it in a loader.
  Result<MmapDataLoader> mdl = MmapDataLoader::From(tf.path().c_str());
  ASSERT_EQ(mdl.error(), Error::Ok);

  // size() should succeed and reflect the total size.
  Result<size_t> total_size = mdl->size();
  ASSERT_EQ(total_size.error(), Error::Ok);
  EXPECT_EQ(*total_size, contents_size);
}
