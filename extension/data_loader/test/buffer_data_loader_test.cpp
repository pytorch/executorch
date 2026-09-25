/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/data_loader/buffer_data_loader.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>

#include <gtest/gtest.h>

#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/runtime.h>

using namespace ::testing;
using executorch::extension::BufferDataLoader;
using executorch::runtime::DataLoader;
using executorch::runtime::Error;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Result;

class BufferDataLoaderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    executorch::runtime::runtime_init();
  }
};

TEST_F(BufferDataLoaderTest, InBoundsLoadsSucceed) {
  // Create some heterogeneous data.
  uint8_t data[256];
  for (int i = 0; i < sizeof(data); ++i) {
    data[i] = i;
  }

  // Wrap it in a loader.
  BufferDataLoader edl(data, sizeof(data));

  // size() should succeed and reflect the total size.
  Result<size_t> size = edl.size();
  EXPECT_TRUE(size.ok());
  EXPECT_EQ(*size, sizeof(data));

  // Load the first bytes of the data.
  {
    Result<FreeableBuffer> fb = edl.load(
        /*offset=*/0,
        /*size=*/8,
        /*segment_info=*/
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    EXPECT_TRUE(fb.ok());
    EXPECT_EQ(fb->size(), 8);
    EXPECT_EQ(
        0,
        std::memcmp(
            fb->data(),
            "\x00\x01\x02\x03"
            "\x04\x05\x06\x07",
            fb->size()));

    // Freeing should be a no-op but should still clear out the data/size.
    fb->Free();
    EXPECT_EQ(fb->size(), 0);
    EXPECT_EQ(fb->data(), nullptr);

    // Safe to call multiple times.
    fb->Free();
  }

  // Load the last few bytes of the data, a different size than the first time.
  {
    Result<FreeableBuffer> fb = edl.load(
        /*offset=*/sizeof(data) - 3,
        /*size=*/3,
        /*segment_info=*/
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    EXPECT_TRUE(fb.ok());
    EXPECT_EQ(fb->size(), 3);
    EXPECT_EQ(0, std::memcmp(fb->data(), "\xfd\xfe\xff", fb->size()));
  }

  // Loading all of the data succeeds.
  {
    Result<FreeableBuffer> fb = edl.load(
        /*offset=*/0,
        /*size=*/sizeof(data),
        /*segment_info*/
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    EXPECT_TRUE(fb.ok());
    EXPECT_EQ(fb->size(), sizeof(data));
    EXPECT_EQ(0, std::memcmp(fb->data(), data, fb->size()));
  }

  // Loading zero-sized data succeeds, even at the end of the data.
  {
    Result<FreeableBuffer> fb = edl.load(
        /*offset=*/sizeof(data),
        /*size=*/0,
        /*segment_info*/
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    EXPECT_TRUE(fb.ok());
    EXPECT_EQ(fb->size(), 0);
  }
}

TEST_F(BufferDataLoaderTest, OutOfBoundsLoadFails) {
  // Wrap some data in a loader.
  uint8_t data[256] = {};
  BufferDataLoader edl(data, sizeof(data));

  // Loading beyond the end of the data should fail.
  {
    Result<FreeableBuffer> fb = edl.load(
        /*offset=*/0,
        /*size=*/sizeof(data) + 1,
        /*segment_info=*/
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    EXPECT_NE(fb.error(), Error::Ok);
  }

  // Loading zero bytes still fails if it's past the end of the data.
  {
    Result<FreeableBuffer> fb = edl.load(
        /*offset=*/sizeof(data) + 1,
        /*size=*/0,
        /*segment_info=*/
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    EXPECT_NE(fb.error(), Error::Ok);
  }
}

TEST_F(BufferDataLoaderTest, OverflowLoadFails) {
  // Wrap some data in a loader.
  uint8_t data[256] = {};
  BufferDataLoader edl(data, sizeof(data));

  // Loading with offset + size that would overflow should fail.
  // Use a small valid offset but a size that causes overflow.
  // If overflow wasn't checked, 1 + SIZE_MAX would wrap to 0, which is <= 256.
  {
    Result<FreeableBuffer> fb = edl.load(
        /*offset=*/1,
        /*size=*/SIZE_MAX,
        /*segment_info=*/
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    EXPECT_NE(fb.error(), Error::Ok);
  }

  // Another overflow case: offset within bounds, size causes overflow.
  // 128 + (SIZE_MAX - 127) wraps to 0.
  {
    Result<FreeableBuffer> fb = edl.load(
        /*offset=*/128,
        /*size=*/SIZE_MAX - 127,
        /*segment_info=*/
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    EXPECT_NE(fb.error(), Error::Ok);
  }
}

TEST_F(BufferDataLoaderTest, LoadIntoNullDstFails) {
  // Wrap some data in a loader.
  uint8_t data[256] = {};
  BufferDataLoader edl(data, sizeof(data));

  // Loading beyond the end of the data should fail.
  {
    Result<FreeableBuffer> fb = edl.load_into(
        /*offset=*/0,
        /*size=*/1,
        /*segment_info=*/
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program),
        nullptr);
    EXPECT_NE(fb.error(), Error::Ok);
  }

  // Loading zero bytes still fails if dst is null.
  {
    Result<FreeableBuffer> fb = edl.load_into(
        /*offset=*/0,
        /*size=*/0,
        /*segment_info=*/
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program),
        nullptr);
    EXPECT_NE(fb.error(), Error::Ok);
  }
}

TEST_F(BufferDataLoaderTest, InBoundsLoadIntoSucceeds) {
  // Wrap some data in a loader.
  uint8_t data[256] = {};
  data[0] = 1;
  uint8_t buffer[256] = {};
  buffer[0] = 0;
  BufferDataLoader edl(data, sizeof(data));

  {
    // Buffer contains 0 before load_into.
    EXPECT_EQ(buffer[0], 0);
    Error fb = edl.load_into(
        /*offset=*/0,
        /*size=*/1,
        /*segment_info=*/
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program),
        buffer);
    EXPECT_EQ(fb, Error::Ok);
    // Buffer contains 1 after load_into.
    EXPECT_EQ(buffer[0], 1);
    // Data is unaltered.
    EXPECT_EQ(data[0], 1);
  }
}

TEST_F(BufferDataLoaderTest, WideInterfaceForwardsToLegacyMethods) {
  std::array<uint8_t, 3> data{1, 2, 3};
  BufferDataLoader loader(data.data(), data.size());
  const DataLoader::SegmentInfo segment_info(
      DataLoader::SegmentInfo::Type::Program);

  Result<FreeableBuffer> loaded =
      loader.load_at_offset(1, 2, segment_info);
  ASSERT_TRUE(loaded.ok());
  const std::array<uint8_t, 2> expected{2, 3};
  EXPECT_EQ(
      std::memcmp(loaded->data(), expected.data(), expected.size()), 0);

  std::array<uint8_t, 2> destination{};
  EXPECT_EQ(
      loader.load_into_at_offset(
          1, destination.size(), segment_info, destination.data()),
      Error::Ok);
  EXPECT_EQ(destination, expected);

  Result<uint64_t> source_size = loader.source_size();
  ASSERT_TRUE(source_size.ok());
  EXPECT_EQ(source_size.get(), sizeof(data));
}

TEST_F(BufferDataLoaderTest, WideInterfaceRejectsUnrepresentableRange) {
  std::array<uint8_t, 1> data{};
  BufferDataLoader loader(data.data(), data.size());
  const DataLoader::SegmentInfo segment_info(
      DataLoader::SegmentInfo::Type::Program);

  EXPECT_EQ(
      loader
          .load_at_offset(
              std::numeric_limits<uint64_t>::max(), 1, segment_info)
          .error(),
      Error::NotSupported);
  EXPECT_EQ(
      loader.load_into_at_offset(
          std::numeric_limits<uint64_t>::max(), 1, segment_info, data.data()),
      Error::NotSupported);
}

#if SIZE_MAX < UINT64_MAX
TEST_F(BufferDataLoaderTest, WideInterfaceRejectsUnrepresentableOffsets) {
  std::array<uint8_t, 1> data{};
  BufferDataLoader loader(data.data(), data.size());
  const DataLoader::SegmentInfo segment_info(
      DataLoader::SegmentInfo::Type::Program);
  constexpr uint64_t kUnrepresentableOffset =
      static_cast<uint64_t>(SIZE_MAX) + 1;

  EXPECT_EQ(
      loader.load_at_offset(kUnrepresentableOffset, 0, segment_info).error(),
      Error::NotSupported);
  EXPECT_EQ(
      loader.load_into_at_offset(
          kUnrepresentableOffset, 0, segment_info, data.data()),
      Error::NotSupported);
}
#endif
