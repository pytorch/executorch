/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/data_loader/shared_ptr_data_loader.h>

#include <cstring>
#include <memory>

#include <gtest/gtest.h>

#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/runtime.h>

using namespace ::testing;
using torch::executor::Error;
using torch::executor::FreeableBuffer;
using torch::executor::Result;
using torch::executor::util::SharedPtrDataLoader;

class SharedPtrDataLoaderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    torch::executor::runtime_init();
  }
};

TEST_F(SharedPtrDataLoaderTest, InBoundsLoadsSucceed) {
  // Create some heterogeneous data.
  const size_t SIZE = 256;
  std::shared_ptr<uint8_t[]> data(
      new uint8_t[SIZE], std::default_delete<uint8_t[]>());
  for (int i = 0; i < SIZE; ++i) {
    data[i] = i;
  }

  // Wrap it in a loader.
  SharedPtrDataLoader sbdl(data, SIZE);

  // size() should succeed and reflect the total size.
  Result<size_t> size = sbdl.size();
  EXPECT_TRUE(size.ok());
  EXPECT_EQ(*size, SIZE);

  // Load the first bytes of the data.
  {
    Result<FreeableBuffer> fb = sbdl.Load(/*offset=*/0, /*size=*/8);
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
    Result<FreeableBuffer> fb = sbdl.Load(/*offset=*/SIZE - 3, /*size=*/3);
    EXPECT_TRUE(fb.ok());
    EXPECT_EQ(fb->size(), 3);
    EXPECT_EQ(0, std::memcmp(fb->data(), "\xfd\xfe\xff", fb->size()));
  }

  // Loading all of the data succeeds.
  {
    Result<FreeableBuffer> fb = sbdl.Load(/*offset=*/0, /*size=*/SIZE);
    EXPECT_TRUE(fb.ok());
    EXPECT_EQ(fb->size(), SIZE);
    EXPECT_EQ(0, std::memcmp(fb->data(), data.get(), fb->size()));
  }

  // Loading zero-sized data succeeds, even at the end of the data.
  {
    Result<FreeableBuffer> fb = sbdl.Load(/*offset=*/SIZE, /*size=*/0);
    EXPECT_TRUE(fb.ok());
    EXPECT_EQ(fb->size(), 0);
  }
}

TEST_F(SharedPtrDataLoaderTest, OutOfBoundsLoadFails) {
  // Wrap some data in a loader.
  const size_t SIZE = 256;
  std::shared_ptr<uint8_t[]> data(
      new uint8_t[SIZE], std::default_delete<uint8_t[]>());

  // Wrap it in a loader.
  SharedPtrDataLoader sbdl(data, SIZE);

  // Loading beyond the end of the data should fail.
  {
    Result<FreeableBuffer> fb = sbdl.Load(/*offset=*/0, /*size=*/SIZE + 1);
    EXPECT_NE(fb.error(), Error::Ok);
  }

  // Loading zero bytes still fails if it's past the end of the data.
  {
    Result<FreeableBuffer> fb = sbdl.Load(/*offset=*/SIZE + 1, /*size=*/0);
    EXPECT_NE(fb.error(), Error::Ok);
  }
}
