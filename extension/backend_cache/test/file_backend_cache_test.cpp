/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/backend_cache/file_backend_cache.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

#include <sys/stat.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using executorch::extension::FileBackendCache;
using executorch::runtime::Error;

namespace {

const char* kBackend = "TestBackend";
constexpr size_t kIdx = 0;

std::string make_temp_dir() {
  std::string tmpl = "/tmp/et_cache_test_XXXXXX";
  char* dir = mkdtemp(&tmpl[0]);
  return std::string(dir);
}

void remove_dir_recursive(const std::string& path) {
  std::string cmd = "rm -rf " + path;
  std::system(cmd.c_str());
}

} // namespace

class FileBackendCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    executorch::runtime::runtime_init();
    cache_dir_ = make_temp_dir();
  }

  void TearDown() override {
    remove_dir_recursive(cache_dir_);
  }

  std::string cache_dir_;
};

TEST_F(FileBackendCacheTest, SaveAndLoad) {
  FileBackendCache cache(cache_dir_);

  const char* key = "test_data";
  const char* data = "hello file cache";
  size_t size = std::strlen(data) + 1;

  auto err = cache.save(kBackend, kIdx, key, data, size);
  EXPECT_EQ(err, Error::Ok);

  auto result = cache.load(kBackend, kIdx, key);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->size(), size);
  EXPECT_STREQ(static_cast<const char*>(result->data()), data);
}

TEST_F(FileBackendCacheTest, LoadNonExistent) {
  FileBackendCache cache(cache_dir_);

  auto result = cache.load(kBackend, kIdx, "nonexistent");
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error(), Error::NotFound);
}

TEST_F(FileBackendCacheTest, SaveCreatesSubdirectories) {
  FileBackendCache cache(cache_dir_);

  const char* key = "v1/packed_weights";
  const char* data = "packed weight data";
  size_t size = std::strlen(data) + 1;

  auto err = cache.save(kBackend, kIdx, key, data, size);
  EXPECT_EQ(err, Error::Ok);

  auto result = cache.load(kBackend, kIdx, key);
  ASSERT_TRUE(result.ok());
  EXPECT_STREQ(static_cast<const char*>(result->data()), data);
}

TEST_F(FileBackendCacheTest, SaveOverwrites) {
  FileBackendCache cache(cache_dir_);

  const char* key = "data";
  const char* data1 = "first version";
  const char* data2 = "second version";

  EXPECT_EQ(
      cache.save(kBackend, kIdx, key, data1, std::strlen(data1) + 1),
      Error::Ok);
  EXPECT_EQ(
      cache.save(kBackend, kIdx, key, data2, std::strlen(data2) + 1),
      Error::Ok);

  auto result = cache.load(kBackend, kIdx, key);
  ASSERT_TRUE(result.ok());
  EXPECT_STREQ(static_cast<const char*>(result->data()), data2);
}

TEST_F(FileBackendCacheTest, LargeData) {
  FileBackendCache cache(cache_dir_);

  const size_t size = 1024 * 1024; // 1MB
  std::vector<uint8_t> data(size);
  for (size_t i = 0; i < size; ++i) {
    data[i] = static_cast<uint8_t>(i & 0xFF);
  }

  auto err = cache.save(kBackend, kIdx, "large", data.data(), size);
  EXPECT_EQ(err, Error::Ok);

  auto result = cache.load(kBackend, kIdx, "large");
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->size(), size);
  EXPECT_EQ(std::memcmp(result->data(), data.data(), size), 0);
}

TEST_F(FileBackendCacheTest, Remove) {
  FileBackendCache cache(cache_dir_);

  const char* key = "removable";
  const char* data = "some data";

  EXPECT_EQ(cache.remove(kBackend, kIdx, key), Error::NotFound);

  EXPECT_EQ(
      cache.save(kBackend, kIdx, key, data, std::strlen(data) + 1), Error::Ok);
  EXPECT_EQ(cache.remove(kBackend, kIdx, key), Error::Ok);

  auto result = cache.load(kBackend, kIdx, key);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error(), Error::NotFound);
}

TEST_F(FileBackendCacheTest, AlignedLoad) {
  FileBackendCache cache(cache_dir_);

  const size_t size = 256;
  std::vector<uint8_t> data(size, 0xAB);

  EXPECT_EQ(
      cache.save(kBackend, kIdx, "aligned", data.data(), size), Error::Ok);

  constexpr size_t kAlignment = 64;
  auto result = cache.load(kBackend, kIdx, "aligned", kAlignment);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->size(), size);
  EXPECT_EQ(
      reinterpret_cast<uintptr_t>(result->data()) % kAlignment,
      static_cast<uintptr_t>(0));
  EXPECT_EQ(std::memcmp(result->data(), data.data(), size), 0);
}

TEST_F(FileBackendCacheTest, IsolatesBackendsAndDelegates) {
  FileBackendCache cache(cache_dir_);

  const char* key = "weights";
  const char* data_a0 = "backend_a delegate 0";
  const char* data_a1 = "backend_a delegate 1";
  const char* data_b0 = "backend_b delegate 0";

  EXPECT_EQ(
      cache.save("BackendA", 0, key, data_a0, std::strlen(data_a0) + 1),
      Error::Ok);
  EXPECT_EQ(
      cache.save("BackendA", 1, key, data_a1, std::strlen(data_a1) + 1),
      Error::Ok);
  EXPECT_EQ(
      cache.save("BackendB", 0, key, data_b0, std::strlen(data_b0) + 1),
      Error::Ok);

  auto r_a0 = cache.load("BackendA", 0, key);
  ASSERT_TRUE(r_a0.ok());
  EXPECT_STREQ(static_cast<const char*>(r_a0->data()), data_a0);

  auto r_a1 = cache.load("BackendA", 1, key);
  ASSERT_TRUE(r_a1.ok());
  EXPECT_STREQ(static_cast<const char*>(r_a1->data()), data_a1);

  auto r_b0 = cache.load("BackendB", 0, key);
  ASSERT_TRUE(r_b0.ok());
  EXPECT_STREQ(static_cast<const char*>(r_b0->data()), data_b0);
}
