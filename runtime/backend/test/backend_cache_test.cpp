/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/backend/backend_cache.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

using executorch::runtime::BackendCache;
using executorch::runtime::DelegateBackendCache;
using executorch::runtime::Error;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Result;

namespace {

std::string
make_store_key(const char* backend_id, size_t delegate_index, const char* key) {
  return std::string(backend_id) + "/" + std::to_string(delegate_index) + "/" +
      key;
}

class InMemoryBackendCache : public BackendCache {
 public:
  Result<FreeableBuffer> load(
      const char* backend_id,
      size_t delegate_index,
      const char* key,
      size_t /*alignment*/ = alignof(std::max_align_t)) const override {
    auto store_key = make_store_key(backend_id, delegate_index, key);
    auto it = store_.find(store_key);
    if (it == store_.end()) {
      return Error::NotFound;
    }
    const auto& data = it->second;
    void* copy = std::malloc(data.size());
    if (!copy) {
      return Error::MemoryAllocationFailed;
    }
    std::memcpy(copy, data.data(), data.size());
    return FreeableBuffer(
        copy, data.size(), [](void*, void* d, size_t) { std::free(d); });
  }

  Error save(
      const char* backend_id,
      size_t delegate_index,
      const char* key,
      const void* data,
      size_t size) override {
    auto store_key = make_store_key(backend_id, delegate_index, key);
    auto bytes = static_cast<const uint8_t*>(data);
    store_[store_key] = std::vector<uint8_t>(bytes, bytes + size);
    return Error::Ok;
  }

  Error remove(const char* backend_id, size_t delegate_index, const char* key)
      override {
    auto store_key = make_store_key(backend_id, delegate_index, key);
    return store_.erase(store_key) > 0 ? Error::Ok : Error::NotFound;
  }

  bool has_key(const std::string& key) const {
    return store_.count(key) > 0;
  }

 private:
  std::unordered_map<std::string, std::vector<uint8_t>> store_;
};

} // namespace

class BackendCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    executorch::runtime::runtime_init();
  }
};

TEST_F(BackendCacheTest, InMemorySaveAndLoad) {
  InMemoryBackendCache cache;

  const char* bid = "TestBackend";
  size_t idx = 0;
  const char* key = "test_key";
  const char* data = "hello world";
  size_t size = std::strlen(data) + 1;

  {
    auto result = cache.load(bid, idx, key);
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.error(), Error::NotFound);
  }

  auto err = cache.save(bid, idx, key, data, size);
  EXPECT_EQ(err, Error::Ok);

  auto result = cache.load(bid, idx, key);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->size(), size);
  EXPECT_STREQ(static_cast<const char*>(result->data()), data);
}

TEST_F(BackendCacheTest, LoadNonExistent) {
  InMemoryBackendCache cache;

  auto result = cache.load("Backend", 0, "nonexistent");
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error(), Error::NotFound);
}

TEST_F(BackendCacheTest, SaveOverwrites) {
  InMemoryBackendCache cache;

  const char* bid = "Backend";
  size_t idx = 0;
  const char* key = "key";
  const char* data1 = "first";
  const char* data2 = "second";

  EXPECT_EQ(
      cache.save(bid, idx, key, data1, std::strlen(data1) + 1), Error::Ok);
  EXPECT_EQ(
      cache.save(bid, idx, key, data2, std::strlen(data2) + 1), Error::Ok);

  auto result = cache.load(bid, idx, key);
  ASSERT_TRUE(result.ok());
  EXPECT_STREQ(static_cast<const char*>(result->data()), data2);
}

TEST_F(BackendCacheTest, Remove) {
  InMemoryBackendCache cache;

  const char* bid = "Backend";
  size_t idx = 0;
  const char* key = "key";
  const char* data = "data";

  EXPECT_EQ(cache.remove(bid, idx, key), Error::NotFound);

  EXPECT_EQ(cache.save(bid, idx, key, data, std::strlen(data) + 1), Error::Ok);
  EXPECT_EQ(cache.remove(bid, idx, key), Error::Ok);

  auto result = cache.load(bid, idx, key);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error(), Error::NotFound);
}

TEST_F(BackendCacheTest, ScopedCacheForwardsComponents) {
  InMemoryBackendCache base_cache;
  DelegateBackendCache scoped(&base_cache, "TestBackend", 0);

  const char* key = "packed_weights";
  const char* data = "weight data";
  size_t size = std::strlen(data) + 1;

  {
    auto result = scoped.load(key);
    EXPECT_FALSE(result.ok());
  }

  auto err = scoped.save(key, data, size);
  EXPECT_EQ(err, Error::Ok);

  EXPECT_TRUE(base_cache.has_key("TestBackend/0/packed_weights"));
  EXPECT_FALSE(base_cache.has_key("packed_weights"));

  auto result = scoped.load(key);
  ASSERT_TRUE(result.ok());
  EXPECT_STREQ(static_cast<const char*>(result->data()), data);
}

TEST_F(BackendCacheTest, ScopedCacheIsolatesDelegates) {
  InMemoryBackendCache base_cache;
  DelegateBackendCache scoped0(&base_cache, "Backend", 0);
  DelegateBackendCache scoped1(&base_cache, "Backend", 1);

  const char* key = "model";
  const char* data0 = "delegate0";
  const char* data1 = "delegate1";

  EXPECT_EQ(scoped0.save(key, data0, std::strlen(data0) + 1), Error::Ok);
  EXPECT_EQ(scoped1.save(key, data1, std::strlen(data1) + 1), Error::Ok);

  auto result0 = scoped0.load(key);
  ASSERT_TRUE(result0.ok());
  EXPECT_STREQ(static_cast<const char*>(result0->data()), data0);

  auto result1 = scoped1.load(key);
  ASSERT_TRUE(result1.ok());
  EXPECT_STREQ(static_cast<const char*>(result1->data()), data1);
}

TEST_F(BackendCacheTest, ScopedCacheIsolatesBackends) {
  InMemoryBackendCache base_cache;
  DelegateBackendCache xnnpack(&base_cache, "XnnpackBackend", 0);
  DelegateBackendCache vulkan(&base_cache, "VulkanBackend", 0);

  const char* key = "weights";
  const char* xnnpack_data = "xnnpack weights";
  const char* vulkan_data = "vulkan weights";

  EXPECT_EQ(
      xnnpack.save(key, xnnpack_data, std::strlen(xnnpack_data) + 1),
      Error::Ok);
  EXPECT_EQ(
      vulkan.save(key, vulkan_data, std::strlen(vulkan_data) + 1), Error::Ok);

  auto xnnpack_result = xnnpack.load(key);
  ASSERT_TRUE(xnnpack_result.ok());
  EXPECT_STREQ(static_cast<const char*>(xnnpack_result->data()), xnnpack_data);

  auto vulkan_result = vulkan.load(key);
  ASSERT_TRUE(vulkan_result.ok());
  EXPECT_STREQ(static_cast<const char*>(vulkan_result->data()), vulkan_data);
}

TEST_F(BackendCacheTest, ScopedCacheRemove) {
  InMemoryBackendCache base_cache;
  DelegateBackendCache scoped(&base_cache, "TestBackend", 0);

  const char* key = "weights";
  const char* data = "data";

  EXPECT_EQ(scoped.save(key, data, std::strlen(data) + 1), Error::Ok);
  EXPECT_EQ(scoped.remove(key), Error::Ok);

  auto result = scoped.load(key);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error(), Error::NotFound);
}
