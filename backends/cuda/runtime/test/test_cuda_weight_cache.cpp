/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cuda/runtime/cuda_weight_cache.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace cuda = ::executorch::backends::cuda;
using ::executorch::runtime::Error;

namespace {

void append_u32(std::vector<uint8_t>& output, uint32_t value) {
  for (size_t index = 0; index < 4; ++index) {
    output.push_back(static_cast<uint8_t>(value >> (index * 8)));
  }
}

void append_u64(std::vector<uint8_t>& output, uint64_t value) {
  for (size_t index = 0; index < 8; ++index) {
    output.push_back(static_cast<uint8_t>(value >> (index * 8)));
  }
}

void append_string(std::vector<uint8_t>& output, const std::string& value) {
  append_u32(output, static_cast<uint32_t>(value.size()));
  output.insert(output.end(), value.begin(), value.end());
}

std::vector<uint8_t> serialized_metadata(
    uint32_t dtype = 6,
    uint32_t device_type = 1) {
  std::vector<uint8_t> output(
      cuda::CudaWeightCache::kFormatMagic,
      cuda::CudaWeightCache::kFormatMagic +
          cuda::CudaWeightCache::kFormatMagicSize);
  append_string(output, "so-key");
  append_u32(output, 1); // entries
  append_string(output, "model.weight");
  append_string(output, "storage-key");
  append_u64(output, 24); // storage bytes
  append_u32(output, dtype); // dtype
  append_u32(output, device_type); // device type (CUDA)
  append_u64(output, 0); // storage offset
  append_u32(output, 2); // ndim
  append_u64(output, 2);
  append_u64(output, 3);
  append_u64(output, 3);
  append_u64(output, 1);
  return output;
}

} // namespace

TEST(CudaWeightCacheTest, LegacyPayloadIsNotMisdetected) {
  const std::string legacy = "so-key\nweights-key";
  EXPECT_FALSE(
      cuda::CudaWeightCache::is_serialized(legacy.data(), legacy.size()));
}

TEST(CudaWeightCacheTest, ParsesSerializedMetadata) {
  const std::vector<uint8_t> bytes = serialized_metadata();
  cuda::CudaWeightCache::Metadata metadata;
  ASSERT_EQ(
      cuda::CudaWeightCache::parse(bytes.data(), bytes.size(), metadata),
      Error::Ok);
  ASSERT_EQ(metadata.so_blob_key, "so-key");
  ASSERT_EQ(metadata.entries.size(), 1u);
  const auto& entry = metadata.entries[0];
  EXPECT_EQ(entry.fqn, "model.weight");
  EXPECT_EQ(entry.storage_key, "storage-key");
  EXPECT_EQ(entry.storage_nbytes, 24u);
  EXPECT_EQ(entry.dtype, 6);
  EXPECT_EQ(entry.device_type, 1);
  EXPECT_EQ(entry.sizes, (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(entry.strides, (std::vector<int64_t>{3, 1}));
}

TEST(CudaWeightCacheTest, RejectsTruncationAndTrailingData) {
  std::vector<uint8_t> bytes = serialized_metadata();
  cuda::CudaWeightCache::Metadata metadata;
  ASSERT_GT(bytes.size(), 1u);
  EXPECT_EQ(
      cuda::CudaWeightCache::parse(bytes.data(), bytes.size() - 1, metadata),
      Error::InvalidProgram);
  bytes.push_back(0);
  EXPECT_EQ(
      cuda::CudaWeightCache::parse(bytes.data(), bytes.size(), metadata),
      Error::InvalidProgram);
}

TEST(CudaWeightCacheTest, RejectsUnsupportedDtype) {
  const std::vector<uint8_t> bytes =
      serialized_metadata(7); // Double is unsupported.
  cuda::CudaWeightCache::Metadata metadata;
  EXPECT_EQ(
      cuda::CudaWeightCache::parse(bytes.data(), bytes.size(), metadata),
      Error::InvalidProgram);
}

TEST(CudaWeightCacheTest, RejectsUnsupportedDeviceType) {
  const std::vector<uint8_t> bytes = serialized_metadata(6, 2);
  cuda::CudaWeightCache::Metadata metadata;
  EXPECT_EQ(
      cuda::CudaWeightCache::parse(bytes.data(), bytes.size(), metadata),
      Error::InvalidProgram);
}
