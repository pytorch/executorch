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

std::vector<uint8_t> serialized_multi_arch_metadata() {
  std::vector<uint8_t> output(
      cuda::CudaWeightCache::kMultiArchFormatMagic,
      cuda::CudaWeightCache::kMultiArchFormatMagic +
          cuda::CudaWeightCache::kFormatMagicSize);
  append_u32(output, 3); // variants
  append_u32(output, 80);
  append_u32(output, 80);
  append_string(output, "sm80-so");
  append_u32(output, 90);
  append_u32(output, 90);
  append_string(output, "sm90-so");
  append_u32(output, 120);
  append_u32(output, 120);
  append_string(output, "sm120-so");
  append_u32(output, 1); // entries
  append_string(output, "model.weight");
  append_string(output, "storage-key");
  append_u64(output, 24);
  append_u32(output, 6);
  append_u32(output, 1);
  append_u64(output, 0);
  append_u32(output, 2);
  append_u64(output, 2);
  append_u64(output, 3);
  append_u64(output, 3);
  append_u64(output, 1);
  return output;
}

std::vector<uint8_t> serialized_fallback_metadata() {
  std::vector<uint8_t> output(
      cuda::CudaWeightCache::kMultiArchFallbackFormatMagic,
      cuda::CudaWeightCache::kMultiArchFallbackFormatMagic +
          cuda::CudaWeightCache::kFormatMagicSize);
  append_u32(output, 2); // variants
  append_u32(output, 80);
  append_u32(output, 0);
  append_u32(output, 0); // regular
  append_string(output, "sm80-so");
  append_u32(output, 80);
  append_u32(output, 80);
  append_u32(output, 1); // fallback only
  append_string(output, "fallback-so");
  append_u32(output, 0); // entries
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
  ASSERT_EQ(metadata.variants.size(), 1u);
  EXPECT_EQ(metadata.variants[0].target_sm, 0u);
  EXPECT_EQ(metadata.variants[0].so_blob_key, "so-key");
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

TEST(CudaWeightCacheTest, ParsesAndSelectsMultiArchMetadata) {
  const std::vector<uint8_t> bytes = serialized_multi_arch_metadata();
  cuda::CudaWeightCache::Metadata metadata;
  ASSERT_EQ(
      cuda::CudaWeightCache::parse(bytes.data(), bytes.size(), metadata),
      Error::Ok);
  ASSERT_EQ(metadata.variants.size(), 3u);
  EXPECT_EQ(metadata.variants[0].target_sm, 80u);
  EXPECT_EQ(metadata.variants[2].so_blob_key, "sm120-so");

  size_t variant_index = 0;
  bool uses_ptx_fallback = false;
  EXPECT_EQ(
      cuda::CudaWeightCache::select_variant(
          metadata, 120, variant_index, uses_ptx_fallback),
      Error::Ok);
  EXPECT_EQ(variant_index, 2u);
  EXPECT_FALSE(uses_ptx_fallback);

  EXPECT_EQ(
      cuda::CudaWeightCache::select_variant(
          metadata, 100, variant_index, uses_ptx_fallback),
      Error::Ok);
  EXPECT_EQ(variant_index, 0u);
  EXPECT_TRUE(uses_ptx_fallback);
}

TEST(CudaWeightCacheTest, SelectsLowestCompatiblePtxFallback) {
  cuda::CudaWeightCache::Metadata metadata;
  metadata.variants = {
      {90, 90, "sm90-so"},
      {80, 80, "sm80-so"},
      {120, 0, "sm120-so"},
  };
  size_t variant_index = 0;
  bool uses_ptx_fallback = false;
  ASSERT_EQ(
      cuda::CudaWeightCache::select_variant(
          metadata, 120, variant_index, uses_ptx_fallback),
      Error::Ok);
  EXPECT_EQ(metadata.variants[variant_index].target_sm, 120u);
  EXPECT_FALSE(uses_ptx_fallback);

  ASSERT_EQ(
      cuda::CudaWeightCache::select_variant(
          metadata, 100, variant_index, uses_ptx_fallback),
      Error::Ok);
  EXPECT_EQ(metadata.variants[variant_index].target_sm, 80u);
  EXPECT_TRUE(uses_ptx_fallback);
}

TEST(CudaWeightCacheTest, FallbackOnlyVariantNeverWinsNativeMatch) {
  const std::vector<uint8_t> bytes = serialized_fallback_metadata();
  cuda::CudaWeightCache::Metadata metadata;
  ASSERT_EQ(
      cuda::CudaWeightCache::parse(bytes.data(), bytes.size(), metadata),
      Error::Ok);
  ASSERT_EQ(metadata.variants.size(), 2u);
  EXPECT_FALSE(metadata.variants[0].fallback_only);
  EXPECT_TRUE(metadata.variants[1].fallback_only);

  size_t variant_index = 0;
  bool uses_ptx_fallback = false;
  ASSERT_EQ(
      cuda::CudaWeightCache::select_variant(
          metadata, 80, variant_index, uses_ptx_fallback),
      Error::Ok);
  EXPECT_EQ(variant_index, 0u);
  EXPECT_FALSE(uses_ptx_fallback);

  ASSERT_EQ(
      cuda::CudaWeightCache::select_variant(
          metadata, 90, variant_index, uses_ptx_fallback),
      Error::Ok);
  EXPECT_EQ(variant_index, 1u);
  EXPECT_TRUE(uses_ptx_fallback);
}

TEST(CudaWeightCacheTest, RejectsWhenNoVariantIsCompatible) {
  cuda::CudaWeightCache::Metadata metadata;
  metadata.variants = {{120, 0, "sm120-so"}};
  size_t variant_index = 0;
  bool uses_ptx_fallback = false;
  EXPECT_EQ(
      cuda::CudaWeightCache::select_variant(
          metadata, 90, variant_index, uses_ptx_fallback),
      Error::NotSupported);
}

TEST(CudaWeightCacheTest, RejectsDuplicateMultiArchTarget) {
  std::vector<uint8_t> bytes(
      cuda::CudaWeightCache::kMultiArchFormatMagic,
      cuda::CudaWeightCache::kMultiArchFormatMagic +
          cuda::CudaWeightCache::kFormatMagicSize);
  append_u32(bytes, 2);
  append_u32(bytes, 80);
  append_u32(bytes, 80);
  append_string(bytes, "first-so");
  append_u32(bytes, 80);
  append_u32(bytes, 0);
  append_string(bytes, "second-so");
  append_u32(bytes, 0);

  cuda::CudaWeightCache::Metadata metadata;
  EXPECT_EQ(
      cuda::CudaWeightCache::parse(bytes.data(), bytes.size(), metadata),
      Error::InvalidProgram);
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
