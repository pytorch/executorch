/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include <executorch/backends/apple/metal/runtime/shims/memory.h>
#include <executorch/backends/apple/metal/runtime/shims/utils.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/platform.h>

using namespace executorch::backends::metal;
using executorch::runtime::Error;

namespace {

// ScalarType::Float, as AOTInductor passes it to the shims.
constexpr int32_t kFloat32 = 6;
// DeviceType::MPS.
constexpr int32_t kDeviceMps = 13;

} // namespace

class MetalMemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();
  }

  void TearDown() override {
    cleanup_memory();
  }

  // Wraps `data` the way a model's constants are wrapped: a tensor that
  // borrows memory it does not own.
  Error createFromBlob(void* data, AOTITensorHandle* tensor) {
    const std::vector<int64_t> sizes = {2, 2};
    const std::vector<int64_t> strides = {2, 1};
    return aoti_torch_create_tensor_from_blob_v2(
        data,
        static_cast<int64_t>(sizes.size()),
        sizes.data(),
        strides.data(),
        /*storage_offset=*/0,
        kFloat32,
        kDeviceMps,
        /*device_index=*/0,
        tensor,
        /*layout=*/0,
        /*opaque_metadata=*/nullptr,
        /*opaque_metadata_size=*/0);
  }

  std::vector<float> blob_ = std::vector<float>(4, 1.0f);
};

TEST_F(MetalMemoryTest, BlobAddressIsTrackedWhileTensorIsAlive) {
  AOTITensorHandle first = nullptr;
  ASSERT_EQ(createFromBlob(blob_.data(), &first), Error::Ok);

  AOTITensorHandle second = nullptr;
  EXPECT_NE(createFromBlob(blob_.data(), &second), Error::Ok);
}

// A model's constants blob can be mapped at an address a previously destroyed
// model used. cleanup_memory() runs when a model is destroyed, so it must leave
// nothing tracked behind.
TEST_F(MetalMemoryTest, BlobAddressCanBeReusedAfterCleanup) {
  AOTITensorHandle first = nullptr;
  ASSERT_EQ(createFromBlob(blob_.data(), &first), Error::Ok);

  cleanup_memory();

  AOTITensorHandle second = nullptr;
  EXPECT_EQ(createFromBlob(blob_.data(), &second), Error::Ok);
}

TEST_F(MetalMemoryTest, CleanupLeavesNoTrackedMemory) {
  AOTITensorHandle tensor = nullptr;
  ASSERT_EQ(createFromBlob(blob_.data(), &tensor), Error::Ok);
  ASSERT_FALSE(memory_to_n_tensor.empty());

  cleanup_memory();

  EXPECT_TRUE(tensors.empty());
  EXPECT_TRUE(memory_to_n_tensor.empty());
}

class MetalStrideTest : public MetalMemoryTest {
 protected:
  AOTITensorHandle make(
      std::vector<int64_t> sizes,
      std::vector<int64_t> strides,
      int32_t device_type = kDeviceMps) {
    AOTITensorHandle tensor = nullptr;
    EXPECT_EQ(
        aoti_torch_empty_strided(
            static_cast<int64_t>(sizes.size()),
            sizes.data(),
            strides.data(),
            kFloat32,
            device_type,
            0,
            &tensor),
        Error::Ok);
    return tensor;
  }
};

// The stride of a size-1 dimension is not looked at, as in PyTorch.
TEST_F(MetalStrideTest, RowMajorDenseIgnoresSizeOneDims) {
  EXPECT_TRUE(is_row_major_dense(*make({2, 1, 4}, {4, 4, 1})));
  EXPECT_TRUE(is_row_major_dense(*make({2, 1, 4}, {4, 1, 1})));
  EXPECT_TRUE(is_row_major_dense(*make({1, 8}, {1, 1})));
  EXPECT_FALSE(is_row_major_dense(*make({4, 2}, {1, 4})));
  EXPECT_FALSE(is_row_major_dense(*make({2, 1, 4}, {1, 1, 2})));
}

// Copying between tensors that differ only in the stride of a size-1
// dimension is a plain copy.
TEST_F(MetalStrideTest, CopyIgnoresStrideOfSizeOneDim) {
  AOTITensorHandle src = make({2, 1, 4}, {4, 1, 1});
  AOTITensorHandle dst = make({2, 1, 4}, {4, 4, 1}, /*device_type=*/0);
  float* src_data = static_cast<float*>(src->mutable_data_ptr());
  for (int i = 0; i < 8; i++) {
    src_data[i] = static_cast<float>(i);
  }

  ASSERT_EQ(aoti_torch_copy_(dst, src, 0), Error::Ok);
  const float* got = static_cast<const float*>(dst->const_data_ptr());
  EXPECT_EQ(
      std::vector<float>(got, got + 8),
      (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7}));
}
