/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include <executorch/backends/apple/metal/runtime/shims/et_metal.h>
#include <executorch/backends/apple/metal/runtime/shims/memory.h>
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

class MetalStridedViewTest : public ::testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();
  }

  void TearDown() override {
    cleanup_memory();
  }

  // A 4x4 Metal tensor holding 0..15, and the view of its right half: a
  // {4, 2} tensor with strides {4, 1} at offset 2, which is not densely
  // packed.
  void createBaseAndRightHalf(AOTITensorHandle* base, AOTITensorHandle* view) {
    const int64_t base_sizes[2] = {4, 4};
    const int64_t base_strides[2] = {4, 1};
    ASSERT_EQ(
        aoti_torch_empty_strided(
            2, base_sizes, base_strides, kFloat32, kDeviceMps, 0, base),
        Error::Ok);
    float* data = static_cast<float*>((*base)->mutable_data_ptr());
    for (int i = 0; i < 16; i++) {
      data[i] = static_cast<float>(i);
    }
    const int64_t view_sizes[2] = {4, 2};
    ASSERT_EQ(
        aoti_torch__reinterpret_tensor(
            *base, 2, view_sizes, base_strides, /*storage_offset=*/2, view),
        Error::Ok);
  }
};

TEST_F(MetalStridedViewTest, NonPackedViewStaysInParentBuffer) {
  AOTITensorHandle base = nullptr;
  AOTITensorHandle view = nullptr;
  createBaseAndRightHalf(&base, &view);

  EXPECT_TRUE(metal_is_strided_view(view));
  EXPECT_EQ(
      view->mutable_data_ptr(),
      static_cast<float*>(base->mutable_data_ptr()) + 2);
  // The tensor carries packed strides; the real ones are recorded aside.
  EXPECT_EQ(view->strides()[0], 2);
  EXPECT_EQ(view->strides()[1], 1);
}

TEST_F(MetalStridedViewTest, PackedCopyGathersTheViewsElements) {
  AOTITensorHandle base = nullptr;
  AOTITensorHandle view = nullptr;
  createBaseAndRightHalf(&base, &view);

  @autoreleasepool {
    id<MTLBuffer> packed = metal_packed_copy_of_strided_view(*view);
    ASSERT_NE(packed, nil);
    getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);

    const std::vector<float> expected = {2, 3, 6, 7, 10, 11, 14, 15};
    std::vector<float> got(expected.size());
    std::memcpy(got.data(), [packed contents], got.size() * sizeof(float));
    EXPECT_EQ(got, expected);
  }
}

TEST_F(MetalStridedViewTest, CopiedHandleIsAStridedViewToo) {
  AOTITensorHandle base = nullptr;
  AOTITensorHandle view = nullptr;
  createBaseAndRightHalf(&base, &view);

  AOTITensorHandle alias = nullptr;
  ASSERT_EQ(aoti_torch_new_tensor_handle(view, &alias), Error::Ok);
  EXPECT_TRUE(metal_is_strided_view(alias));

  ASSERT_EQ(aoti_torch_delete_tensor_object(view), Error::Ok);
  EXPECT_TRUE(metal_is_strided_view(alias));
  ASSERT_EQ(aoti_torch_delete_tensor_object(alias), Error::Ok);
  EXPECT_FALSE(metal_is_strided_view(alias));
}
