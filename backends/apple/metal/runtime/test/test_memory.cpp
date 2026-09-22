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

  // Allocates an 8-element Metal buffer and a view of its last 4 elements.
  void createBaseAndView(AOTITensorHandle* base, AOTITensorHandle* view) {
    const int64_t base_size = 8;
    ASSERT_EQ(
        aoti_torch_empty_strided(
            1, &base_size, &kStride, kFloat32, kDeviceMps, 0, base),
        Error::Ok);
    ASSERT_EQ(
        aoti_torch__reinterpret_tensor(
            *base, 1, &kViewSize, &kStride, /*storage_offset=*/4, view),
        Error::Ok);
    ASSERT_NE((*view)->mutable_data_ptr(), (*base)->mutable_data_ptr());
    ASSERT_TRUE(metal_is_device_pointer((*view)->mutable_data_ptr()));
  }

  // A second handle at the address of `view`, made the way inductor's wrapper
  // makes one: by copying the handle, or by reinterpreting at offset 0.
  Error createAlias(
      AOTITensorHandle view,
      bool by_reinterpret,
      AOTITensorHandle* alias) {
    if (by_reinterpret) {
      return aoti_torch__reinterpret_tensor(
          view, 1, &kViewSize, &kStride, /*storage_offset=*/0, alias);
    }
    return aoti_torch_new_tensor_handle(view, alias);
  }

  // Deleting one of two handles to the same view must leave the view bound to
  // its parent's Metal buffer for the other handle.
  void expectViewOutlivesDeletedHandle(bool by_reinterpret, bool delete_view) {
    AOTITensorHandle base = nullptr;
    AOTITensorHandle view = nullptr;
    createBaseAndView(&base, &view);
    AOTITensorHandle alias = nullptr;
    ASSERT_EQ(createAlias(view, by_reinterpret, &alias), Error::Ok);
    void* view_ptr = view->mutable_data_ptr();
    ASSERT_EQ(alias->mutable_data_ptr(), view_ptr);

    ASSERT_EQ(
        aoti_torch_delete_tensor_object(delete_view ? view : alias), Error::Ok);
    EXPECT_TRUE(metal_is_device_pointer(view_ptr));

    ASSERT_EQ(
        aoti_torch_delete_tensor_object(delete_view ? alias : view), Error::Ok);
    EXPECT_FALSE(metal_is_device_pointer(view_ptr));
  }

  static constexpr int64_t kViewSize = 4;
  static constexpr int64_t kStride = 1;

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

TEST_F(MetalMemoryTest, ViewOutlivesDeletedOriginalOfCopiedHandle) {
  expectViewOutlivesDeletedHandle(
      /*by_reinterpret=*/false, /*delete_view=*/true);
}

TEST_F(MetalMemoryTest, ViewOutlivesDeletedCopiedHandle) {
  expectViewOutlivesDeletedHandle(
      /*by_reinterpret=*/false, /*delete_view=*/false);
}

TEST_F(MetalMemoryTest, ViewOutlivesDeletedOriginalOfSameAddressReinterpret) {
  expectViewOutlivesDeletedHandle(
      /*by_reinterpret=*/true, /*delete_view=*/true);
}

TEST_F(MetalMemoryTest, ViewOutlivesDeletedSameAddressReinterpret) {
  expectViewOutlivesDeletedHandle(
      /*by_reinterpret=*/true, /*delete_view=*/false);
}

// A view at another address counts towards its parent's memory, and the
// parent must not be freed under it. Once the last handle on that memory is
// gone, view or parent, the buffer has to be released.
TEST_F(MetalMemoryTest, ParentFreedAfterViewDeletedLast) {
  AOTITensorHandle base = nullptr;
  AOTITensorHandle view = nullptr;
  createBaseAndView(&base, &view);
  void* base_ptr = base->mutable_data_ptr();

  ASSERT_EQ(aoti_torch_delete_tensor_object(base), Error::Ok);
  EXPECT_TRUE(metal_is_device_pointer(base_ptr));
  EXPECT_TRUE(metal_is_device_pointer(view->mutable_data_ptr()));

  ASSERT_EQ(aoti_torch_delete_tensor_object(view), Error::Ok);
  EXPECT_FALSE(metal_is_device_pointer(base_ptr));
  EXPECT_EQ(memory_to_n_tensor.count(base_ptr), 0u);
}

TEST_F(MetalMemoryTest, ParentFreedAfterParentDeletedLast) {
  AOTITensorHandle base = nullptr;
  AOTITensorHandle view = nullptr;
  createBaseAndView(&base, &view);
  void* base_ptr = base->mutable_data_ptr();

  ASSERT_EQ(aoti_torch_delete_tensor_object(view), Error::Ok);
  EXPECT_TRUE(metal_is_device_pointer(base_ptr));

  ASSERT_EQ(aoti_torch_delete_tensor_object(base), Error::Ok);
  EXPECT_FALSE(metal_is_device_pointer(base_ptr));
  EXPECT_EQ(memory_to_n_tensor.count(base_ptr), 0u);
}

// A view of a view lives in the same allocation, and keeps it alive after the
// base and the first view are gone.
TEST_F(MetalMemoryTest, NestedViewKeepsParentAlive) {
  for (int64_t nested_offset : {0, 2}) {
    AOTITensorHandle base = nullptr;
    AOTITensorHandle view = nullptr;
    createBaseAndView(&base, &view);
    void* base_ptr = base->mutable_data_ptr();

    const int64_t nested_size = 2;
    AOTITensorHandle nested = nullptr;
    ASSERT_EQ(
        aoti_torch__reinterpret_tensor(
            view, 1, &nested_size, &kStride, nested_offset, &nested),
        Error::Ok);
    void* nested_ptr = nested->mutable_data_ptr();

    ASSERT_EQ(aoti_torch_delete_tensor_object(base), Error::Ok);
    ASSERT_EQ(aoti_torch_delete_tensor_object(view), Error::Ok);
    EXPECT_TRUE(metal_is_device_pointer(base_ptr)) << nested_offset;
    EXPECT_TRUE(metal_is_device_pointer(nested_ptr)) << nested_offset;

    ASSERT_EQ(aoti_torch_delete_tensor_object(nested), Error::Ok);
    EXPECT_FALSE(metal_is_device_pointer(base_ptr)) << nested_offset;
    EXPECT_EQ(memory_to_n_tensor.count(base_ptr), 0u) << nested_offset;
  }
}

// A handle copied from a view keeps the allocation alive after the base and
// the original view are gone.
TEST_F(MetalMemoryTest, CopiedViewHandleKeepsParentAlive) {
  AOTITensorHandle base = nullptr;
  AOTITensorHandle view = nullptr;
  createBaseAndView(&base, &view);
  void* base_ptr = base->mutable_data_ptr();
  AOTITensorHandle alias = nullptr;
  ASSERT_EQ(aoti_torch_new_tensor_handle(view, &alias), Error::Ok);

  ASSERT_EQ(aoti_torch_delete_tensor_object(base), Error::Ok);
  ASSERT_EQ(aoti_torch_delete_tensor_object(view), Error::Ok);
  EXPECT_TRUE(metal_is_device_pointer(base_ptr));
  EXPECT_TRUE(metal_is_device_pointer(alias->mutable_data_ptr()));

  ASSERT_EQ(aoti_torch_delete_tensor_object(alias), Error::Ok);
  EXPECT_FALSE(metal_is_device_pointer(base_ptr));
  EXPECT_EQ(memory_to_n_tensor.count(base_ptr), 0u);
}
