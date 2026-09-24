/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include <executorch/backends/apple/metal/runtime/shims/et_metal.h>
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
// DeviceType::CPU.
constexpr int32_t kDeviceCpu = 0;

} // namespace

extern "C" AOTITorchError aoti_torch_mps_mm_out(
    AOTITensorHandle out,
    AOTITensorHandle self,
    AOTITensorHandle mat2);

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

class MetalGraphViewTest : public MetalMemoryTest {
 protected:
  // An 8-element tensor holding 1..8 on `device_type`, and the 2x2 view of
  // its last four elements.
  void createOffsetMatrix(
      int32_t device_type,
      AOTITensorHandle* base,
      AOTITensorHandle* view) {
    const int64_t base_size = 8;
    ASSERT_EQ(
        aoti_torch_empty_strided(
            1, &base_size, &kStride, kFloat32, device_type, 0, base),
        Error::Ok);
    auto* data = static_cast<float*>((*base)->mutable_data_ptr());
    for (int i = 0; i < 8; i++) {
      data[i] = static_cast<float>(i + 1);
    }
    ASSERT_EQ(
        aoti_torch__reinterpret_tensor(
            *base, 2, kMatrixSizes, kMatrixStrides, /*storage_offset=*/4, view),
        Error::Ok);
  }

  // A 2x2 Metal tensor holding the identity matrix.
  void createIdentity(AOTITensorHandle* identity) {
    ASSERT_EQ(
        aoti_torch_empty_strided(
            2, kMatrixSizes, kMatrixStrides, kFloat32, kDeviceMps, 0, identity),
        Error::Ok);
    auto* data = static_cast<float*>((*identity)->mutable_data_ptr());
    std::fill_n(data, 4, 0.0f);
    data[0] = data[3] = 1.0f;
  }

  void createMatrix(int32_t device_type, AOTITensorHandle* matrix) {
    ASSERT_EQ(
        aoti_torch_empty_strided(
            2, kMatrixSizes, kMatrixStrides, kFloat32, device_type, 0, matrix),
        Error::Ok);
  }

  static constexpr int64_t kMatrixSizes[2] = {2, 2};
  static constexpr int64_t kMatrixStrides[2] = {2, 1};
};

// A graph fed an offset view through an alias settles the stream itself: its
// result is there as soon as the op returns.
TEST_F(MetalGraphViewTest, AliasedGraphSettlesItsOwnWork) {
  AOTITensorHandle base = nullptr;
  AOTITensorHandle view = nullptr;
  createOffsetMatrix(kDeviceMps, &base, &view);
  AOTITensorHandle identity = nullptr;
  createIdentity(&identity);
  AOTITensorHandle out = nullptr;
  createMatrix(kDeviceMps, &out);

  ASSERT_EQ(aoti_torch_mps_mm_out(out, view, identity), Error::Ok);
  EXPECT_TRUE(getCurrentMetalStream()->isEmpty());
  const auto* got = static_cast<const float*>(out->const_data_ptr());
  EXPECT_EQ(std::vector<float>(got, got + 4), (std::vector<float>{5, 6, 7, 8}));
}

// An op that fails after taking an alias for one of its inputs must not leave
// a wait behind for the next, unrelated graph.
TEST_F(MetalGraphViewTest, FailedAliasedGraphLeavesNoWaitBehind) {
  AOTITensorHandle base = nullptr;
  AOTITensorHandle view = nullptr;
  createOffsetMatrix(kDeviceMps, &base, &view);
  AOTITensorHandle unmapped = nullptr;
  createMatrix(kDeviceCpu, &unmapped);
  AOTITensorHandle out = nullptr;
  createMatrix(kDeviceMps, &out);
  EXPECT_NE(aoti_torch_mps_mm_out(out, view, unmapped), Error::Ok);

  AOTITensorHandle identity = nullptr;
  createIdentity(&identity);
  getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);
  ASSERT_EQ(aoti_torch_mps_mm_out(out, identity, identity), Error::Ok);
  // Left pending on the stream, as a graph fed no alias always is.
  EXPECT_FALSE(getCurrentMetalStream()->isEmpty());
  getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);
}

// A view of CPU memory gets a no-copy Metal buffer of its own, which MPSGraph
// ops can read.
TEST_F(MetalGraphViewTest, CpuBackedOffsetViewIsUsableByMm) {
  AOTITensorHandle base = nullptr;
  AOTITensorHandle view = nullptr;
  createOffsetMatrix(kDeviceCpu, &base, &view);
  AOTITensorHandle identity = nullptr;
  createIdentity(&identity);
  AOTITensorHandle out = nullptr;
  createMatrix(kDeviceMps, &out);

  ASSERT_EQ(aoti_torch_mps_mm_out(out, view, identity), Error::Ok);
  getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);
  const auto* got = static_cast<const float*>(out->const_data_ptr());
  EXPECT_EQ(std::vector<float>(got, got + 4), (std::vector<float>{5, 6, 7, 8}));
}

// The Metal buffer of a view of CPU memory stays while any handle to the view
// does, and goes with the last one, before the CPU memory can be freed.
TEST_F(MetalGraphViewTest, CpuBackedViewBufferGoesWithLastHandle) {
  AOTITensorHandle base = nullptr;
  AOTITensorHandle view = nullptr;
  createOffsetMatrix(kDeviceCpu, &base, &view);
  void* view_ptr = view->mutable_data_ptr();
  AOTITensorHandle alias = nullptr;
  ASSERT_EQ(aoti_torch_new_tensor_handle(view, &alias), Error::Ok);

  ASSERT_EQ(aoti_torch_delete_tensor_object(view), Error::Ok);
  EXPECT_TRUE(metal_is_device_pointer(view_ptr));
  ASSERT_EQ(aoti_torch_delete_tensor_object(alias), Error::Ok);
  EXPECT_FALSE(metal_is_device_pointer(view_ptr));
}

// A second view of CPU memory at the same address can be longer than the
// first; the Metal buffer they share has to cover it.
TEST_F(MetalGraphViewTest, CpuBackedViewBufferCoversLongerViewAtSameAddress) {
  AOTITensorHandle base = nullptr;
  AOTITensorHandle view = nullptr;
  const int64_t short_size = 2;
  AOTITensorHandle short_view = nullptr;
  const int64_t base_size = 8;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &base_size, &kStride, kFloat32, kDeviceCpu, 0, &base),
      Error::Ok);
  auto* data = static_cast<float*>(base->mutable_data_ptr());
  for (int i = 0; i < 8; i++) {
    data[i] = static_cast<float>(i + 1);
  }
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          base, 1, &short_size, &kStride, /*storage_offset=*/4, &short_view),
      Error::Ok);
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          base, 2, kMatrixSizes, kMatrixStrides, /*storage_offset=*/4, &view),
      Error::Ok);
  AOTITensorHandle identity = nullptr;
  createIdentity(&identity);
  AOTITensorHandle out = nullptr;
  createMatrix(kDeviceMps, &out);

  ASSERT_EQ(aoti_torch_mps_mm_out(out, view, identity), Error::Ok);
  getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);
  const auto* got = static_cast<const float*>(out->const_data_ptr());
  EXPECT_EQ(std::vector<float>(got, got + 4), (std::vector<float>{5, 6, 7, 8}));
}

// A copy to the CPU sees what a graph still pending on the stream writes.
TEST_F(MetalGraphViewTest, CopyToHostWaitsForPendingWrites) {
  AOTITensorHandle identity = nullptr;
  createIdentity(&identity);
  AOTITensorHandle out = nullptr;
  createMatrix(kDeviceMps, &out);
  std::fill_n(static_cast<float*>(out->mutable_data_ptr()), 4, -1.0f);
  AOTITensorHandle host = nullptr;
  createMatrix(kDeviceCpu, &host);

  ASSERT_EQ(aoti_torch_mps_mm_out(out, identity, identity), Error::Ok);
  ASSERT_EQ(aoti_torch_copy_(host, out, 0), Error::Ok);
  const auto* got = static_cast<const float*>(host->const_data_ptr());
  EXPECT_EQ(std::vector<float>(got, got + 4), (std::vector<float>{1, 0, 0, 1}));
}

// A copy from the CPU does not overwrite what a graph still pending on the
// stream is to read.
TEST_F(MetalGraphViewTest, CopyToDeviceWaitsForPendingReads) {
  AOTITensorHandle identity = nullptr;
  createIdentity(&identity);
  AOTITensorHandle input = nullptr;
  createMatrix(kDeviceMps, &input);
  auto* input_data = static_cast<float*>(input->mutable_data_ptr());
  for (int i = 0; i < 4; i++) {
    input_data[i] = static_cast<float>(i + 1);
  }
  AOTITensorHandle out = nullptr;
  createMatrix(kDeviceMps, &out);
  AOTITensorHandle host = nullptr;
  createMatrix(kDeviceCpu, &host);
  std::fill_n(static_cast<float*>(host->mutable_data_ptr()), 4, 9.0f);

  ASSERT_EQ(aoti_torch_mps_mm_out(out, input, identity), Error::Ok);
  ASSERT_EQ(aoti_torch_copy_(input, host, 0), Error::Ok);
  getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);
  const auto* got = static_cast<const float*>(out->const_data_ptr());
  EXPECT_EQ(std::vector<float>(got, got + 4), (std::vector<float>{1, 2, 3, 4}));
}

// CPU memory under a view is not freed while queued GPU work still reads it
// through the view's no-copy buffer, which does not own the memory.
TEST_F(MetalGraphViewTest, QueuedCpuViewKeepsBackingStorageAlive) {
  auto* stream = getCurrentMetalStream();
  AOTITensorHandle base = nullptr;
  AOTITensorHandle view = nullptr;
  AOTITensorHandle identity = nullptr;
  AOTITensorHandle out = nullptr;
  createOffsetMatrix(kDeviceCpu, &base, &view);
  createIdentity(&identity);
  createMatrix(kDeviceMps, &out);

  ASSERT_EQ(aoti_torch_mps_mm_out(out, view, identity), Error::Ok);
  EXPECT_FALSE(stream->isEmpty());
  ASSERT_EQ(aoti_torch_delete_tensor_object(base), Error::Ok);
  ASSERT_EQ(aoti_torch_delete_tensor_object(view), Error::Ok);
  stream->synchronize(SyncType::COMMIT_AND_WAIT);

  const auto* got = static_cast<const float*>(out->const_data_ptr());
  EXPECT_EQ(std::vector<float>(got, got + 4), (std::vector<float>{5, 6, 7, 8}));
}

// A graph can write CPU memory through the no-copy buffer of a view of it. A
// view of that memory which is not densely packed is copied on the CPU, and
// the copy has to see the write.
TEST_F(MetalGraphViewTest, MaterializingCpuMemoryWaitsForGpuWrites) {
  const int64_t base_size = 12;
  AOTITensorHandle base = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &base_size, &kStride, kFloat32, kDeviceCpu, 0, &base),
      Error::Ok);
  std::fill_n(static_cast<float*>(base->mutable_data_ptr()), 12, 0.0f);
  AOTITensorHandle target = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          base, 2, kMatrixSizes, kMatrixStrides, /*storage_offset=*/4, &target),
      Error::Ok);
  AOTITensorHandle input = nullptr;
  createMatrix(kDeviceMps, &input);
  auto* input_data = static_cast<float*>(input->mutable_data_ptr());
  for (int i = 0; i < 4; i++) {
    input_data[i] = static_cast<float>(i + 1);
  }
  AOTITensorHandle identity = nullptr;
  createIdentity(&identity);
  ASSERT_EQ(aoti_torch_mps_mm_out(target, input, identity), Error::Ok);
  EXPECT_FALSE(getCurrentMetalStream()->isEmpty());

  // Elements 4, 5, 8 and 9: not densely packed, so copied on the CPU.
  const int64_t strides[2] = {4, 1};
  AOTITensorHandle read = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          base, 2, kMatrixSizes, strides, /*storage_offset=*/4, &read),
      Error::Ok);
  const auto* got = static_cast<const float*>(read->const_data_ptr());
  EXPECT_EQ(std::vector<float>(got, got + 4), (std::vector<float>{1, 2, 0, 0}));
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
