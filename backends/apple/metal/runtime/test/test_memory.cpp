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
#include <functional>
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

  // Encodes a hand-written kernel copying `n` floats from `in` to `out`,
  // leaving it queued on the stream. Unlike a graph fed an offset view, a
  // kernel binds the view's buffer at an offset and does not wait.
  void queueCopy(Tensor& in, Tensor& out, uint64_t n) {
    static ETMetalShaderLibrary library(R"(
      #include <metal_stdlib>
      using namespace metal;
      kernel void copy_float(
          device const float* in [[buffer(0)]],
          device float* out [[buffer(1)]],
          uint i [[thread_position_in_grid]]) {
        out[i] = in[i];
      }
    )");
    auto copy = library.getKernelFunction("copy_float");
    ASSERT_NE(copy, nullptr);
    copy->runCommandBlock([&]() {
      copy->startEncoding();
      copy->setArg(0, in);
      copy->setArg(1, out);
      copy->dispatchSingle(n);
    });
  }

  // Queues a large copy, submits it with `submit`, then waits: the copy must
  // be done, since the pool takes a completed wait to mean that nothing
  // queued before it still runs.
  void expectWaitAfterSubmitWaitsForWork(const std::function<void()>& submit) {
    constexpr int64_t kCount = 1 << 24;
    AOTITensorHandle input = nullptr;
    ASSERT_EQ(
        aoti_torch_empty_strided(
            1, &kCount, &kStride, kFloat32, kDeviceMps, 0, &input),
        Error::Ok);
    std::fill_n(static_cast<float*>(input->mutable_data_ptr()), kCount, 1.0f);
    AOTITensorHandle out = nullptr;
    ASSERT_EQ(
        aoti_torch_empty_strided(
            1, &kCount, &kStride, kFloat32, kDeviceMps, 0, &out),
        Error::Ok);
    auto* out_data = static_cast<float*>(out->mutable_data_ptr());
    std::fill_n(out_data, kCount, 0.0f);

    queueCopy(*input, *out, kCount);
    submit();
    getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);
    EXPECT_EQ(out_data[kCount - 1], 1.0f);
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
// through the no-copy buffer the view is bound into, which does not own the
// memory.
TEST_F(MetalGraphViewTest, QueuedCpuViewKeepsBackingStorageAlive) {
  AOTITensorHandle base = nullptr;
  AOTITensorHandle view = nullptr;
  AOTITensorHandle out = nullptr;
  createOffsetMatrix(kDeviceCpu, &base, &view);
  createMatrix(kDeviceMps, &out);

  queueCopy(*view, *out, 4);
  ASSERT_EQ(aoti_torch_delete_tensor_object(base), Error::Ok);
  ASSERT_EQ(aoti_torch_delete_tensor_object(view), Error::Ok);
  getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);

  const auto* got = static_cast<const float*>(out->const_data_ptr());
  EXPECT_EQ(std::vector<float>(got, got + 4), (std::vector<float>{5, 6, 7, 8}));
}

// A kernel can write CPU memory through the no-copy buffer a view of it is
// bound into. A view of that memory which is not densely packed is copied on
// the CPU, and the copy has to see the write.
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
  queueCopy(*input, *target, 4);

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

// CPU memory the GPU has no buffer over cannot have pending GPU writes, so
// materializing a view of it leaves unrelated queued work alone.
TEST_F(MetalGraphViewTest, MaterializingUnreachableCpuMemoryDoesNotWait) {
  const int64_t base_size = 12;
  AOTITensorHandle base = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &base_size, &kStride, kFloat32, kDeviceCpu, 0, &base),
      Error::Ok);
  auto* data = static_cast<float*>(base->mutable_data_ptr());
  for (int i = 0; i < 12; i++) {
    data[i] = static_cast<float>(i + 1);
  }
  AOTITensorHandle input = nullptr;
  createMatrix(kDeviceMps, &input);
  AOTITensorHandle out = nullptr;
  createMatrix(kDeviceMps, &out);
  queueCopy(*input, *out, 4);

  const int64_t strides[2] = {4, 1};
  AOTITensorHandle read = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          base, 2, kMatrixSizes, strides, /*storage_offset=*/4, &read),
      Error::Ok);
  EXPECT_FALSE(getCurrentMetalStream()->isEmpty());
  const auto* got = static_cast<const float*>(read->const_data_ptr());
  EXPECT_EQ(
      std::vector<float>(got, got + 4), (std::vector<float>{5, 6, 9, 10}));
  getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);
}

// Freed Metal buffers go back to the pool without a wait. When the packed copy
// gets one that queued work still writes, that work has to be done before the
// CPU fills it, or it overwrites the copy.
TEST_F(MetalGraphViewTest, MaterializingIntoRecycledBufferWaitsForItsWork) {
  AOTITensorHandle input = nullptr;
  createMatrix(kDeviceMps, &input);
  auto* input_data = static_cast<float*>(input->mutable_data_ptr());
  for (int i = 0; i < 4; i++) {
    input_data[i] = static_cast<float>(100 + i);
  }
  AOTITensorHandle freed = nullptr;
  createMatrix(kDeviceMps, &freed);
  queueCopy(*input, *freed, 4);
  void* freed_ptr = freed->mutable_data_ptr();
  ASSERT_EQ(aoti_torch_delete_tensor_object(freed), Error::Ok);

  const int64_t base_size = 12;
  AOTITensorHandle base = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &base_size, &kStride, kFloat32, kDeviceCpu, 0, &base),
      Error::Ok);
  auto* data = static_cast<float*>(base->mutable_data_ptr());
  for (int i = 0; i < 12; i++) {
    data[i] = static_cast<float>(i + 1);
  }
  const int64_t strides[2] = {4, 1};
  AOTITensorHandle read = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          base, 2, kMatrixSizes, strides, /*storage_offset=*/4, &read),
      Error::Ok);
  ASSERT_EQ(read->const_data_ptr(), freed_ptr);
  getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);
  const auto* got = static_cast<const float*>(read->const_data_ptr());
  EXPECT_EQ(
      std::vector<float>(got, got + 4), (std::vector<float>{5, 6, 9, 10}));
}

// A wait after a plain commit also waits for the committed work: the pool
// takes a completed wait to mean that nothing queued before it still runs.
TEST_F(MetalGraphViewTest, WaitAfterCommitWaitsForCommittedWork) {
  expectWaitAfterSubmitWaitsForWork(
      [] { getCurrentMetalStream()->synchronize(SyncType::COMMIT); });
}

// Likewise after a flush, with or without commit-and-continue.
TEST_F(MetalGraphViewTest, WaitAfterFlushWaitsForFlushedWork) {
  expectWaitAfterSubmitWaitsForWork([] {
    getCurrentMetalStream()->endKernelCoalescing();
    getCurrentMetalStream()->flush();
  });
}

// A buffer freed before the stream last waited has no work left on it, so
// getting it back from the pool does not make the packed copy wait.
TEST_F(MetalGraphViewTest, MaterializingIntoSettledRecycledBufferDoesNotWait) {
  AOTITensorHandle input = nullptr;
  createMatrix(kDeviceMps, &input);
  AOTITensorHandle out = nullptr;
  createMatrix(kDeviceMps, &out);
  AOTITensorHandle freed = nullptr;
  createMatrix(kDeviceMps, &freed);
  void* freed_ptr = freed->mutable_data_ptr();
  ASSERT_EQ(aoti_torch_delete_tensor_object(freed), Error::Ok);
  getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);
  queueCopy(*input, *out, 4);

  const int64_t base_size = 12;
  AOTITensorHandle base = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &base_size, &kStride, kFloat32, kDeviceCpu, 0, &base),
      Error::Ok);
  auto* data = static_cast<float*>(base->mutable_data_ptr());
  for (int i = 0; i < 12; i++) {
    data[i] = static_cast<float>(i + 1);
  }
  const int64_t strides[2] = {4, 1};
  AOTITensorHandle read = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          base, 2, kMatrixSizes, strides, /*storage_offset=*/4, &read),
      Error::Ok);
  ASSERT_EQ(read->const_data_ptr(), freed_ptr);
  EXPECT_FALSE(getCurrentMetalStream()->isEmpty());
  const auto* got = static_cast<const float*>(read->const_data_ptr());
  EXPECT_EQ(
      std::vector<float>(got, got + 4), (std::vector<float>{5, 6, 9, 10}));
  getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);
}

// A tensor can start inside a Metal buffer without being registered as a view
// of it, e.g. a blob at an offset into it. Materializing a view of it still
// waits for queued writes to that buffer.
TEST_F(MetalGraphViewTest, MaterializingBlobInsideMetalBufferWaits) {
  const int64_t base_size = 12;
  AOTITensorHandle base = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &base_size, &kStride, kFloat32, kDeviceMps, 0, &base),
      Error::Ok);
  auto* memory = static_cast<float*>(base->mutable_data_ptr());
  std::fill_n(memory, 12, 0.0f);
  AOTITensorHandle written = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          base,
          2,
          kMatrixSizes,
          kMatrixStrides,
          /*storage_offset=*/4,
          &written),
      Error::Ok);
  AOTITensorHandle inner = nullptr;
  ASSERT_EQ(createFromBlob(memory + 5, &inner), Error::Ok);
  ASSERT_FALSE(metal_is_device_pointer(inner->mutable_data_ptr()));
  AOTITensorHandle input = nullptr;
  createMatrix(kDeviceMps, &input);
  auto* input_data = static_cast<float*>(input->mutable_data_ptr());
  for (int i = 0; i < 4; i++) {
    input_data[i] = static_cast<float>(i + 1);
  }
  queueCopy(*input, *written, 4);

  // Elements 5 and 7 of `base`: not densely packed, so copied on the CPU.
  const int64_t size = 2;
  const int64_t stride = 2;
  AOTITensorHandle read = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          inner, 1, &size, &stride, /*storage_offset=*/0, &read),
      Error::Ok);
  const auto* got = static_cast<const float*>(read->const_data_ptr());
  EXPECT_EQ(std::vector<float>(got, got + 2), (std::vector<float>{2, 4}));
}

// A tensor can start inside the region of another one without being a view of
// it, e.g. two blobs over the same memory. The GPU still writes it through that
// region's buffer, so materializing a view of it waits.
TEST_F(MetalGraphViewTest, MaterializingMemoryInsideAnotherRegionWaits) {
  std::vector<float> memory(12, 0.0f);
  AOTITensorHandle outer = nullptr;
  ASSERT_EQ(createFromBlob(memory.data(), &outer), Error::Ok);
  AOTITensorHandle written = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          outer,
          2,
          kMatrixSizes,
          kMatrixStrides,
          /*storage_offset=*/4,
          &written),
      Error::Ok);
  AOTITensorHandle inner = nullptr;
  ASSERT_EQ(createFromBlob(memory.data() + 5, &inner), Error::Ok);
  ASSERT_FALSE(metal_is_device_pointer(inner->mutable_data_ptr()));
  AOTITensorHandle input = nullptr;
  createMatrix(kDeviceMps, &input);
  auto* input_data = static_cast<float*>(input->mutable_data_ptr());
  for (int i = 0; i < 4; i++) {
    input_data[i] = static_cast<float>(i + 1);
  }
  queueCopy(*input, *written, 4);

  // Elements 5 and 7 of `memory`: not densely packed, so copied on the CPU.
  const int64_t size = 2;
  const int64_t stride = 2;
  AOTITensorHandle read = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          inner, 1, &size, &stride, /*storage_offset=*/0, &read),
      Error::Ok);
  const auto* got = static_cast<const float*>(read->const_data_ptr());
  EXPECT_EQ(std::vector<float>(got, got + 2), (std::vector<float>{2, 4}));
}

// Views of the same CPU memory are bound into one Metal buffer, so a graph
// reading through one view sees what a graph before it wrote through another,
// overlapping one.
TEST_F(MetalGraphViewTest, OverlappingCpuViewsSeeEachOthersWrites) {
  const int64_t base_size = 12;
  AOTITensorHandle base = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &base_size, &kStride, kFloat32, kDeviceCpu, 0, &base),
      Error::Ok);
  std::fill_n(static_cast<float*>(base->mutable_data_ptr()), 12, 0.0f);
  AOTITensorHandle written = nullptr;
  AOTITensorHandle read = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          base,
          2,
          kMatrixSizes,
          kMatrixStrides,
          /*storage_offset=*/4,
          &written),
      Error::Ok);
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          base, 2, kMatrixSizes, kMatrixStrides, /*storage_offset=*/6, &read),
      Error::Ok);
  AOTITensorHandle input = nullptr;
  createMatrix(kDeviceMps, &input);
  auto* input_data = static_cast<float*>(input->mutable_data_ptr());
  for (int i = 0; i < 4; i++) {
    input_data[i] = static_cast<float>(i + 1);
  }
  AOTITensorHandle identity = nullptr;
  createIdentity(&identity);
  AOTITensorHandle out = nullptr;
  createMatrix(kDeviceMps, &out);

  ASSERT_EQ(aoti_torch_mps_mm_out(written, input, identity), Error::Ok);
  ASSERT_EQ(aoti_torch_mps_mm_out(out, read, identity), Error::Ok);
  getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);
  const auto* got = static_cast<const float*>(out->const_data_ptr());
  EXPECT_EQ(std::vector<float>(got, got + 4), (std::vector<float>{3, 4, 0, 0}));
}

// For CPU memory the runtime did not allocate, a view can reach further than
// the buffer made for the views before it. The buffer is then replaced by a
// longer one, which Metal does not relate to the old one, so work queued
// through the old buffer has to be done before the new one is used.
TEST_F(MetalGraphViewTest, CpuRegionBufferGrowsUnderQueuedWork) {
  std::vector<float> blob(12, 0.0f);
  const int64_t six = 6;
  AOTITensorHandle outer = nullptr;
  ASSERT_EQ(
      aoti_torch_create_tensor_from_blob_v2(
          blob.data(),
          1,
          &six,
          &kStride,
          /*storage_offset=*/0,
          kFloat32,
          kDeviceCpu,
          /*device_index=*/0,
          &outer,
          /*layout=*/0,
          /*opaque_metadata=*/nullptr,
          /*opaque_metadata_size=*/0),
      Error::Ok);
  // Elements 2..5, inside the 6 elements of `outer`.
  AOTITensorHandle first = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          outer, 2, kMatrixSizes, kMatrixStrides, /*storage_offset=*/2, &first),
      Error::Ok);
  AOTITensorHandle input = nullptr;
  createMatrix(kDeviceMps, &input);
  auto* input_data = static_cast<float*>(input->mutable_data_ptr());
  for (int i = 0; i < 4; i++) {
    input_data[i] = static_cast<float>(i + 1);
  }
  queueCopy(*input, *first, 4);

  // Elements 4..7 through a view of `first`: past what `outer` covers, and
  // overlapping what the queued copy writes.
  AOTITensorHandle second = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          first,
          2,
          kMatrixSizes,
          kMatrixStrides,
          /*storage_offset=*/2,
          &second),
      Error::Ok);
  AOTITensorHandle out = nullptr;
  createMatrix(kDeviceMps, &out);
  queueCopy(*second, *out, 4);
  getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);

  const auto* got = static_cast<const float*>(out->const_data_ptr());
  EXPECT_EQ(std::vector<float>(got, got + 4), (std::vector<float>{3, 4, 0, 0}));
}

// An empty view at the end of its buffer points where the next allocation may
// start. It must not be registered there.
TEST_F(MetalMemoryTest, EmptyViewAtEndOfBufferIsNotRegisteredPastIt) {
  AOTITensorHandle base = nullptr;
  const int64_t base_size = 8;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &base_size, &kStride, kFloat32, kDeviceMps, 0, &base),
      Error::Ok);
  void* end = static_cast<float*>(base->mutable_data_ptr()) + 8;
  const int64_t empty = 0;
  AOTITensorHandle view = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          base, 1, &empty, &kStride, /*storage_offset=*/8, &view),
      Error::Ok);
  EXPECT_EQ(view->mutable_data_ptr(), base->mutable_data_ptr());
  EXPECT_EQ(memory_to_n_tensor.count(end), 0u);
  EXPECT_FALSE(metal_is_device_pointer(end));
}

// Nothing stays recorded at a view's address once its last handle is gone, so
// a later allocation there is not mistaken for one.
TEST_F(MetalMemoryTest, ViewAddressIsForgottenWithItsLastHandle) {
  AOTITensorHandle base = nullptr;
  AOTITensorHandle view = nullptr;
  createBaseAndView(&base, &view);
  void* view_ptr = view->mutable_data_ptr();
  AOTITensorHandle alias = nullptr;
  ASSERT_EQ(aoti_torch_new_tensor_handle(view, &alias), Error::Ok);

  ASSERT_EQ(aoti_torch_delete_tensor_object(view), Error::Ok);
  EXPECT_EQ(memory_to_n_tensor.count(view_ptr), 1u);
  ASSERT_EQ(aoti_torch_delete_tensor_object(alias), Error::Ok);
  EXPECT_EQ(memory_to_n_tensor.count(view_ptr), 0u);
}

// Views of CPU memory are deleted and made again all the time. The buffer they
// are bound into stays, so work queued through a deleted view is ordered with
// work through the next one.
TEST_F(MetalGraphViewTest, CpuRegionOutlivesItsViews) {
  const int64_t base_size = 12;
  AOTITensorHandle base = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &base_size, &kStride, kFloat32, kDeviceCpu, 0, &base),
      Error::Ok);
  std::fill_n(static_cast<float*>(base->mutable_data_ptr()), 12, 0.0f);
  AOTITensorHandle input = nullptr;
  createMatrix(kDeviceMps, &input);
  auto* input_data = static_cast<float*>(input->mutable_data_ptr());
  for (int i = 0; i < 4; i++) {
    input_data[i] = static_cast<float>(i + 1);
  }

  AOTITensorHandle first = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          base, 2, kMatrixSizes, kMatrixStrides, /*storage_offset=*/4, &first),
      Error::Ok);
  queueCopy(*input, *first, 4);
  ASSERT_EQ(aoti_torch_delete_tensor_object(first), Error::Ok);

  AOTITensorHandle second = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          base, 2, kMatrixSizes, kMatrixStrides, /*storage_offset=*/4, &second),
      Error::Ok);
  AOTITensorHandle out = nullptr;
  createMatrix(kDeviceMps, &out);
  queueCopy(*second, *out, 4);
  getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);

  const auto* got = static_cast<const float*>(out->const_data_ptr());
  EXPECT_EQ(std::vector<float>(got, got + 4), (std::vector<float>{1, 2, 3, 4}));
}

// The CPU allocation itself is bound into the buffer of its region too, so a
// kernel reading it sees what a kernel before it wrote through a view.
TEST_F(MetalGraphViewTest, CpuAllocationIsBoundIntoItsRegion) {
  const int64_t base_size = 8;
  AOTITensorHandle base = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &base_size, &kStride, kFloat32, kDeviceCpu, 0, &base),
      Error::Ok);
  std::fill_n(static_cast<float*>(base->mutable_data_ptr()), 8, 0.0f);
  AOTITensorHandle view = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          base, 2, kMatrixSizes, kMatrixStrides, /*storage_offset=*/4, &view),
      Error::Ok);
  AOTITensorHandle input = nullptr;
  createMatrix(kDeviceMps, &input);
  auto* input_data = static_cast<float*>(input->mutable_data_ptr());
  for (int i = 0; i < 4; i++) {
    input_data[i] = static_cast<float>(i + 1);
  }
  AOTITensorHandle out = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &base_size, &kStride, kFloat32, kDeviceMps, 0, &out),
      Error::Ok);

  queueCopy(*input, *view, 4);
  queueCopy(*base, *out, 8);
  getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);

  const auto* got = static_cast<const float*>(out->const_data_ptr());
  EXPECT_EQ(
      std::vector<float>(got, got + 8),
      (std::vector<float>{0, 0, 0, 0, 1, 2, 3, 4}));
}

// Freeing CPU memory the GPU never reached does not wait for queued work.
TEST_F(MetalGraphViewTest, FreeingUnusedCpuMemoryDoesNotWait) {
  AOTITensorHandle input = nullptr;
  createMatrix(kDeviceMps, &input);
  AOTITensorHandle identity = nullptr;
  createIdentity(&identity);
  AOTITensorHandle out = nullptr;
  createMatrix(kDeviceMps, &out);
  AOTITensorHandle scratch = nullptr;
  createMatrix(kDeviceCpu, &scratch);
  getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);

  ASSERT_EQ(aoti_torch_mps_mm_out(out, input, identity), Error::Ok);
  ASSERT_EQ(aoti_torch_delete_tensor_object(scratch), Error::Ok);
  EXPECT_FALSE(getCurrentMetalStream()->isEmpty());
  getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);
}

// Once CPU memory has a Metal buffer, a copy into it is still done by the time
// aoti_torch_copy_ returns: CPU code reads that memory directly.
TEST_F(MetalGraphViewTest, CopyIntoCpuMemoryWithABufferIsSynchronous) {
  const int64_t size = 8;
  AOTITensorHandle host = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &size, &kStride, kFloat32, kDeviceCpu, 0, &host),
      Error::Ok);
  std::fill_n(static_cast<float*>(host->mutable_data_ptr()), 8, 0.0f);
  AOTITensorHandle view = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          host, 2, kMatrixSizes, kMatrixStrides, /*storage_offset=*/4, &view),
      Error::Ok);
  ASSERT_EQ(aoti_torch_delete_tensor_object(view), Error::Ok);
  AOTITensorHandle device = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &size, &kStride, kFloat32, kDeviceMps, 0, &device),
      Error::Ok);
  auto* device_data = static_cast<float*>(device->mutable_data_ptr());
  for (int i = 0; i < 8; i++) {
    device_data[i] = static_cast<float>(i + 1);
  }

  ASSERT_EQ(aoti_torch_copy_(host, device, 0), Error::Ok);
  const auto* got = static_cast<const float*>(host->const_data_ptr());
  EXPECT_EQ(
      std::vector<float>(got, got + 8),
      (std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}));
}

// CPU memory the runtime did not allocate can be freed and its address reused
// by its owner, so the Metal buffer over it goes with its last view.
TEST_F(MetalGraphViewTest, UnownedCpuRegionGoesWithItsLastView) {
  std::vector<float> blob(8, 1.0f);
  const int64_t eight = 8;
  AOTITensorHandle outer = nullptr;
  ASSERT_EQ(
      aoti_torch_create_tensor_from_blob_v2(
          blob.data(),
          1,
          &eight,
          &kStride,
          /*storage_offset=*/0,
          kFloat32,
          kDeviceCpu,
          /*device_index=*/0,
          &outer,
          /*layout=*/0,
          /*opaque_metadata=*/nullptr,
          /*opaque_metadata_size=*/0),
      Error::Ok);
  AOTITensorHandle view = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          outer, 2, kMatrixSizes, kMatrixStrides, /*storage_offset=*/4, &view),
      Error::Ok);
  EXPECT_TRUE(metal_is_device_pointer(blob.data()));

  ASSERT_EQ(aoti_torch_delete_tensor_object(view), Error::Ok);
  EXPECT_FALSE(metal_is_device_pointer(blob.data()));
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
