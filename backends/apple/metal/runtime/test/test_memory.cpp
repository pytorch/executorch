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
#include <new>
#include <vector>

#include <executorch/backends/apple/metal/runtime/shims/et_metal.h>
#include <executorch/backends/apple/metal/runtime/shims/memory.h>
#include <executorch/backends/apple/metal/runtime/shims/shim_mps.h>
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

  // Wraps `data` in a float tensor with the given sizes and strides, the way
  // AOTInductor wraps memory it does not own.
  Error createStridedBlob(
      void* data,
      const std::vector<int64_t>& sizes,
      const std::vector<int64_t>& strides,
      AOTITensorHandle* tensor) {
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
    EXPECT_TRUE(metal_is_view(view_ptr));

    ASSERT_EQ(
        aoti_torch_delete_tensor_object(delete_view ? alias : view), Error::Ok);
    EXPECT_FALSE(metal_is_view(view_ptr));
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
  EXPECT_TRUE(metal_is_view(view_ptr));
  ASSERT_EQ(aoti_torch_delete_tensor_object(alias), Error::Ok);
  EXPECT_FALSE(metal_is_view(view_ptr));
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

// Waits on one stream do not cover work queued on another, so a buffer freed
// on one stream is not handed to another one.
TEST_F(MetalGraphViewTest, PoolReusesBuffersOnlyOnTheirOwnStream) {
  ETMetalStream* original = getCurrentMetalStream();
  ETMetalStream other;
  struct Restore {
    ETMetalStream* stream;
    ~Restore() {
      setCurrentMetalStream(stream);
    }
  } restore{original};

  setCurrentMetalStream(&other);
  AOTITensorHandle freed = nullptr;
  createMatrix(kDeviceMps, &freed);
  void* freed_ptr = freed->mutable_data_ptr();
  ASSERT_EQ(aoti_torch_delete_tensor_object(freed), Error::Ok);

  setCurrentMetalStream(original);
  AOTITensorHandle elsewhere = nullptr;
  createMatrix(kDeviceMps, &elsewhere);
  EXPECT_NE(elsewhere->mutable_data_ptr(), freed_ptr);

  setCurrentMetalStream(&other);
  AOTITensorHandle again = nullptr;
  createMatrix(kDeviceMps, &again);
  EXPECT_EQ(again->mutable_data_ptr(), freed_ptr);
}

// A stream that goes away takes the buffers freed on it out of the pool, so a
// later stream at the same address does not get them.
TEST_F(MetalGraphViewTest, PoolForgetsBuffersOfADestroyedStream) {
  ETMetalStream* original = getCurrentMetalStream();
  struct Restore {
    ETMetalStream* stream;
    ~Restore() {
      setCurrentMetalStream(stream);
    }
  } restore{original};
  alignas(ETMetalStream) unsigned char storage[sizeof(ETMetalStream)];

  auto* first = new (storage) ETMetalStream();
  setCurrentMetalStream(first);
  AOTITensorHandle freed = nullptr;
  createMatrix(kDeviceMps, &freed);
  ASSERT_EQ(aoti_torch_delete_tensor_object(freed), Error::Ok);
  setCurrentMetalStream(original);
  first->~ETMetalStream();

  // A new stream at the same address, with no waits yet like the old one
  // when it freed the buffer: an entry left behind would be handed to it as
  // possibly still in use.
  auto* second = new (storage) ETMetalStream();
  ASSERT_EQ(static_cast<void*>(second), static_cast<void*>(first));
  setCurrentMetalStream(second);
  bool may_be_in_use = true;
  void* fresh =
      metal_allocate_buffer_tracking_use(4 * sizeof(float), &may_be_in_use);
  ASSERT_NE(fresh, nullptr);
  EXPECT_FALSE(may_be_in_use);
  metal_deallocate_buffer(fresh);
  setCurrentMetalStream(original);
  second->~ETMetalStream();
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

// A region over memory the runtime did not allocate grows with its views. It
// is refused where it would take in a tensor that lies there without being
// registered in it, e.g. a blob made before the region: that tensor would be
// bound by copy, unordered with work through the region.
TEST_F(MetalGraphViewTest, RegionOverAnotherTensorIsRefused) {
  std::vector<float> memory(12, 0.0f);
  AOTITensorHandle inner = nullptr;
  ASSERT_EQ(createFromBlob(memory.data() + 5, &inner), Error::Ok);
  AOTITensorHandle outer = nullptr;
  ASSERT_EQ(createFromBlob(memory.data(), &outer), Error::Ok);
  AOTITensorHandle view = nullptr;
  EXPECT_NE(
      aoti_torch__reinterpret_tensor(
          outer, 2, kMatrixSizes, kMatrixStrides, /*storage_offset=*/4, &view),
      Error::Ok);
  EXPECT_EQ(view, nullptr);
}

// A tensor made from a blob at an offset into a Metal buffer is bound into
// that buffer, as a view of it would be: a kernel writing it writes the buffer,
// and a view of it is registered in the same buffer, not given one of its own.
TEST_F(MetalGraphViewTest, BlobInsideMetalBufferIsBoundIntoIt) {
  const int64_t base_size = 12;
  AOTITensorHandle base = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &base_size, &kStride, kFloat32, kDeviceMps, 0, &base),
      Error::Ok);
  auto* memory = static_cast<float*>(base->mutable_data_ptr());
  std::fill_n(memory, 12, 0.0f);
  AOTITensorHandle blob = nullptr;
  ASSERT_EQ(createFromBlob(memory + 4, &blob), Error::Ok);
  void* found = nullptr;
  size_t found_nbytes = 0;
  bool cpu = true;
  ASSERT_TRUE(metal_find_memory(memory + 4, &found, &cpu, &found_nbytes));
  EXPECT_EQ(found, memory);
  EXPECT_FALSE(cpu);

  const int64_t size = 2;
  AOTITensorHandle view = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          blob, 1, &size, &kStride, /*storage_offset=*/2, &view),
      Error::Ok);
  EXPECT_EQ(view->mutable_data_ptr(), memory + 6);
  found = nullptr;
  cpu = true;
  ASSERT_TRUE(
      metal_find_memory(view->mutable_data_ptr(), &found, &cpu, &found_nbytes));
  EXPECT_EQ(found, memory);
  EXPECT_FALSE(cpu);

  AOTITensorHandle input = nullptr;
  createMatrix(kDeviceMps, &input);
  auto* input_data = static_cast<float*>(input->mutable_data_ptr());
  for (int i = 0; i < 4; i++) {
    input_data[i] = static_cast<float>(i + 1);
  }
  queueCopy(*input, *blob, 4);
  getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);
  EXPECT_EQ(
      std::vector<float>(memory + 4, memory + 8),
      (std::vector<float>{1, 2, 3, 4}));
}

// Likewise for a blob inside a CPU region: it is bound into the region's
// buffer, and a view of it joins that region instead of making a second buffer
// over the same memory.
TEST_F(MetalGraphViewTest, BlobInsideCpuRegionIsBoundIntoIt) {
  const int64_t base_size = 12;
  AOTITensorHandle base = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &base_size, &kStride, kFloat32, kDeviceCpu, 0, &base),
      Error::Ok);
  auto* memory = static_cast<float*>(base->mutable_data_ptr());
  std::fill_n(memory, 12, 0.0f);
  AOTITensorHandle first = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          base,
          2,
          kMatrixSizes,
          kMatrixStrides,
          /*storage_offset=*/4,
          &first),
      Error::Ok);

  AOTITensorHandle blob = nullptr;
  ASSERT_EQ(createFromBlob(memory + 8, &blob), Error::Ok);
  EXPECT_TRUE(metal_is_cpu_memory(memory + 8));
  void* found = nullptr;
  size_t found_nbytes = 0;
  bool cpu = false;
  ASSERT_TRUE(metal_find_memory(memory + 8, &found, &cpu, &found_nbytes));
  EXPECT_EQ(found, memory);
  EXPECT_TRUE(cpu);

  const int64_t size = 2;
  AOTITensorHandle view = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          blob, 1, &size, &kStride, /*storage_offset=*/1, &view),
      Error::Ok);
  EXPECT_EQ(view->mutable_data_ptr(), memory + 9);
  found = nullptr;
  cpu = false;
  ASSERT_TRUE(
      metal_find_memory(view->mutable_data_ptr(), &found, &cpu, &found_nbytes));
  EXPECT_EQ(found, memory);
  EXPECT_TRUE(cpu);

  AOTITensorHandle input = nullptr;
  createMatrix(kDeviceMps, &input);
  auto* input_data = static_cast<float*>(input->mutable_data_ptr());
  for (int i = 0; i < 4; i++) {
    input_data[i] = static_cast<float>(i + 1);
  }
  queueCopy(*input, *blob, 4);
  getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);
  EXPECT_EQ(
      std::vector<float>(memory + 8, memory + 12),
      (std::vector<float>{1, 2, 3, 4}));
}

// A blob inside a CPU allocation of the runtime's joins the region over the
// whole allocation, even before anything else made that region: views of the
// blob and of the allocation then share one buffer.
TEST_F(MetalGraphViewTest, BlobInsideCpuAllocationJoinsItsRegion) {
  const int64_t base_size = 12;
  AOTITensorHandle base = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &base_size, &kStride, kFloat32, kDeviceCpu, 0, &base),
      Error::Ok);
  auto* memory = static_cast<float*>(base->mutable_data_ptr());
  AOTITensorHandle blob = nullptr;
  ASSERT_EQ(createFromBlob(memory + 4, &blob), Error::Ok);
  const int64_t size = 2;
  AOTITensorHandle blob_view = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          blob, 1, &size, &kStride, /*storage_offset=*/1, &blob_view),
      Error::Ok);
  AOTITensorHandle base_view = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          base,
          2,
          kMatrixSizes,
          kMatrixStrides,
          /*storage_offset=*/4,
          &base_view),
      Error::Ok);
  for (void* ptr :
       {blob->mutable_data_ptr(),
        blob_view->mutable_data_ptr(),
        base_view->mutable_data_ptr()}) {
    void* found = nullptr;
    size_t found_nbytes = 0;
    bool cpu = false;
    ASSERT_TRUE(metal_find_memory(ptr, &found, &cpu, &found_nbytes));
    EXPECT_EQ(found, memory);
    EXPECT_TRUE(cpu);
    EXPECT_EQ(found_nbytes, 12 * sizeof(float));
  }
}

// Two CPU regions over the same bytes would be two buffers Metal does not
// order, so a view that would make one is an error.
TEST_F(MetalGraphViewTest, OverlappingCpuRegionsAreRefused) {
  std::vector<float> memory(12, 0.0f);
  AOTITensorHandle inner = nullptr;
  ASSERT_EQ(createFromBlob(memory.data() + 4, &inner), Error::Ok);
  const int64_t size = 2;
  AOTITensorHandle inner_view = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          inner, 1, &size, &kStride, /*storage_offset=*/1, &inner_view),
      Error::Ok);
  // A blob over the region's start that is not inside it is refused
  // already.
  AOTITensorHandle outer = nullptr;
  EXPECT_NE(createFromBlob(memory.data() + 2, &outer), Error::Ok);
  EXPECT_EQ(outer, nullptr);
}

// A blob and a view can sit at one address. Each handle gives back only the
// registration it took, so deleting the blob leaves the view registered.
TEST_F(MetalGraphViewTest, BlobAndViewAtOneAddressKeepTheirCounts) {
  const int64_t base_size = 12;
  AOTITensorHandle base = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &base_size, &kStride, kFloat32, kDeviceMps, 0, &base),
      Error::Ok);
  auto* memory = static_cast<float*>(base->mutable_data_ptr());
  AOTITensorHandle blob = nullptr;
  ASSERT_EQ(createFromBlob(memory + 4, &blob), Error::Ok);
  const int64_t size = 4;
  AOTITensorHandle view = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          base, 1, &size, &kStride, /*storage_offset=*/4, &view),
      Error::Ok);
  ASSERT_EQ(view->mutable_data_ptr(), memory + 4);

  ASSERT_EQ(aoti_torch_delete_tensor_object(blob), Error::Ok);
  EXPECT_TRUE(metal_is_view(memory + 4));
  ASSERT_EQ(aoti_torch_delete_tensor_object(view), Error::Ok);
  EXPECT_FALSE(metal_is_view(memory + 4));
}

// The same in a CPU region: a blob made inside it and a view at the blob's
// address each give back only their own registration.
TEST_F(MetalGraphViewTest, BlobAndViewInARegionKeepTheirCounts) {
  std::vector<float> memory(12, 0.0f);
  AOTITensorHandle outer = nullptr;
  ASSERT_EQ(createFromBlob(memory.data(), &outer), Error::Ok);
  const int64_t size = 2;
  AOTITensorHandle first = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          outer, 1, &size, &kStride, /*storage_offset=*/6, &first),
      Error::Ok);
  AOTITensorHandle blob = nullptr;
  ASSERT_EQ(createFromBlob(memory.data() + 2, &blob), Error::Ok);
  ASSERT_TRUE(metal_is_view(memory.data() + 2));
  AOTITensorHandle view = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          outer, 1, &size, &kStride, /*storage_offset=*/2, &view),
      Error::Ok);

  ASSERT_EQ(aoti_torch_delete_tensor_object(blob), Error::Ok);
  EXPECT_TRUE(metal_is_view(memory.data() + 2));
  ASSERT_EQ(aoti_torch_delete_tensor_object(view), Error::Ok);
  EXPECT_FALSE(metal_is_view(memory.data() + 2));
}

// A blob that starts inside a Metal buffer but runs past its end cannot be
// bound into it, and is an error.
TEST_F(MetalGraphViewTest, BlobPastTheEndOfItsBufferIsRefused) {
  const int64_t base_size = 4;
  AOTITensorHandle base = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &base_size, &kStride, kFloat32, kDeviceMps, 0, &base),
      Error::Ok);
  auto* memory = static_cast<float*>(base->mutable_data_ptr());
  AOTITensorHandle blob = nullptr;
  EXPECT_NE(createFromBlob(memory + 2, &blob), Error::Ok);
  EXPECT_EQ(blob, nullptr);
  EXPECT_FALSE(metal_is_view(memory + 2));
}

// A blob inside a constant is bound into that constant's own buffer, the one
// the constant's tensor is bound to, not into the buffer of all constants that
// holds it.
TEST_F(MetalGraphViewTest, BlobInsideAConstantTakesThatConstantsBuffer) {
  void* constants = nullptr;
  ASSERT_EQ(aoti_torch_mps_malloc(&constants, 16 * sizeof(float)), Error::Ok);
  std::vector<float> data(8, 1.0f);
  ASSERT_EQ(
      aoti_torch_mps_memcpy(
          constants,
          /*constant_offset=*/4 * sizeof(float),
          /*bytes_read=*/0,
          /*data_size=*/8 * sizeof(float),
          reinterpret_cast<uint8_t*>(data.data())),
      Error::Ok);
  auto* memory = static_cast<float*>(constants);
  AOTITensorHandle blob = nullptr;
  ASSERT_EQ(createFromBlob(memory + 6, &blob), Error::Ok);
  void* found = nullptr;
  size_t found_nbytes = 0;
  bool cpu = true;
  ASSERT_TRUE(metal_find_memory(memory + 6, &found, &cpu, &found_nbytes));
  EXPECT_EQ(found, memory + 4);
  EXPECT_FALSE(cpu);
}

// AOTInductor loads constants one at a time: it copies one into the constants
// buffer, wraps it in a tensor, and then copies the next. The tensor over the
// first constant does not make the buffer a tensor's memory.
TEST_F(MetalGraphViewTest, ConstantsLoadOneAfterAnother) {
  void* constants = nullptr;
  ASSERT_EQ(aoti_torch_mps_malloc(&constants, 8 * sizeof(float)), Error::Ok);
  std::vector<float> first = {1, 2, 3, 4};
  std::vector<float> second = {5, 6, 7, 8};
  ASSERT_EQ(
      aoti_torch_mps_memcpy(
          constants,
          /*constant_offset=*/0,
          /*bytes_read=*/0,
          /*data_size=*/4 * sizeof(float),
          reinterpret_cast<uint8_t*>(first.data())),
      Error::Ok);
  auto* memory = static_cast<float*>(constants);
  AOTITensorHandle first_tensor = nullptr;
  ASSERT_EQ(createFromBlob(memory, &first_tensor), Error::Ok);
  ASSERT_EQ(
      aoti_torch_mps_memcpy(
          constants,
          /*constant_offset=*/4 * sizeof(float),
          /*bytes_read=*/0,
          /*data_size=*/4 * sizeof(float),
          reinterpret_cast<uint8_t*>(second.data())),
      Error::Ok);
  AOTITensorHandle second_tensor = nullptr;
  ASSERT_EQ(createFromBlob(memory + 4, &second_tensor), Error::Ok);
  const auto* got = static_cast<const float*>(second_tensor->const_data_ptr());
  EXPECT_EQ(std::vector<float>(got, got + 4), second);
}

// Only constants lie in a buffer inside another one. Mapping any other buffer
// over memory already bound would make two buffers Metal does not order.
TEST_F(MetalGraphViewTest, NoCopyBufferOverBoundMemoryIsRefused) {
  const int64_t base_size = 16;
  AOTITensorHandle base = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &base_size, &kStride, kFloat32, kDeviceMps, 0, &base),
      Error::Ok);
  auto* memory = static_cast<float*>(base->mutable_data_ptr());
  EXPECT_FALSE(metal_buffer_nocopy(memory + 4, 8 * sizeof(float), true));
  EXPECT_FALSE(metal_is_device_pointer(memory + 4));
}

// A blob inside a Metal allocation keeps it alive, as a view of it does.
TEST_F(MetalGraphViewTest, BlobKeepsTheAllocationItLiesInAlive) {
  const int64_t base_size = 12;
  AOTITensorHandle base = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &base_size, &kStride, kFloat32, kDeviceMps, 0, &base),
      Error::Ok);
  auto* memory = static_cast<float*>(base->mutable_data_ptr());
  AOTITensorHandle blob = nullptr;
  ASSERT_EQ(createFromBlob(memory + 4, &blob), Error::Ok);

  ASSERT_EQ(aoti_torch_delete_tensor_object(base), Error::Ok);
  EXPECT_TRUE(metal_is_device_pointer(memory));
  EXPECT_TRUE(metal_is_device_pointer(memory + 4));
  ASSERT_EQ(aoti_torch_delete_tensor_object(blob), Error::Ok);
  EXPECT_FALSE(metal_is_device_pointer(memory));
}

// The packed copy waits for GPU writes anywhere in the bytes it reads, also
// when the view starts in memory the GPU does not reach and runs into a region
// that it does.
TEST_F(MetalGraphViewTest, MaterializingSourceRunningIntoARegionWaits) {
  std::vector<float> memory(12, 0.0f);
  AOTITensorHandle region_blob = nullptr;
  ASSERT_EQ(createFromBlob(memory.data() + 5, &region_blob), Error::Ok);
  const int64_t two = 2;
  AOTITensorHandle written = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          region_blob, 1, &two, &kStride, /*storage_offset=*/1, &written),
      Error::Ok);
  AOTITensorHandle outside = nullptr;
  ASSERT_EQ(createFromBlob(memory.data(), &outside), Error::Ok);
  ASSERT_FALSE(metal_is_device_pointer(memory.data()));
  AOTITensorHandle input = nullptr;
  createMatrix(kDeviceMps, &input);
  auto* input_data = static_cast<float*>(input->mutable_data_ptr());
  for (int i = 0; i < 4; i++) {
    input_data[i] = static_cast<float>(i + 1);
  }
  queueCopy(*input, *written, 2);

  // Elements 0, 3 and 6 of `memory`: not densely packed, so copied on the CPU.
  const int64_t three = 3;
  AOTITensorHandle read = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          outside, 1, &three, &three, /*storage_offset=*/0, &read),
      Error::Ok);
  const auto* got = static_cast<const float*>(read->const_data_ptr());
  EXPECT_EQ(std::vector<float>(got, got + 3), (std::vector<float>{0, 0, 1}));
}

// Deleting a view first leaves a blob at the same address tracked: the address
// counts every handle at it.
TEST_F(MetalGraphViewTest, ViewDeletedBeforeABlobAtItsAddressKeepsItTracked) {
  std::vector<float> memory(12, 0.0f);
  AOTITensorHandle outer = nullptr;
  ASSERT_EQ(createFromBlob(memory.data(), &outer), Error::Ok);
  const int64_t size = 2;
  AOTITensorHandle first = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          outer, 1, &size, &kStride, /*storage_offset=*/6, &first),
      Error::Ok);
  AOTITensorHandle blob = nullptr;
  ASSERT_EQ(createFromBlob(memory.data() + 2, &blob), Error::Ok);
  AOTITensorHandle view = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          outer, 1, &size, &kStride, /*storage_offset=*/2, &view),
      Error::Ok);

  ASSERT_EQ(aoti_torch_delete_tensor_object(view), Error::Ok);
  AOTITensorHandle again = nullptr;
  EXPECT_NE(createFromBlob(memory.data() + 2, &again), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(blob), Error::Ok);
  EXPECT_EQ(createFromBlob(memory.data() + 2, &again), Error::Ok);
}

// A tensor's strides are those of a dense tensor in some dim order: other
// strides, such as a view with holes or a stride of 0, are an error rather
// than an abort in make_tensor_ptr.
TEST_F(MetalGraphViewTest, StridesOfNoDenseTensorAreRefused) {
  std::vector<float> memory(12, 0.0f);
  AOTITensorHandle holes = nullptr;
  EXPECT_NE(createStridedBlob(memory.data(), {2}, {3}, &holes), Error::Ok);
  EXPECT_EQ(holes, nullptr);
  AOTITensorHandle repeated = nullptr;
  EXPECT_NE(createStridedBlob(memory.data(), {4}, {0}, &repeated), Error::Ok);
  EXPECT_EQ(repeated, nullptr);
  AOTITensorHandle column_major = nullptr;
  EXPECT_EQ(
      createStridedBlob(memory.data(), {2, 3}, {1, 2}, &column_major),
      Error::Ok);

  const int64_t size = 2;
  const int64_t stride = 3;
  AOTITensorHandle allocated = nullptr;
  EXPECT_NE(
      aoti_torch_empty_strided(
          1, &size, &stride, kFloat32, kDeviceMps, 0, &allocated),
      Error::Ok);
  EXPECT_EQ(allocated, nullptr);
}

// A tensor with no elements has no layout: any strides do.
TEST_F(MetalGraphViewTest, EmptyTensorsTakeAnyStrides) {
  std::vector<float> memory(12, 0.0f);
  AOTITensorHandle blob = nullptr;
  EXPECT_EQ(
      createStridedBlob(memory.data(), {4, 0}, {10, 1}, &blob), Error::Ok);
  const int64_t sizes[4] = {2, 0, 3, 4};
  const int64_t strides[4] = {12, 1, 4, 1};
  AOTITensorHandle allocated = nullptr;
  EXPECT_EQ(
      aoti_torch_empty_strided(
          4, sizes, strides, kFloat32, kDeviceMps, 0, &allocated),
      Error::Ok);
  const int64_t base_size = 12;
  AOTITensorHandle base = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &base_size, &kStride, kFloat32, kDeviceMps, 0, &base),
      Error::Ok);
  const int64_t view_sizes[2] = {4, 0};
  const int64_t view_strides[2] = {10, 1};
  AOTITensorHandle view = nullptr;
  EXPECT_EQ(
      aoti_torch__reinterpret_tensor(
          base, 2, view_sizes, view_strides, /*storage_offset=*/5, &view),
      Error::Ok);
}

// A view whose dims overlap, which no strides of a tensor can describe, is
// materialized like one with holes.
TEST_F(MetalGraphViewTest, ViewWithOverlappingDimsIsMaterialized) {
  const int64_t base_size = 12;
  AOTITensorHandle base = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &base_size, &kStride, kFloat32, kDeviceCpu, 0, &base),
      Error::Ok);
  auto* memory = static_cast<float*>(base->mutable_data_ptr());
  for (int i = 0; i < 12; i++) {
    memory[i] = static_cast<float>(i);
  }
  const int64_t sizes[3] = {2, 2, 3};
  const int64_t strides[3] = {5, 4, 1};
  AOTITensorHandle view = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          base, 3, sizes, strides, /*storage_offset=*/0, &view),
      Error::Ok);
  const auto* got = static_cast<const float*>(view->const_data_ptr());
  EXPECT_EQ(got[3], 4.0f);
  EXPECT_EQ(got[6], 5.0f);
  EXPECT_EQ(got[11], 11.0f);
}

// A view lies inside the memory of the tensor it is made from.
TEST_F(MetalGraphViewTest, ViewOutsideItsMemoryIsRefused) {
  const int64_t base_size = 8;
  AOTITensorHandle base = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &base_size, &kStride, kFloat32, kDeviceMps, 0, &base),
      Error::Ok);
  const int64_t four = 4;
  const int64_t two = 2;
  AOTITensorHandle past = nullptr;
  EXPECT_NE(
      aoti_torch__reinterpret_tensor(
          base, 1, &four, &kStride, /*storage_offset=*/6, &past),
      Error::Ok);
  EXPECT_EQ(past, nullptr);
  AOTITensorHandle before = nullptr;
  EXPECT_NE(
      aoti_torch__reinterpret_tensor(
          base, 1, &two, &kStride, /*storage_offset=*/-1, &before),
      Error::Ok);
  EXPECT_EQ(before, nullptr);

  const int64_t cpu_size = 12;
  AOTITensorHandle cpu = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &cpu_size, &kStride, kFloat32, kDeviceCpu, 0, &cpu),
      Error::Ok);
  AOTITensorHandle cpu_past = nullptr;
  EXPECT_NE(
      aoti_torch__reinterpret_tensor(
          cpu, 1, &four, &kStride, /*storage_offset=*/10, &cpu_past),
      Error::Ok);
}

// A view of a view can start where the allocation starts. It is then another
// handle to the allocation, and the counts still balance.
TEST_F(MetalGraphViewTest, ViewBackAtTheAllocationStartIsTheAllocation) {
  const int64_t base_size = 8;
  AOTITensorHandle base = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &base_size, &kStride, kFloat32, kDeviceMps, 0, &base),
      Error::Ok);
  void* memory = base->mutable_data_ptr();
  const int64_t four = 4;
  AOTITensorHandle half = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          base, 1, &four, &kStride, /*storage_offset=*/4, &half),
      Error::Ok);
  AOTITensorHandle back = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          half, 1, &four, &kStride, /*storage_offset=*/-4, &back),
      Error::Ok);
  ASSERT_EQ(back->mutable_data_ptr(), memory);
  EXPECT_FALSE(metal_is_view(memory));

  ASSERT_EQ(aoti_torch_delete_tensor_object(base), Error::Ok);
  ASSERT_EQ(aoti_torch_delete_tensor_object(half), Error::Ok);
  EXPECT_TRUE(metal_is_device_pointer(memory));
  ASSERT_EQ(aoti_torch_delete_tensor_object(back), Error::Ok);
  EXPECT_FALSE(metal_is_device_pointer(memory));
}

// A negative size is refused with an error, before anything is allocated.
TEST_F(MetalGraphViewTest, NegativeSizesAreRefused) {
  const int64_t size = -4;
  AOTITensorHandle allocated = nullptr;
  EXPECT_NE(
      aoti_torch_empty_strided(
          1, &size, &kStride, kFloat32, kDeviceMps, 0, &allocated),
      Error::Ok);
  EXPECT_EQ(allocated, nullptr);
  std::vector<float> memory(4, 0.0f);
  AOTITensorHandle blob = nullptr;
  EXPECT_NE(
      createStridedBlob(memory.data(), {-1, 0}, {1, 1}, &blob), Error::Ok);
  EXPECT_EQ(blob, nullptr);

  const int64_t base_size = 32;
  AOTITensorHandle base = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &base_size, &kStride, kFloat32, kDeviceMps, 0, &base),
      Error::Ok);
  const int64_t sizes[2] = {-2, -3};
  const int64_t strides[2] = {3, 1};
  AOTITensorHandle view = nullptr;
  EXPECT_NE(
      aoti_torch__reinterpret_tensor(
          base, 2, sizes, strides, /*storage_offset=*/13, &view),
      Error::Ok);
  EXPECT_EQ(view, nullptr);
}

// A region's buffer starts at the region, so a view that starts below it is
// refused rather than bound at an offset before the buffer.
TEST_F(MetalGraphViewTest, ViewBelowItsCpuRegionIsRefused) {
  std::vector<float> memory(16, 0.0f);
  AOTITensorHandle blob = nullptr;
  ASSERT_EQ(createFromBlob(memory.data() + 4, &blob), Error::Ok);
  const int64_t four = 4;
  AOTITensorHandle view = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          blob, 1, &four, &kStride, /*storage_offset=*/4, &view),
      Error::Ok);
  AOTITensorHandle below = nullptr;
  EXPECT_NE(
      aoti_torch__reinterpret_tensor(
          view, 1, &four, &kStride, /*storage_offset=*/-5, &below),
      Error::Ok);
  EXPECT_EQ(below, nullptr);
}

// Constants get buffers of their own only inside a constants buffer, never
// inside a tensor's memory.
TEST_F(MetalGraphViewTest, ConstantsAreNotCopiedIntoATensor) {
  const int64_t base_size = 16;
  AOTITensorHandle base = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &base_size, &kStride, kFloat32, kDeviceMps, 0, &base),
      Error::Ok);
  std::vector<float> data(4, 1.0f);
  EXPECT_NE(
      aoti_torch_mps_memcpy(
          base->mutable_data_ptr(),
          /*constant_offset=*/4 * sizeof(float),
          /*bytes_read=*/0,
          /*data_size=*/4 * sizeof(float),
          reinterpret_cast<uint8_t*>(data.data())),
      Error::Ok);
  EXPECT_FALSE(metal_is_device_pointer(
      static_cast<float*>(base->mutable_data_ptr()) + 4));
}

// Freeing a constants buffer takes its constants' buffers with it.
TEST_F(MetalGraphViewTest, FreeingConstantsTakesTheirBuffers) {
  void* constants = nullptr;
  ASSERT_EQ(aoti_torch_mps_malloc(&constants, 16 * sizeof(float)), Error::Ok);
  std::vector<float> data(8, 1.0f);
  ASSERT_EQ(
      aoti_torch_mps_memcpy(
          constants,
          /*constant_offset=*/4 * sizeof(float),
          /*bytes_read=*/0,
          /*data_size=*/8 * sizeof(float),
          reinterpret_cast<uint8_t*>(data.data())),
      Error::Ok);
  auto* memory = static_cast<float*>(constants);
  ASSERT_TRUE(metal_is_device_pointer(memory + 4));
  ASSERT_EQ(aoti_torch_mps_free(constants), Error::Ok);
  EXPECT_FALSE(metal_is_device_pointer(memory + 4));
}

// With negative strides the packed copy reads below the view's start: it
// waits for GPU writes there too.
TEST_F(MetalGraphViewTest, MaterializingBackwardsIntoARegionWaits) {
  std::vector<float> memory(12, 0.0f);
  AOTITensorHandle region_blob = nullptr;
  ASSERT_EQ(createFromBlob(memory.data() + 4, &region_blob), Error::Ok);
  const int64_t two = 2;
  AOTITensorHandle written = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          region_blob, 1, &two, &kStride, /*storage_offset=*/1, &written),
      Error::Ok);
  AOTITensorHandle after = nullptr;
  ASSERT_EQ(createFromBlob(memory.data() + 8, &after), Error::Ok);
  ASSERT_FALSE(metal_is_device_pointer(memory.data() + 8));
  AOTITensorHandle input = nullptr;
  createMatrix(kDeviceMps, &input);
  auto* input_data = static_cast<float*>(input->mutable_data_ptr());
  for (int i = 0; i < 4; i++) {
    input_data[i] = static_cast<float>(i + 1);
  }
  queueCopy(*input, *written, 2);

  // Elements 8, 6 and 4 of `memory`.
  const int64_t three = 3;
  const int64_t back = -2;
  AOTITensorHandle read = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          after, 1, &three, &back, /*storage_offset=*/0, &read),
      Error::Ok);
  const auto* got = static_cast<const float*>(read->const_data_ptr());
  EXPECT_EQ(std::vector<float>(got, got + 3), (std::vector<float>{0, 2, 0}));
}

// Sizes and extents that do not fit in 64 bits are refused, not wrapped.
TEST_F(MetalGraphViewTest, OverflowingSizesAreRefused) {
  // 2^64 elements: the count wraps to 0. The strides are the ones int32
  // dense strides wrap to, so only the count gives it away.
  const int64_t huge[4] = {65536, 65536, 65536, 65536};
  const int64_t wrapped_strides[4] = {0, 0, 65536, 1};
  AOTITensorHandle allocated = nullptr;
  EXPECT_NE(
      aoti_torch_empty_strided(
          4, huge, wrapped_strides, kFloat32, kDeviceMps, 0, &allocated),
      Error::Ok);
  EXPECT_EQ(allocated, nullptr);
  // The same for a blob: int64 elements whose count fits and whose size in
  // bytes wraps, with strides that fit.
  std::vector<float> memory(1, 0.0f);
  const int64_t blob_sizes[2] = {2147483647, 1073741825};
  const int64_t blob_strides[2] = {1073741825, 1};
  AOTITensorHandle blob = nullptr;
  EXPECT_NE(
      aoti_torch_create_tensor_from_blob_v2(
          memory.data(),
          2,
          blob_sizes,
          blob_strides,
          0,
          /*int64=*/4,
          kDeviceMps,
          0,
          &blob,
          0,
          nullptr,
          0),
      Error::Ok);
  EXPECT_EQ(blob, nullptr);
  // About 2^62 int64 elements: the count and the strides fit, the size in
  // bytes wraps.
  const int64_t large[2] = {2147483647, 2147483647};
  EXPECT_NE(
      aoti_torch_empty_strided(
          2, large, nullptr, /*int64=*/4, kDeviceMps, 0, &allocated),
      Error::Ok);
  EXPECT_EQ(allocated, nullptr);

  const int64_t base_size = 12;
  AOTITensorHandle base = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &base_size, &kStride, kFloat32, kDeviceMps, 0, &base),
      Error::Ok);
  const int64_t sizes[2] = {2147483647, 2147483647};
  const int64_t strides[2] = {2147483647, 2147483647};
  AOTITensorHandle view = nullptr;
  EXPECT_NE(
      aoti_torch__reinterpret_tensor(
          base, 2, sizes, strides, /*storage_offset=*/0, &view),
      Error::Ok);
  EXPECT_EQ(view, nullptr);
}

// Constants are only copied into a buffer of aoti_torch_mps_malloc's that no
// tensor uses, and only where they fit.
TEST_F(MetalGraphViewTest, ConstantsOnlyGoIntoAConstantsBuffer) {
  AOTITensorHandle base = nullptr;
  AOTITensorHandle view = nullptr;
  createBaseAndView(&base, &view);
  std::vector<float> data(4, 1.0f);
  auto copy = [&](void* buffer, size_t offset, size_t size) {
    return aoti_torch_mps_memcpy(
        buffer, offset, 0, size, reinterpret_cast<uint8_t*>(data.data()));
  };
  EXPECT_NE(copy(view->mutable_data_ptr(), 4, 8), Error::Ok);
  void* constants = nullptr;
  ASSERT_EQ(aoti_torch_mps_malloc(&constants, 4 * sizeof(float)), Error::Ok);
  EXPECT_NE(copy(constants, 8, 16), Error::Ok);
  EXPECT_NE(copy(static_cast<float*>(constants) + 1, 4, 4), Error::Ok);
  EXPECT_EQ(copy(constants, 8, 8), Error::Ok);
}

// An empty tensor copies as nothing, rather than failing.
TEST_F(MetalGraphViewTest, EmptyTensorsCanBeCopied) {
  const int64_t size = 0;
  AOTITensorHandle host = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &size, &kStride, kFloat32, kDeviceCpu, 0, &host),
      Error::Ok);
  AOTITensorHandle device = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &size, &kStride, kFloat32, kDeviceMps, 0, &device),
      Error::Ok);
  EXPECT_EQ(aoti_torch_copy_(device, host, 0), Error::Ok);
  EXPECT_EQ(aoti_torch_copy_(host, device, 0), Error::Ok);
}

// A blob that is not inside memory the runtime binds must not reach into
// any: it would be bound by copy, unordered with work on that memory.
TEST_F(MetalGraphViewTest, BlobReachingIntoAnAllocationIsRefused) {
  const int64_t base_size = 12;
  AOTITensorHandle base = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &base_size, &kStride, kFloat32, kDeviceCpu, 0, &base),
      Error::Ok);
  auto* memory = static_cast<float*>(base->mutable_data_ptr());
  // Starts just before the allocation and reaches into it.
  AOTITensorHandle blob = nullptr;
  EXPECT_NE(createFromBlob(memory - 2, &blob), Error::Ok);
  EXPECT_EQ(blob, nullptr);
}

// A region is refused where it would take in a tensor that starts before it
// and runs into it.
TEST_F(MetalGraphViewTest, RegionOverATensorBeforeItIsRefused) {
  std::vector<float> memory(16, 0.0f);
  AOTITensorHandle before = nullptr;
  ASSERT_EQ(createFromBlob(memory.data(), &before), Error::Ok);
  AOTITensorHandle blob = nullptr;
  ASSERT_EQ(createFromBlob(memory.data() + 2, &blob), Error::Ok);
  const int64_t two = 2;
  AOTITensorHandle view = nullptr;
  EXPECT_NE(
      aoti_torch__reinterpret_tensor(
          blob, 1, &two, &kStride, /*storage_offset=*/1, &view),
      Error::Ok);
  EXPECT_EQ(view, nullptr);
}

// A blob made inside a region grows it only where no other tensor lies.
TEST_F(MetalGraphViewTest, BlobGrowingARegionOverATensorIsRefused) {
  std::vector<float> memory(64, 0.0f);
  AOTITensorHandle first = nullptr;
  ASSERT_EQ(createFromBlob(memory.data(), &first), Error::Ok);
  const int64_t two = 2;
  AOTITensorHandle view = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          first, 1, &two, &kStride, /*storage_offset=*/1, &view),
      Error::Ok);
  AOTITensorHandle later = nullptr;
  ASSERT_EQ(createFromBlob(memory.data() + 8, &later), Error::Ok);
  AOTITensorHandle growing = nullptr;
  EXPECT_NE(
      createStridedBlob(memory.data() + 2, {32}, {1}, &growing), Error::Ok);
  EXPECT_EQ(growing, nullptr);
}

// ... or over memory a Metal buffer covers.
TEST_F(MetalGraphViewTest, RegionOverAMetalBufferIsRefused) {
  std::vector<float> memory(12, 0.0f);
  ASSERT_TRUE(metal_buffer_nocopy(memory.data() + 4, 4 * sizeof(float), true));
  AOTITensorHandle blob = nullptr;
  ASSERT_EQ(createFromBlob(memory.data(), &blob), Error::Ok);
  const int64_t size = 4;
  AOTITensorHandle view = nullptr;
  EXPECT_NE(
      aoti_torch__reinterpret_tensor(
          blob, 1, &size, &kStride, /*storage_offset=*/2, &view),
      Error::Ok);
}

// A view of the constant at the start of the constants buffer stays inside
// that constant: the next one starts where its own buffer does.
TEST_F(MetalGraphViewTest, ViewOfTheFirstConstantStaysInsideIt) {
  void* constants = nullptr;
  ASSERT_EQ(aoti_torch_mps_malloc(&constants, 8 * sizeof(float)), Error::Ok);
  std::vector<float> data(4, 1.0f);
  for (size_t offset : {size_t{0}, 4 * sizeof(float)}) {
    ASSERT_EQ(
        aoti_torch_mps_memcpy(
            constants,
            offset,
            0,
            4 * sizeof(float),
            reinterpret_cast<uint8_t*>(data.data())),
        Error::Ok);
  }
  AOTITensorHandle first = nullptr;
  ASSERT_EQ(createFromBlob(constants, &first), Error::Ok);
  const int64_t two = 2;
  const int64_t four = 4;
  AOTITensorHandle inside = nullptr;
  EXPECT_EQ(
      aoti_torch__reinterpret_tensor(
          first, 1, &two, &kStride, /*storage_offset=*/2, &inside),
      Error::Ok);
  AOTITensorHandle across = nullptr;
  EXPECT_NE(
      aoti_torch__reinterpret_tensor(
          first, 1, &four, &kStride, /*storage_offset=*/2, &across),
      Error::Ok);
}

// Copying a constant again waits for queued work that still reads it.
TEST_F(MetalGraphViewTest, ReloadingAConstantWaitsForItsReaders) {
  void* constants = nullptr;
  ASSERT_EQ(aoti_torch_mps_malloc(&constants, 8 * sizeof(float)), Error::Ok);
  std::vector<float> old_data = {1, 2, 3, 4};
  std::vector<float> new_data = {9, 9, 9, 9};
  auto load = [&](std::vector<float>& data) {
    return aoti_torch_mps_memcpy(
        constants,
        4 * sizeof(float),
        0,
        4 * sizeof(float),
        reinterpret_cast<uint8_t*>(data.data()));
  };
  ASSERT_EQ(load(old_data), Error::Ok);
  AOTITensorHandle constant = nullptr;
  ASSERT_EQ(
      createFromBlob(static_cast<float*>(constants) + 4, &constant), Error::Ok);
  AOTITensorHandle out = nullptr;
  createMatrix(kDeviceMps, &out);
  queueCopy(*constant, *out, 4);
  ASSERT_EQ(load(new_data), Error::Ok);
  getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);
  const auto* got = static_cast<const float*>(out->const_data_ptr());
  EXPECT_EQ(std::vector<float>(got, got + 4), old_data);
}

// Freeing a constants buffer holding only the constant at its start does not
// wait: queued work reads that constant through the buffer itself, which owns
// its memory and which the command buffer keeps alive until the work is done.
// Only the constants' own buffers, which do not own their memory, need a wait.
TEST_F(MetalGraphViewTest, FreeingTheFirstConstantKeepsItForItsReaders) {
  void* constants = nullptr;
  ASSERT_EQ(aoti_torch_mps_malloc(&constants, 4 * sizeof(float)), Error::Ok);
  std::vector<float> data = {1, 2, 3, 4};
  ASSERT_EQ(
      aoti_torch_mps_memcpy(
          constants,
          /*constant_offset=*/0,
          /*bytes_read=*/0,
          /*data_size=*/4 * sizeof(float),
          reinterpret_cast<uint8_t*>(data.data())),
      Error::Ok);
  AOTITensorHandle constant = nullptr;
  ASSERT_EQ(createFromBlob(constants, &constant), Error::Ok);
  AOTITensorHandle out = nullptr;
  createMatrix(kDeviceMps, &out);
  queueCopy(*constant, *out, 4);
  ASSERT_EQ(aoti_torch_mps_free(constants), Error::Ok);
  EXPECT_FALSE(getCurrentMetalStream()->isEmpty());
  getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);
  const auto* got = static_cast<const float*>(out->const_data_ptr());
  EXPECT_EQ(std::vector<float>(got, got + 4), data);
}

// Freeing a constants buffer forgets the views registered in it, and an empty
// constant maps no buffer.
TEST_F(MetalGraphViewTest, FreeingConstantsForgetsTheirViews) {
  void* constants = nullptr;
  ASSERT_EQ(aoti_torch_mps_malloc(&constants, 8 * sizeof(float)), Error::Ok);
  std::vector<float> data(4, 1.0f);
  ASSERT_EQ(
      aoti_torch_mps_memcpy(
          constants,
          4 * sizeof(float),
          0,
          4 * sizeof(float),
          reinterpret_cast<uint8_t*>(data.data())),
      Error::Ok);
  auto* memory = static_cast<float*>(constants);
  AOTITensorHandle constant = nullptr;
  ASSERT_EQ(createFromBlob(memory + 4, &constant), Error::Ok);
  const int64_t two = 2;
  AOTITensorHandle view = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          constant, 1, &two, &kStride, /*storage_offset=*/1, &view),
      Error::Ok);
  ASSERT_TRUE(metal_is_view(memory + 5));
  EXPECT_EQ(
      aoti_torch_mps_memcpy(
          constants,
          6 * sizeof(float),
          0,
          0,
          reinterpret_cast<uint8_t*>(data.data())),
      Error::Ok);
  EXPECT_FALSE(metal_is_device_pointer(memory + 6));

  ASSERT_EQ(aoti_torch_mps_free(constants), Error::Ok);
  EXPECT_FALSE(metal_is_view(memory + 5));
}

// Arguments no tensor can have are refused at the entry of each shim.
TEST_F(MetalGraphViewTest, ShimsRefuseArgumentsNoTensorCanHave) {
  const int64_t size = 4;
  const int64_t big = int64_t{1} << 32;
  const int64_t wide_stride = int64_t{1} << 31;
  AOTITensorHandle made = nullptr;
  EXPECT_NE(
      aoti_torch_empty_strided(
          -1, &size, &kStride, kFloat32, kDeviceMps, 0, &made),
      Error::Ok);
  EXPECT_NE(
      aoti_torch_empty_strided(
          1, nullptr, nullptr, kFloat32, kDeviceMps, 0, &made),
      Error::Ok);
  EXPECT_NE(
      aoti_torch_empty_strided(
          1, &size, &kStride, kFloat32, kDeviceMps, 0, nullptr),
      Error::Ok);
  EXPECT_NE(
      aoti_torch_empty_strided(
          1, &big, &kStride, kFloat32, kDeviceCpu, 0, &made),
      Error::Ok);
  EXPECT_NE(
      aoti_torch_empty_strided(
          1, &size, &wide_stride, kFloat32, kDeviceMps, 0, &made),
      Error::Ok);
  EXPECT_EQ(made, nullptr);

  std::vector<float> memory(4, 0.0f);
  AOTITensorHandle blob = nullptr;
  EXPECT_NE(createStridedBlob(memory.data(), {big}, {1}, &blob), Error::Ok);
  // A stride that does not fit, on a dim that steps.
  EXPECT_NE(
      createStridedBlob(memory.data(), {2, 2}, {wide_stride, 1}, &blob),
      Error::Ok);
  // An offset whose size in bytes wraps to 0, back at the blob itself.
  const int64_t two = 2;
  AOTITensorHandle plain = nullptr;
  ASSERT_EQ(createFromBlob(memory.data(), &plain), Error::Ok);
  AOTITensorHandle wrapped = nullptr;
  EXPECT_NE(
      aoti_torch__reinterpret_tensor(
          plain,
          1,
          &two,
          &kStride,
          /*storage_offset=*/int64_t{1} << 62,
          &wrapped),
      Error::Ok);
  EXPECT_EQ(wrapped, nullptr);
  EXPECT_NE(
      aoti_torch_create_tensor_from_blob_v2(
          memory.data(),
          1,
          &size,
          &kStride,
          /*storage_offset=*/INT64_MAX,
          kFloat32,
          kDeviceMps,
          0,
          &blob,
          0,
          nullptr,
          0),
      Error::Ok);
  EXPECT_EQ(blob, nullptr);

  const int64_t base_size = 8;
  AOTITensorHandle base = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &base_size, &kStride, kFloat32, kDeviceMps, 0, &base),
      Error::Ok);
  AOTITensorHandle view = nullptr;
  EXPECT_NE(
      aoti_torch__reinterpret_tensor(base, 1, &big, &kStride, 0, &view),
      Error::Ok);
  EXPECT_NE(
      aoti_torch__reinterpret_tensor(
          base, 1, &size, &kStride, INT64_MAX, &view),
      Error::Ok);
  EXPECT_EQ(view, nullptr);
}

// Tensors of different ranks copy as their bytes when both are dense in
// row-major order, and are refused otherwise rather than read past their dims.
TEST_F(MetalGraphViewTest, CopyAcrossRanks) {
  const int64_t host_sizes[3] = {4, 1, 5};
  const int64_t host_strides[3] = {5, 5, 1};
  AOTITensorHandle host = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          3, host_sizes, host_strides, kFloat32, kDeviceCpu, 0, &host),
      Error::Ok);
  auto* host_data = static_cast<float*>(host->mutable_data_ptr());
  for (int i = 0; i < 20; i++) {
    host_data[i] = static_cast<float>(i);
  }
  const int64_t device_sizes[2] = {4, 5};
  const int64_t device_strides[2] = {5, 1};
  AOTITensorHandle device = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          2, device_sizes, device_strides, kFloat32, kDeviceMps, 0, &device),
      Error::Ok);
  ASSERT_EQ(aoti_torch_copy_(device, host, 0), Error::Ok);
  getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);
  const auto* got = static_cast<const float*>(device->const_data_ptr());
  EXPECT_EQ(got[7], 7.0f);

  const int64_t transposed_sizes[2] = {5, 4};
  const int64_t transposed_strides[2] = {1, 5};
  AOTITensorHandle transposed = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          2,
          transposed_sizes,
          transposed_strides,
          kFloat32,
          kDeviceMps,
          0,
          &transposed),
      Error::Ok);
  EXPECT_NE(aoti_torch_copy_(transposed, host, 0), Error::Ok);
}

// An empty view at an offset into a buffer goes through a graph: it needs no
// buffer of its own there.
TEST_F(MetalGraphViewTest, EmptyViewsAtOffsetsGoThroughAGraph) {
  const int64_t base_size = 16;
  AOTITensorHandle base = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &base_size, &kStride, kFloat32, kDeviceMps, 0, &base),
      Error::Ok);
  const int64_t four = 4;
  AOTITensorHandle half = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          base, 1, &four, &kStride, /*storage_offset=*/8, &half),
      Error::Ok);
  const int64_t left_sizes[2] = {2, 0};
  const int64_t left_strides[2] = {1, 1};
  AOTITensorHandle left = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          half, 2, left_sizes, left_strides, /*storage_offset=*/0, &left),
      Error::Ok);
  const int64_t right_sizes[2] = {0, 2};
  const int64_t right_strides[2] = {2, 1};
  AOTITensorHandle right = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          half, 2, right_sizes, right_strides, /*storage_offset=*/0, &right),
      Error::Ok);
  AOTITensorHandle out = nullptr;
  createMatrix(kDeviceMps, &out);
  EXPECT_EQ(aoti_torch_mps_mm_out(out, left, right), Error::Ok);
  getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);
}

// A CPU region no queued work has bound goes without a wait; one that work
// bound since the stream last waited waits first.
TEST_F(MetalGraphViewTest, FreeingACpuRegionWaitsOnlyIfItWasBound) {
  AOTITensorHandle input = nullptr;
  createMatrix(kDeviceMps, &input);
  AOTITensorHandle out = nullptr;
  createMatrix(kDeviceMps, &out);
  const int64_t base_size = 12;
  auto make = [&](AOTITensorHandle* base, AOTITensorHandle* view) {
    ASSERT_EQ(
        aoti_torch_empty_strided(
            1, &base_size, &kStride, kFloat32, kDeviceCpu, 0, base),
        Error::Ok);
    ASSERT_EQ(
        aoti_torch__reinterpret_tensor(
            *base, 2, kMatrixSizes, kMatrixStrides, /*storage_offset=*/4, view),
        Error::Ok);
  };

  AOTITensorHandle unbound = nullptr;
  AOTITensorHandle unbound_view = nullptr;
  make(&unbound, &unbound_view);
  getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);
  queueCopy(*input, *out, 4);
  ASSERT_EQ(aoti_torch_delete_tensor_object(unbound_view), Error::Ok);
  ASSERT_EQ(aoti_torch_delete_tensor_object(unbound), Error::Ok);
  EXPECT_FALSE(getCurrentMetalStream()->isEmpty());

  AOTITensorHandle bound = nullptr;
  AOTITensorHandle bound_view = nullptr;
  make(&bound, &bound_view);
  queueCopy(*input, *bound_view, 4);
  ASSERT_EQ(aoti_torch_delete_tensor_object(bound_view), Error::Ok);
  ASSERT_EQ(aoti_torch_delete_tensor_object(bound), Error::Ok);
  EXPECT_TRUE(getCurrentMetalStream()->isEmpty());
}

// Tearing everything down waits for queued work first, e.g. after a failed
// run: the memory of tensors and constants goes with it.
TEST_F(MetalGraphViewTest, CleanupWaitsForQueuedWork) {
  AOTITensorHandle input = nullptr;
  createMatrix(kDeviceMps, &input);
  AOTITensorHandle out = nullptr;
  createMatrix(kDeviceMps, &out);
  queueCopy(*input, *out, 4);
  cleanup_memory();
  EXPECT_TRUE(getCurrentMetalStream()->isEmpty());
}

// A blob at the start of a Metal buffer must fit in it, and computed
// contiguous strides must fit a tensor's too.
TEST_F(MetalGraphViewTest, BlobsMustFitWhereTheyStart) {
  void* buffer = nullptr;
  ASSERT_EQ(aoti_torch_mps_malloc(&buffer, 4 * sizeof(float)), Error::Ok);
  AOTITensorHandle blob = nullptr;
  EXPECT_NE(createStridedBlob(buffer, {8}, {1}, &blob), Error::Ok);
  EXPECT_EQ(blob, nullptr);
  EXPECT_EQ(createStridedBlob(buffer, {4}, {1}, &blob), Error::Ok);

  std::vector<float> memory(4, 0.0f);
  const int64_t sizes[3] = {3, 1073741824, 2};
  AOTITensorHandle wide = nullptr;
  EXPECT_NE(
      aoti_torch_create_tensor_from_blob_v2(
          memory.data(),
          3,
          sizes,
          /*strides_ptr=*/nullptr,
          0,
          kFloat32,
          kDeviceMps,
          0,
          &wide,
          0,
          nullptr,
          0),
      Error::Ok);
  EXPECT_EQ(wide, nullptr);
}

// A blob at the start of a CPU region whose own tensor is gone joins the
// region, growing it where nothing else lies, instead of being left out.
TEST_F(MetalGraphViewTest, BlobAtARegionStartJoinsIt) {
  std::vector<float> memory(16, 0.0f);
  AOTITensorHandle outer = nullptr;
  ASSERT_EQ(createStridedBlob(memory.data(), {2}, {1}, &outer), Error::Ok);
  const int64_t two = 2;
  AOTITensorHandle view = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          outer, 1, &two, &kStride, /*storage_offset=*/2, &view),
      Error::Ok);
  ASSERT_EQ(aoti_torch_delete_tensor_object(outer), Error::Ok);

  AOTITensorHandle blob = nullptr;
  ASSERT_EQ(createStridedBlob(memory.data(), {12}, {1}, &blob), Error::Ok);
  EXPECT_TRUE(metal_is_view(memory.data()));
  void* found = nullptr;
  size_t found_nbytes = 0;
  bool cpu = false;
  ASSERT_TRUE(metal_find_memory(memory.data(), &found, &cpu, &found_nbytes));
  EXPECT_GE(found_nbytes, 12 * sizeof(float));
  ASSERT_EQ(aoti_torch_delete_tensor_object(view), Error::Ok);
  EXPECT_TRUE(metal_is_view(memory.data()));
}

// A blob inside the first constant stays inside it, as views of it do.
TEST_F(MetalGraphViewTest, BlobInsideTheFirstConstantStaysInsideIt) {
  void* constants = nullptr;
  ASSERT_EQ(aoti_torch_mps_malloc(&constants, 8 * sizeof(float)), Error::Ok);
  std::vector<float> data(4, 1.0f);
  for (size_t offset : {size_t{0}, 4 * sizeof(float)}) {
    ASSERT_EQ(
        aoti_torch_mps_memcpy(
            constants,
            offset,
            0,
            4 * sizeof(float),
            reinterpret_cast<uint8_t*>(data.data())),
        Error::Ok);
  }
  auto* memory = static_cast<float*>(constants);
  AOTITensorHandle blob = nullptr;
  EXPECT_NE(createStridedBlob(memory, {8}, {1}, &blob), Error::Ok);
  EXPECT_NE(createStridedBlob(memory + 2, {4}, {1}, &blob), Error::Ok);
  EXPECT_EQ(blob, nullptr);
  EXPECT_EQ(createStridedBlob(memory + 2, {2}, {1}, &blob), Error::Ok);
}

// A constant of size 0 has no memory: AOTInductor wraps a null pointer for
// it. Such a tensor, views of it and copies of its handle work, and hold
// nothing; only a tensor with no elements can have a null pointer.
TEST_F(MetalGraphViewTest, EmptyTensorsWithoutMemory) {
  // Two of them, as a model with two empty constants makes.
  AOTITensorHandle empty = nullptr;
  ASSERT_EQ(createStridedBlob(nullptr, {0, 4}, {4, 1}, &empty), Error::Ok);
  AOTITensorHandle other = nullptr;
  ASSERT_EQ(createStridedBlob(nullptr, {0}, {1}, &other), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(other), Error::Ok);
  const int64_t size = 0;
  AOTITensorHandle view = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(empty, 1, &size, &kStride, 0, &view),
      Error::Ok);
  AOTITensorHandle copy = nullptr;
  ASSERT_EQ(aoti_torch_new_tensor_handle(empty, &copy), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(view), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(copy), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(empty), Error::Ok);

  AOTITensorHandle not_empty = nullptr;
  EXPECT_NE(createStridedBlob(nullptr, {2}, {1}, &not_empty), Error::Ok);
  EXPECT_EQ(not_empty, nullptr);
}

// A tensor of 2^31 elements or more is fine as long as its strides fit, e.g.
// logits over a long sequence. Only the outermost dim's size is not a
// stride.
TEST_F(MetalGraphViewTest, TensorsOfMoreThan2To31Elements) {
  const int64_t sizes[2] = {8192, 262144};
  const int64_t strides[2] = {262144, 1};
  std::vector<float> memory(1, 0.0f);
  AOTITensorHandle big = nullptr;
  EXPECT_EQ(
      aoti_torch_create_tensor_from_blob_v2(
          memory.data(),
          2,
          sizes,
          strides,
          0,
          kFloat32,
          kDeviceMps,
          0,
          &big,
          0,
          nullptr,
          0),
      Error::Ok);
}

// A dim of size 1 never steps, so its stride need not fit: logits of one long
// sequence, [1, T, V] with T * V >= 2^31, are made by viewing the [T, V]
// result with the stride T * V for the batch dim, or allocated without
// strides.
TEST_F(MetalGraphViewTest, LogitsOfOneLongSequence) {
  std::vector<float> memory(1, 0.0f);
  const int64_t sizes[3] = {1, 8192, 262144};
  const int64_t strides[3] = {int64_t{8192} * 262144, 262144, 1};
  AOTITensorHandle given = nullptr;
  EXPECT_EQ(
      aoti_torch_create_tensor_from_blob_v2(
          memory.data(),
          3,
          sizes,
          strides,
          0,
          kFloat32,
          kDeviceMps,
          0,
          &given,
          0,
          nullptr,
          0),
      Error::Ok);
  AOTITensorHandle computed = nullptr;
  EXPECT_EQ(
      aoti_torch_create_tensor_from_blob_v2(
          memory.data() + 1,
          3,
          sizes,
          nullptr,
          0,
          kFloat32,
          kDeviceMps,
          0,
          &computed,
          0,
          nullptr,
          0),
      Error::Ok);
  AOTITensorHandle view = nullptr;
  EXPECT_EQ(
      aoti_torch__reinterpret_tensor(given, 3, sizes, strides, 0, &view),
      Error::Ok);
  // T * V = 2^32 + 2^20: truncated to int32, the unused stride would look
  // like the largest one, and the batch dim like the outermost.
  const int64_t wrapping_sizes[3] = {1, 16388, 262144};
  const int64_t wrapping_strides[3] = {int64_t{16388} * 262144, 262144, 1};
  AOTITensorHandle wrapping = nullptr;
  EXPECT_EQ(
      aoti_torch_create_tensor_from_blob_v2(
          memory.data() + 2,
          3,
          wrapping_sizes,
          wrapping_strides,
          0,
          kFloat32,
          kDeviceMps,
          0,
          &wrapping,
          0,
          nullptr,
          0),
      Error::Ok);
}

// An empty blob inside a CPU allocation joins the allocation's region too, so a
// view made from it does not start a region of its own inside the allocation.
TEST_F(MetalGraphViewTest, EmptyBlobInsideCpuAllocationJoinsItsRegion) {
  const int64_t base_size = 12;
  AOTITensorHandle base = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &base_size, &kStride, kFloat32, kDeviceCpu, 0, &base),
      Error::Ok);
  auto* memory = static_cast<float*>(base->mutable_data_ptr());
  AOTITensorHandle empty = nullptr;
  ASSERT_EQ(createStridedBlob(memory + 4, {0}, {1}, &empty), Error::Ok);
  const int64_t size = 2;
  AOTITensorHandle view = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          empty, 1, &size, &kStride, /*storage_offset=*/1, &view),
      Error::Ok);
  void* found = nullptr;
  size_t found_nbytes = 0;
  bool cpu = false;
  ASSERT_TRUE(
      metal_find_memory(view->mutable_data_ptr(), &found, &cpu, &found_nbytes));
  EXPECT_EQ(found, memory);
  AOTITensorHandle base_view = nullptr;
  EXPECT_EQ(
      aoti_torch__reinterpret_tensor(
          base,
          2,
          kMatrixSizes,
          kMatrixStrides,
          /*storage_offset=*/4,
          &base_view),
      Error::Ok);
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
