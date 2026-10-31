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
#include <cstring>
#include <memory>
#include <stdexcept>
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
// DeviceType::CPU.
constexpr int32_t kDeviceCpu = 0;

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

  // A hand-written kernel copying buffer 0 to buffer 1.
  std::shared_ptr<ETMetalKernelFunction> copyKernel() {
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
    return library.getKernelFunction("copy_float");
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

// Copying a strided view into Metal memory stays on the stream: the gather
// writes straight into the destination, here a view at an offset into another
// buffer, and nothing is waited for.
TEST_F(MetalStridedViewTest, CopyOfStridedViewIntoMetalMemoryDoesNotWait) {
  AOTITensorHandle base = nullptr;
  AOTITensorHandle view = nullptr;
  createBaseAndRightHalf(&base, &view);
  const int64_t target_size = 16;
  const int64_t target_stride = 1;
  AOTITensorHandle target = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &target_size, &target_stride, kFloat32, kDeviceMps, 0, &target),
      Error::Ok);
  auto* target_data = static_cast<float*>(target->mutable_data_ptr());
  std::fill_n(target_data, 16, -1.0f);
  const int64_t sizes[2] = {4, 2};
  const int64_t strides[2] = {2, 1};
  AOTITensorHandle out = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          target, 2, sizes, strides, /*storage_offset=*/4, &out),
      Error::Ok);
  const uint64_t waits = getCurrentMetalStream()->completedWaits();

  ASSERT_EQ(aoti_torch_copy_(out, view, /*non_blocking=*/0), Error::Ok);
  EXPECT_EQ(getCurrentMetalStream()->completedWaits(), waits);
  getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);
  EXPECT_EQ(
      std::vector<float>(target_data, target_data + 16),
      (std::vector<float>{
          -1, -1, -1, -1, 2, 3, 6, 7, 10, 11, 14, 15, -1, -1, -1, -1}));
}

// A destination overlapping the view is copied through a packed buffer. One
// gather straight into it would race: element i of the view (at 2i) is where
// element 2i - kCount of the destination goes, which an earlier thread writes.
// Earlier threadgroups run first, so at this size the race all but always
// shows, though the GPU does not promise that order.
TEST_F(MetalStridedViewTest, CopyOfStridedViewIntoItsOwnBufferIsCorrect) {
  constexpr int64_t kCount = 1 << 20;
  const int64_t base_size = 2 * kCount;
  const int64_t unit = 1;
  AOTITensorHandle base = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &base_size, &unit, kFloat32, kDeviceMps, 0, &base),
      Error::Ok);
  auto* data = static_cast<float*>(base->mutable_data_ptr());
  for (int64_t i = 0; i < base_size; i++) {
    data[i] = static_cast<float>(i);
  }
  const int64_t two = 2;
  AOTITensorHandle view = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          base, 1, &kCount, &two, /*storage_offset=*/0, &view),
      Error::Ok);
  ASSERT_TRUE(metal_is_strided_view(view));
  AOTITensorHandle out = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          base, 1, &kCount, &unit, /*storage_offset=*/kCount, &out),
      Error::Ok);

  ASSERT_EQ(aoti_torch_copy_(out, view, /*non_blocking=*/0), Error::Ok);
  getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);
  int64_t wrong = 0;
  for (int64_t i = 0; i < kCount; i++) {
    wrong += data[kCount + i] != static_cast<float>(2 * i);
  }
  EXPECT_EQ(wrong, 0);
}

// CPU code reads CPU memory directly, so a copy of a strided view into it is
// done by the time aoti_torch_copy_ returns, even into a view of CPU memory,
// which has a Metal buffer (metal_register_cpu_view).
TEST_F(MetalStridedViewTest, CopyOfStridedViewIntoCpuMemoryIsDoneOnReturn) {
  AOTITensorHandle base = nullptr;
  AOTITensorHandle view = nullptr;
  createBaseAndRightHalf(&base, &view);
  const int64_t cpu_size = 12;
  const int64_t cpu_stride = 1;
  AOTITensorHandle cpu = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &cpu_size, &cpu_stride, kFloat32, kDeviceCpu, 0, &cpu),
      Error::Ok);
  const int64_t sizes[2] = {4, 2};
  const int64_t strides[2] = {2, 1};
  AOTITensorHandle out = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          cpu, 2, sizes, strides, /*storage_offset=*/4, &out),
      Error::Ok);
  ASSERT_TRUE(metal_is_cpu_view(out->mutable_data_ptr()));

  ASSERT_EQ(aoti_torch_copy_(out, view, /*non_blocking=*/0), Error::Ok);
  const auto* got = static_cast<const float*>(out->const_data_ptr());
  EXPECT_EQ(
      std::vector<float>(got, got + 8),
      (std::vector<float>{2, 3, 6, 7, 10, 11, 14, 15}));

  // The start of the CPU allocation is bound into its region's buffer too.
  AOTITensorHandle start = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          cpu, 2, sizes, strides, /*storage_offset=*/0, &start),
      Error::Ok);
  ASSERT_TRUE(metal_is_cpu_memory(start->mutable_data_ptr()));
  ASSERT_FALSE(metal_is_cpu_view(start->mutable_data_ptr()));
  ASSERT_EQ(aoti_torch_copy_(start, view, /*non_blocking=*/0), Error::Ok);
  got = static_cast<const float*>(start->const_data_ptr());
  EXPECT_EQ(
      std::vector<float>(got, got + 8),
      (std::vector<float>{2, 3, 6, 7, 10, 11, 14, 15}));
  getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);
}

// A hand-written kernel reads a strided view through a packed copy, encoded
// on the same encoder the kernel is being set up on.
TEST_F(MetalStridedViewTest, HandWrittenKernelReadsPackedCopy) {
  AOTITensorHandle base = nullptr;
  AOTITensorHandle view = nullptr;
  createBaseAndRightHalf(&base, &view);
  const int64_t out_size = 8;
  const int64_t out_stride = 1;
  AOTITensorHandle out = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &out_size, &out_stride, kFloat32, kDeviceMps, 0, &out),
      Error::Ok);

  auto copy = copyKernel();
  copy->runCommandBlock([&]() {
    copy->startEncoding();
    copy->setArg(0, *view);
    copy->setArg(1, *out, ETMetalKernelFunction::ArgAccess::kWrite);
    copy->dispatchSingle(8);
  });
  getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);

  const float* got = static_cast<const float*>(out->const_data_ptr());
  EXPECT_EQ(
      std::vector<float>(got, got + 8),
      (std::vector<float>{2, 3, 6, 7, 10, 11, 14, 15}));
}

// Writes to a packed copy would be lost, so a hand-written kernel may not
// write through a strided view.
TEST_F(MetalStridedViewTest, HandWrittenKernelMayNotWriteAStridedView) {
  AOTITensorHandle base = nullptr;
  AOTITensorHandle view = nullptr;
  createBaseAndRightHalf(&base, &view);

  auto copy = copyKernel();
  copy->runCommandBlock([&]() {
    copy->startEncoding();
    EXPECT_THROW(
        copy->setArg(1, *view, ETMetalKernelFunction::ArgAccess::kWrite),
        std::runtime_error);
  });
}

// A view the packed copy cannot gather is not recorded: an empty one (the
// gather divides by every size), one with negative strides or one with more
// than 16 dims.
TEST_F(MetalStridedViewTest, ViewsTheGatherCannotPackAreNotRecorded) {
  AOTITensorHandle base = nullptr;
  AOTITensorHandle view = nullptr;
  createBaseAndRightHalf(&base, &view);
  metal_forget_strided_view(view);
  EXPECT_FALSE(metal_record_strided_view(view, {0, 2}, {4, 1}));
  EXPECT_FALSE(metal_record_strided_view(view, {4, 2}, {-4, 1}));
  EXPECT_FALSE(metal_record_strided_view(view, {4, 2}, {4}));
  // Nor one with more dims than the gather takes.
  EXPECT_FALSE(metal_record_strided_view(
      view, std::vector<int64_t>(17, 1), std::vector<int64_t>(17, 1)));
  EXPECT_FALSE(metal_is_strided_view(view));
  EXPECT_TRUE(metal_record_strided_view(view, {4, 2}, {4, 1}));

  // Nor is a view kept in its buffer with negative strides.
  const int64_t sizes[2] = {4, 2};
  const int64_t strides[2] = {-4, 1};
  AOTITensorHandle backwards = nullptr;
  EXPECT_NE(
      aoti_torch__reinterpret_tensor(
          base, 2, sizes, strides, /*storage_offset=*/12, &backwards),
      Error::Ok);
  EXPECT_EQ(backwards, nullptr);
}

// The gather indexes the memory a view spans in 64 bits: a chunk of logits
// whose rows reach past 2^32 elements into their buffer is still a view it
// can pack.
TEST_F(MetalStridedViewTest, ViewsSpanningMoreThan32BitsAreRecorded) {
  AOTITensorHandle base = nullptr;
  AOTITensorHandle view = nullptr;
  createBaseAndRightHalf(&base, &view);
  metal_forget_strided_view(view);
  EXPECT_TRUE(metal_record_strided_view(view, {4, 2}, {int64_t{1} << 33, 1}));
  EXPECT_TRUE(metal_is_strided_view(view));
}

// Packs a view reaching past 2^32 elements into its buffer. It needs a 6.4 GB
// buffer, so it only runs when asked for
// (--gtest_also_run_disabled_tests).
TEST_F(MetalStridedViewTest, DISABLED_PackingAViewSpanningMoreThan32Bits) {
  constexpr int32_t kUint8 = 0;
  constexpr int64_t kRow = (int64_t{1} << 31) - 1;
  const int64_t base_sizes[2] = {3, kRow};
  const int64_t base_strides[2] = {kRow, 1};
  AOTITensorHandle base = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          2, base_sizes, base_strides, kUint8, kDeviceMps, 0, &base),
      Error::Ok);
  auto* data = static_cast<uint8_t*>(base->mutable_data_ptr());
  // Every other byte of the first two in each row: holes, so not packed.
  const int64_t view_sizes[2] = {3, 2};
  const int64_t view_strides[2] = {kRow, 3};
  for (int64_t r = 0; r < 3; r++) {
    for (int64_t c = 0; c < 2; c++) {
      data[r * kRow + c * 3] = static_cast<uint8_t>(10 * r + c + 1);
    }
  }
  AOTITensorHandle view = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          base, 2, view_sizes, view_strides, /*storage_offset=*/0, &view),
      Error::Ok);
  ASSERT_TRUE(metal_is_strided_view(view));
  const int64_t out_strides[2] = {2, 1};
  AOTITensorHandle out = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          2, view_sizes, out_strides, kUint8, kDeviceMps, 0, &out),
      Error::Ok);
  ASSERT_EQ(aoti_torch_copy_(out, view, /*non_blocking=*/0), Error::Ok);
  getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);
  const auto* got = static_cast<const uint8_t*>(out->const_data_ptr());
  EXPECT_EQ(
      std::vector<uint8_t>(got, got + 6),
      (std::vector<uint8_t>{1, 2, 11, 12, 21, 22}));
}

// A copy into memory the view spans is detected, whatever order the GPU would
// run a direct gather in.
TEST_F(MetalStridedViewTest, DestinationOverlappingTheViewIsDetected) {
  AOTITensorHandle base = nullptr;
  AOTITensorHandle view = nullptr;
  createBaseAndRightHalf(&base, &view);
  auto* view_data = static_cast<const uint8_t*>(view->const_data_ptr());
  const size_t nbytes = view->nbytes();
  EXPECT_TRUE(metal_strided_view_overlaps(*view, view_data, nbytes));
  EXPECT_TRUE(metal_strided_view_overlaps(*view, view_data - 4, 8));
  std::vector<uint8_t> elsewhere(nbytes);
  EXPECT_FALSE(metal_strided_view_overlaps(*view, elsewhere.data(), nbytes));
  EXPECT_FALSE(metal_strided_view_overlaps(*base, view_data, nbytes));
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

// CPU memory, and views of it that have a Metal buffer of their own, cannot be
// bound where they are: a view of them that is not densely packed is copied
// into a buffer of its own instead.
TEST_F(MetalStridedViewTest, NonPackedViewOfCpuBackedViewIsMaterialized) {
  const int64_t base_size = 20;
  const int64_t unit = 1;
  AOTITensorHandle base = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &base_size, &unit, kFloat32, kDeviceCpu, 0, &base),
      Error::Ok);
  float* data = static_cast<float*>(base->mutable_data_ptr());
  for (int i = 0; i < 20; i++) {
    data[i] = static_cast<float>(i);
  }
  const int64_t cpu_view_size = 16;
  AOTITensorHandle cpu_view = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          base, 1, &cpu_view_size, &unit, /*storage_offset=*/4, &cpu_view),
      Error::Ok);
  ASSERT_TRUE(metal_is_cpu_view(cpu_view->mutable_data_ptr()));

  const int64_t sizes[2] = {4, 2};
  const int64_t strides[2] = {4, 1};
  AOTITensorHandle half = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          cpu_view, 2, sizes, strides, /*storage_offset=*/2, &half),
      Error::Ok);
  EXPECT_FALSE(metal_is_strided_view(half));
  const float* got = static_cast<const float*>(half->const_data_ptr());
  EXPECT_EQ(
      std::vector<float>(got, got + 8),
      (std::vector<float>{6, 7, 10, 11, 14, 15, 18, 19}));
}

// A kernel can write CPU memory through the Metal buffer of a view of it; a
// copy of that memory made afterwards has to see the write.
TEST_F(MetalStridedViewTest, MaterializingCpuMemoryWaitsForGpuWrites) {
  const int64_t base_size = 12;
  const int64_t unit = 1;
  AOTITensorHandle base = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &base_size, &unit, kFloat32, kDeviceCpu, 0, &base),
      Error::Ok);
  std::fill_n(static_cast<float*>(base->mutable_data_ptr()), 12, 0.0f);
  const int64_t four = 4;
  AOTITensorHandle target = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          base, 1, &four, &unit, /*storage_offset=*/4, &target),
      Error::Ok);
  AOTITensorHandle source = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(
          1, &four, &unit, kFloat32, kDeviceMps, 0, &source),
      Error::Ok);
  float* src = static_cast<float*>(source->mutable_data_ptr());
  for (int i = 0; i < 4; i++) {
    src[i] = static_cast<float>(31 + i);
  }

  auto copy = copyKernel();
  copy->runCommandBlock([&]() {
    copy->startEncoding();
    copy->setArg(0, *source);
    copy->setArg(1, *target, ETMetalKernelFunction::ArgAccess::kWrite);
    copy->dispatchSingle(4);
  });

  // Elements 4, 5, 8 and 9: not densely packed, so copied on the CPU.
  const int64_t sizes[2] = {2, 2};
  const int64_t strides[2] = {4, 1};
  AOTITensorHandle read = nullptr;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          base, 2, sizes, strides, /*storage_offset=*/4, &read),
      Error::Ok);
  const float* got = static_cast<const float*>(read->const_data_ptr());
  EXPECT_EQ(
      std::vector<float>(got, got + 4), (std::vector<float>{31, 32, 0, 0}));
}

// The packed copy of a strided view is encoded on the same encoder and uses
// argument slots of its own. A kernel's arguments in those slots, bound before
// the strided one, have to be bound again afterwards.
TEST_F(MetalStridedViewTest, PackingKeepsArgumentsAlreadyBound) {
  AOTITensorHandle base = nullptr;
  AOTITensorHandle view = nullptr;
  createBaseAndRightHalf(&base, &view);
  const int64_t size = 8;
  const int64_t unit = 1;
  AOTITensorHandle addend = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(1, &size, &unit, kFloat32, kDeviceMps, 0, &addend),
      Error::Ok);
  std::fill_n(static_cast<float*>(addend->mutable_data_ptr()), 8, 100.0f);
  AOTITensorHandle out = nullptr;
  ASSERT_EQ(
      aoti_torch_empty_strided(1, &size, &unit, kFloat32, kDeviceMps, 0, &out),
      Error::Ok);

  static ETMetalShaderLibrary library(R"(
    #include <metal_stdlib>
    using namespace metal;
    kernel void add_float(
        device const float* a [[buffer(0)]],
        device float* out [[buffer(1)]],
        device const float* b [[buffer(29)]],
        constant float& scale [[buffer(30)]],
        uint i [[thread_position_in_grid]]) {
      out[i] = a[i] + b[i] * scale;
    }
  )");
  auto add = library.getKernelFunction("add_float");
  ASSERT_NE(add, nullptr);
  add->runCommandBlock([&]() {
    add->startEncoding();
    add->setArg(29, *addend);
    add->setArg(30, 2.0f);
    add->setArg(1, *out, ETMetalKernelFunction::ArgAccess::kWrite);
    add->setArg(0, *view);
    add->dispatchSingle(8);
  });
  getCurrentMetalStream()->synchronize(SyncType::COMMIT_AND_WAIT);

  const float* got = static_cast<const float*>(out->const_data_ptr());
  EXPECT_EQ(
      std::vector<float>(got, got + 8),
      (std::vector<float>{202, 203, 206, 207, 210, 211, 214, 215}));
}
