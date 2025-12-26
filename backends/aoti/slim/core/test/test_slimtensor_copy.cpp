/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/aoti/slim/core/SlimTensor.h>
#include <executorch/backends/aoti/slim/core/Storage.h>

namespace executorch::backends::aoti::slim {

// Helper function to create a CPU storage with given size
Storage make_cpu_storage(size_t nbytes) {
  return Storage(new MaybeOwningStorage(CPU_DEVICE, nbytes));
}

// Helper function to create a contiguous float tensor and fill with values
SlimTensor make_filled_tensor(
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides,
    const std::vector<float>& values) {
  size_t numel = 1;
  for (auto s : sizes) {
    numel *= static_cast<size_t>(s);
  }
  size_t nbytes = numel * sizeof(float);
  Storage storage = make_cpu_storage(nbytes);

  SlimTensor tensor(
      std::move(storage),
      makeArrayRef(sizes),
      makeArrayRef(strides),
      c10::ScalarType::Float);

  float* data = static_cast<float*>(tensor.data_ptr());
  for (size_t i = 0; i < values.size() && i < numel; ++i) {
    data[i] = values[i];
  }

  return tensor;
}

// =============================================================================
// Basic Copy Tests
// =============================================================================

TEST(SlimTensorCopyTest, CopyContiguousTensors) {
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = {3, 1};
  std::vector<float> src_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  SlimTensor src = make_filled_tensor(sizes, strides, src_values);
  SlimTensor dst =
      make_filled_tensor(sizes, strides, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});

  dst.copy_(src);

  float* dst_data = static_cast<float*>(dst.data_ptr());
  EXPECT_FLOAT_EQ(dst_data[0], 1.0f);
  EXPECT_FLOAT_EQ(dst_data[1], 2.0f);
  EXPECT_FLOAT_EQ(dst_data[2], 3.0f);
  EXPECT_FLOAT_EQ(dst_data[3], 4.0f);
  EXPECT_FLOAT_EQ(dst_data[4], 5.0f);
  EXPECT_FLOAT_EQ(dst_data[5], 6.0f);
}

TEST(SlimTensorCopyTest, CopyOneDimensional) {
  std::vector<int64_t> sizes = {5};
  std::vector<int64_t> strides = {1};
  std::vector<float> src_values = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};

  SlimTensor src = make_filled_tensor(sizes, strides, src_values);
  SlimTensor dst =
      make_filled_tensor(sizes, strides, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f});

  dst.copy_(src);

  float* dst_data = static_cast<float*>(dst.data_ptr());
  for (size_t i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(dst_data[i], src_values[i]);
  }
}

TEST(SlimTensorCopyTest, CopyThreeDimensional) {
  std::vector<int64_t> sizes = {2, 2, 2};
  std::vector<int64_t> strides = {4, 2, 1};
  std::vector<float> src_values = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

  SlimTensor src = make_filled_tensor(sizes, strides, src_values);
  SlimTensor dst = make_filled_tensor(
      sizes, strides, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});

  dst.copy_(src);

  float* dst_data = static_cast<float*>(dst.data_ptr());
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_FLOAT_EQ(dst_data[i], src_values[i]);
  }
}

TEST(SlimTensorCopyTest, CopyEmptyTensor) {
  std::vector<int64_t> sizes = {0, 3};
  std::vector<int64_t> strides = {3, 1};
  Storage storage1 = make_cpu_storage(0);
  Storage storage2 = make_cpu_storage(0);

  SlimTensor src(
      std::move(storage1),
      makeArrayRef(sizes),
      makeArrayRef(strides),
      c10::ScalarType::Float);

  SlimTensor dst(
      std::move(storage2),
      makeArrayRef(sizes),
      makeArrayRef(strides),
      c10::ScalarType::Float);

  // Should not crash
  dst.copy_(src);

  EXPECT_EQ(dst.numel(), 0u);
}

TEST(SlimTensorCopyTest, CopyReturnsSelf) {
  std::vector<int64_t> sizes = {2, 2};
  std::vector<int64_t> strides = {2, 1};
  std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f};

  SlimTensor src = make_filled_tensor(sizes, strides, values);
  SlimTensor dst = make_filled_tensor(sizes, strides, {0.0f, 0.0f, 0.0f, 0.0f});

  SlimTensor& result = dst.copy_(src);

  EXPECT_EQ(&result, &dst);
}

// =============================================================================
// Non-Contiguous Copy Tests
// =============================================================================

TEST(SlimTensorCopyTest, CopyNonContiguousSrc) {
  // Source is transposed (non-contiguous)
  std::vector<int64_t> src_sizes = {2, 3};
  std::vector<int64_t> src_strides = {1, 2};

  // Allocate storage for 6 elements in transposed layout
  Storage src_storage = make_cpu_storage(6 * sizeof(float));
  float* src_data = static_cast<float*>(src_storage->data());
  // Physical layout: [0,3] [1,4] [2,5] for logical [0,1,2; 3,4,5]
  src_data[0] = 0.0f;
  src_data[1] = 3.0f;
  src_data[2] = 1.0f;
  src_data[3] = 4.0f;
  src_data[4] = 2.0f;
  src_data[5] = 5.0f;

  SlimTensor src(
      std::move(src_storage),
      makeArrayRef(src_sizes),
      makeArrayRef(src_strides),
      c10::ScalarType::Float);

  // Destination is contiguous
  std::vector<int64_t> dst_sizes = {2, 3};
  std::vector<int64_t> dst_strides = {3, 1};
  SlimTensor dst = make_filled_tensor(
      dst_sizes, dst_strides, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});

  dst.copy_(src);

  float* dst_data = static_cast<float*>(dst.data_ptr());
  EXPECT_FLOAT_EQ(dst_data[0], 0.0f);
  EXPECT_FLOAT_EQ(dst_data[1], 1.0f);
  EXPECT_FLOAT_EQ(dst_data[2], 2.0f);
  EXPECT_FLOAT_EQ(dst_data[3], 3.0f);
  EXPECT_FLOAT_EQ(dst_data[4], 4.0f);
  EXPECT_FLOAT_EQ(dst_data[5], 5.0f);
}

TEST(SlimTensorCopyTest, CopyNonContiguousDst) {
  // Source is contiguous
  std::vector<int64_t> src_sizes = {2, 3};
  std::vector<int64_t> src_strides = {3, 1};
  std::vector<float> values = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  SlimTensor src = make_filled_tensor(src_sizes, src_strides, values);

  // Destination is transposed (non-contiguous)
  std::vector<int64_t> dst_sizes = {2, 3};
  std::vector<int64_t> dst_strides = {1, 2};
  Storage dst_storage = make_cpu_storage(6 * sizeof(float));

  SlimTensor dst(
      std::move(dst_storage),
      makeArrayRef(dst_sizes),
      makeArrayRef(dst_strides),
      c10::ScalarType::Float);

  dst.copy_(src);

  float* dst_data = static_cast<float*>(dst.storage()->data());
  // After copy, physical layout should be: [0,3] [1,4] [2,5]
  EXPECT_FLOAT_EQ(dst_data[0], 0.0f);
  EXPECT_FLOAT_EQ(dst_data[1], 3.0f);
  EXPECT_FLOAT_EQ(dst_data[2], 1.0f);
  EXPECT_FLOAT_EQ(dst_data[3], 4.0f);
  EXPECT_FLOAT_EQ(dst_data[4], 2.0f);
  EXPECT_FLOAT_EQ(dst_data[5], 5.0f);
}

// =============================================================================
// Storage Offset Tests
// =============================================================================

TEST(SlimTensorCopyTest, CopyWithStorageOffset) {
  // Create a larger storage and use offset
  std::vector<int64_t> sizes = {2, 2};
  std::vector<int64_t> strides = {2, 1};
  size_t total_nbytes = 100 * sizeof(float);

  // Source with offset
  Storage src_storage = make_cpu_storage(total_nbytes);
  float* src_base = static_cast<float*>(src_storage->data());
  src_base[10] = 1.0f;
  src_base[11] = 2.0f;
  src_base[12] = 3.0f;
  src_base[13] = 4.0f;

  SlimTensor src(
      std::move(src_storage),
      makeArrayRef(sizes),
      makeArrayRef(strides),
      c10::ScalarType::Float,
      10);

  // Destination with different offset
  Storage dst_storage = make_cpu_storage(total_nbytes);
  SlimTensor dst(
      std::move(dst_storage),
      makeArrayRef(sizes),
      makeArrayRef(strides),
      c10::ScalarType::Float,
      20);

  dst.copy_(src);

  float* dst_base = static_cast<float*>(dst.storage()->data());
  EXPECT_FLOAT_EQ(dst_base[20], 1.0f);
  EXPECT_FLOAT_EQ(dst_base[21], 2.0f);
  EXPECT_FLOAT_EQ(dst_base[22], 3.0f);
  EXPECT_FLOAT_EQ(dst_base[23], 4.0f);
}

} // namespace executorch::backends::aoti::slim
