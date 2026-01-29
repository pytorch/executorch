/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/aoti/slim/factory/Empty.h>

#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

namespace executorch::backends::aoti::slim {

// =============================================================================
// empty_strided() Tests
// =============================================================================

TEST(EmptyStridedTest, Basic2x3Tensor) {
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = {3, 1};

  SlimTensor tensor = empty_strided(
      makeArrayRef(sizes), makeArrayRef(strides), c10::ScalarType::Float);

  EXPECT_TRUE(tensor.defined());
  EXPECT_EQ(tensor.dim(), 2u);
  EXPECT_EQ(tensor.numel(), 6u);
  EXPECT_EQ(tensor.dtype(), c10::ScalarType::Float);
  EXPECT_TRUE(tensor.is_cpu());

  auto result_sizes = tensor.sizes();
  EXPECT_EQ(result_sizes[0], 2);
  EXPECT_EQ(result_sizes[1], 3);

  auto result_strides = tensor.strides();
  EXPECT_EQ(result_strides[0], 3);
  EXPECT_EQ(result_strides[1], 1);
}

TEST(EmptyStridedTest, ContiguousTensor) {
  std::vector<int64_t> sizes = {2, 3, 4};
  std::vector<int64_t> strides = {12, 4, 1};

  SlimTensor tensor = empty_strided(
      makeArrayRef(sizes), makeArrayRef(strides), c10::ScalarType::Float);

  EXPECT_TRUE(tensor.is_contiguous());
  EXPECT_EQ(tensor.numel(), 24u);
  EXPECT_EQ(tensor.nbytes(), 24 * sizeof(float));
}

TEST(EmptyStridedTest, NonContiguousTensor) {
  std::vector<int64_t> sizes = {3, 2};
  std::vector<int64_t> strides = {1, 3};

  SlimTensor tensor = empty_strided(
      makeArrayRef(sizes), makeArrayRef(strides), c10::ScalarType::Float);

  EXPECT_FALSE(tensor.is_contiguous());
  EXPECT_EQ(tensor.numel(), 6u);
}

TEST(EmptyStridedTest, OneDimensional) {
  std::vector<int64_t> sizes = {10};
  std::vector<int64_t> strides = {1};

  SlimTensor tensor = empty_strided(
      makeArrayRef(sizes), makeArrayRef(strides), c10::ScalarType::Float);

  EXPECT_EQ(tensor.dim(), 1u);
  EXPECT_EQ(tensor.numel(), 10u);
  EXPECT_TRUE(tensor.is_contiguous());
}

TEST(EmptyStridedTest, ZeroSizedTensor) {
  std::vector<int64_t> sizes = {0, 3};
  std::vector<int64_t> strides = {3, 1};

  SlimTensor tensor = empty_strided(
      makeArrayRef(sizes), makeArrayRef(strides), c10::ScalarType::Float);

  EXPECT_TRUE(tensor.defined());
  EXPECT_EQ(tensor.numel(), 0u);
  EXPECT_TRUE(tensor.is_empty());
}

TEST(EmptyStridedTest, LargeDimensionalTensor) {
  std::vector<int64_t> sizes = {2, 3, 4, 5};
  std::vector<int64_t> strides = {60, 20, 5, 1};

  SlimTensor tensor = empty_strided(
      makeArrayRef(sizes), makeArrayRef(strides), c10::ScalarType::Float);

  EXPECT_EQ(tensor.dim(), 4u);
  EXPECT_EQ(tensor.numel(), 120u);
  EXPECT_TRUE(tensor.is_contiguous());
}

TEST(EmptyStridedTest, StorageOffset) {
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = {3, 1};

  SlimTensor tensor = empty_strided(
      makeArrayRef(sizes), makeArrayRef(strides), c10::ScalarType::Float);

  EXPECT_EQ(tensor.storage_offset(), 0);
}

// =============================================================================
// empty() Tests
// =============================================================================

TEST(EmptyTest, BasicWithArrayRef) {
  std::vector<int64_t> sizes = {2, 3, 4};

  SlimTensor tensor = empty(makeArrayRef(sizes), c10::ScalarType::Float);

  EXPECT_TRUE(tensor.defined());
  EXPECT_EQ(tensor.dim(), 3u);
  EXPECT_EQ(tensor.numel(), 24u);
  EXPECT_TRUE(tensor.is_contiguous());
}

TEST(EmptyTest, VerifiesContiguousStrides) {
  std::vector<int64_t> sizes = {2, 3, 4};

  SlimTensor tensor = empty(makeArrayRef(sizes), c10::ScalarType::Float);

  auto strides = tensor.strides();
  EXPECT_EQ(strides[0], 12);
  EXPECT_EQ(strides[1], 4);
  EXPECT_EQ(strides[2], 1);
}

TEST(EmptyTest, InitializerListOverload) {
  SlimTensor tensor = empty({4, 5, 6}, c10::ScalarType::Float);

  EXPECT_EQ(tensor.dim(), 3u);
  EXPECT_EQ(tensor.numel(), 120u);
  EXPECT_TRUE(tensor.is_contiguous());

  auto sizes = tensor.sizes();
  EXPECT_EQ(sizes[0], 4);
  EXPECT_EQ(sizes[1], 5);
  EXPECT_EQ(sizes[2], 6);
}

TEST(EmptyTest, OneDimensional) {
  SlimTensor tensor = empty({10}, c10::ScalarType::Float);

  EXPECT_EQ(tensor.dim(), 1u);
  EXPECT_EQ(tensor.numel(), 10u);
  EXPECT_EQ(tensor.stride(0), 1);
}

TEST(EmptyTest, ZeroSized) {
  SlimTensor tensor = empty({0, 5}, c10::ScalarType::Float);

  EXPECT_TRUE(tensor.is_empty());
  EXPECT_EQ(tensor.numel(), 0u);
}

// =============================================================================
// empty_like() Tests
// =============================================================================

TEST(EmptyLikeTest, CopiesMetadata) {
  std::vector<int64_t> sizes = {2, 3, 4};
  std::vector<int64_t> strides = {12, 4, 1};

  SlimTensor original = empty_strided(
      makeArrayRef(sizes), makeArrayRef(strides), c10::ScalarType::Float);
  SlimTensor copy = empty_like(original);

  EXPECT_EQ(copy.dim(), original.dim());
  EXPECT_EQ(copy.numel(), original.numel());
  EXPECT_EQ(copy.dtype(), original.dtype());
  EXPECT_EQ(copy.is_cpu(), original.is_cpu());
  EXPECT_EQ(copy.is_contiguous(), original.is_contiguous());

  for (size_t i = 0; i < copy.dim(); i++) {
    EXPECT_EQ(copy.size(i), original.size(i));
    EXPECT_EQ(copy.stride(i), original.stride(i));
  }
}

TEST(EmptyLikeTest, HasDifferentStorage) {
  SlimTensor original = empty({2, 3}, c10::ScalarType::Float);
  SlimTensor copy = empty_like(original);

  EXPECT_NE(original.data_ptr(), copy.data_ptr());
}

TEST(EmptyLikeTest, NonContiguousTensor) {
  std::vector<int64_t> sizes = {3, 2};
  std::vector<int64_t> strides = {1, 3};

  SlimTensor original = empty_strided(
      makeArrayRef(sizes), makeArrayRef(strides), c10::ScalarType::Float);
  SlimTensor copy = empty_like(original);

  EXPECT_FALSE(copy.is_contiguous());
  EXPECT_EQ(copy.stride(0), 1);
  EXPECT_EQ(copy.stride(1), 3);
}

// =============================================================================
// Data Access Tests
// =============================================================================

TEST(EmptyTest, DataPtrIsValid) {
  SlimTensor tensor = empty({2, 3}, c10::ScalarType::Float);

  void* data = tensor.data_ptr();
  EXPECT_NE(data, nullptr);
}

TEST(EmptyTest, CanWriteAndReadData) {
  SlimTensor tensor = empty({2, 3}, c10::ScalarType::Float);

  float* data = static_cast<float*>(tensor.data_ptr());
  for (size_t i = 0; i < tensor.numel(); i++) {
    data[i] = static_cast<float>(i);
  }

  for (size_t i = 0; i < tensor.numel(); i++) {
    EXPECT_EQ(data[i], static_cast<float>(i));
  }
}

#ifdef CUDA_AVAILABLE

// =============================================================================
// CUDA Empty Tensor Tests
// Tests are skipped at runtime if CUDA hardware is not available.
// =============================================================================

// =============================================================================
// empty_strided() CUDA Tests
// =============================================================================

TEST(EmptyStridedCUDATest, Basic2x3Tensor) {
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = {3, 1};

  SlimTensor tensor = empty_strided(
      makeArrayRef(sizes),
      makeArrayRef(strides),
      c10::ScalarType::Float,
      DEFAULT_CUDA_DEVICE);

  EXPECT_TRUE(tensor.defined());
  EXPECT_EQ(tensor.dim(), 2u);
  EXPECT_EQ(tensor.numel(), 6u);
  EXPECT_EQ(tensor.dtype(), c10::ScalarType::Float);
  EXPECT_TRUE(tensor.is_cuda());
  EXPECT_FALSE(tensor.is_cpu());

  auto result_sizes = tensor.sizes();
  EXPECT_EQ(result_sizes[0], 2);
  EXPECT_EQ(result_sizes[1], 3);

  auto result_strides = tensor.strides();
  EXPECT_EQ(result_strides[0], 3);
  EXPECT_EQ(result_strides[1], 1);
}

TEST(EmptyStridedCUDATest, ContiguousTensor) {
  std::vector<int64_t> sizes = {2, 3, 4};
  std::vector<int64_t> strides = {12, 4, 1};

  SlimTensor tensor = empty_strided(
      makeArrayRef(sizes),
      makeArrayRef(strides),
      c10::ScalarType::Float,
      DEFAULT_CUDA_DEVICE);

  EXPECT_TRUE(tensor.is_contiguous());
  EXPECT_EQ(tensor.numel(), 24u);
  EXPECT_EQ(tensor.nbytes(), 24 * sizeof(float));
  EXPECT_TRUE(tensor.is_cuda());
}

TEST(EmptyStridedCUDATest, NonContiguousTensor) {
  std::vector<int64_t> sizes = {3, 2};
  std::vector<int64_t> strides = {1, 3};

  SlimTensor tensor = empty_strided(
      makeArrayRef(sizes),
      makeArrayRef(strides),
      c10::ScalarType::Float,
      DEFAULT_CUDA_DEVICE);

  EXPECT_FALSE(tensor.is_contiguous());
  EXPECT_EQ(tensor.numel(), 6u);
  EXPECT_TRUE(tensor.is_cuda());
}

TEST(EmptyStridedCUDATest, OneDimensional) {
  std::vector<int64_t> sizes = {10};
  std::vector<int64_t> strides = {1};

  SlimTensor tensor = empty_strided(
      makeArrayRef(sizes),
      makeArrayRef(strides),
      c10::ScalarType::Float,
      DEFAULT_CUDA_DEVICE);

  EXPECT_EQ(tensor.dim(), 1u);
  EXPECT_EQ(tensor.numel(), 10u);
  EXPECT_TRUE(tensor.is_contiguous());
  EXPECT_TRUE(tensor.is_cuda());
}

TEST(EmptyStridedCUDATest, ZeroSizedTensor) {
  std::vector<int64_t> sizes = {0, 3};
  std::vector<int64_t> strides = {3, 1};

  SlimTensor tensor = empty_strided(
      makeArrayRef(sizes),
      makeArrayRef(strides),
      c10::ScalarType::Float,
      DEFAULT_CUDA_DEVICE);

  EXPECT_TRUE(tensor.defined());
  EXPECT_EQ(tensor.numel(), 0u);
  EXPECT_TRUE(tensor.is_empty());
  EXPECT_TRUE(tensor.is_cuda());
}

TEST(EmptyStridedCUDATest, LargeDimensionalTensor) {
  std::vector<int64_t> sizes = {2, 3, 4, 5};
  std::vector<int64_t> strides = {60, 20, 5, 1};

  SlimTensor tensor = empty_strided(
      makeArrayRef(sizes),
      makeArrayRef(strides),
      c10::ScalarType::Float,
      DEFAULT_CUDA_DEVICE);

  EXPECT_EQ(tensor.dim(), 4u);
  EXPECT_EQ(tensor.numel(), 120u);
  EXPECT_TRUE(tensor.is_contiguous());
  EXPECT_TRUE(tensor.is_cuda());
}

// =============================================================================
// empty() CUDA Tests
// =============================================================================

TEST(EmptyCUDATest, BasicWithArrayRef) {
  std::vector<int64_t> sizes = {2, 3, 4};

  SlimTensor tensor =
      empty(makeArrayRef(sizes), c10::ScalarType::Float, DEFAULT_CUDA_DEVICE);

  EXPECT_TRUE(tensor.defined());
  EXPECT_EQ(tensor.dim(), 3u);
  EXPECT_EQ(tensor.numel(), 24u);
  EXPECT_TRUE(tensor.is_contiguous());
  EXPECT_TRUE(tensor.is_cuda());
}

TEST(EmptyCUDATest, VerifiesContiguousStrides) {
  std::vector<int64_t> sizes = {2, 3, 4};

  SlimTensor tensor =
      empty(makeArrayRef(sizes), c10::ScalarType::Float, DEFAULT_CUDA_DEVICE);

  auto strides = tensor.strides();
  EXPECT_EQ(strides[0], 12);
  EXPECT_EQ(strides[1], 4);
  EXPECT_EQ(strides[2], 1);
  EXPECT_TRUE(tensor.is_cuda());
}

TEST(EmptyCUDATest, InitializerListOverload) {
  SlimTensor tensor =
      empty({4, 5, 6}, c10::ScalarType::Float, DEFAULT_CUDA_DEVICE);

  EXPECT_EQ(tensor.dim(), 3u);
  EXPECT_EQ(tensor.numel(), 120u);
  EXPECT_TRUE(tensor.is_contiguous());
  EXPECT_TRUE(tensor.is_cuda());

  auto sizes = tensor.sizes();
  EXPECT_EQ(sizes[0], 4);
  EXPECT_EQ(sizes[1], 5);
  EXPECT_EQ(sizes[2], 6);
}

TEST(EmptyCUDATest, OneDimensional) {
  SlimTensor tensor = empty({10}, c10::ScalarType::Float, DEFAULT_CUDA_DEVICE);

  EXPECT_EQ(tensor.dim(), 1u);
  EXPECT_EQ(tensor.numel(), 10u);
  EXPECT_EQ(tensor.stride(0), 1);
  EXPECT_TRUE(tensor.is_cuda());
}

TEST(EmptyCUDATest, ZeroSized) {
  SlimTensor tensor =
      empty({0, 5}, c10::ScalarType::Float, DEFAULT_CUDA_DEVICE);

  EXPECT_TRUE(tensor.is_empty());
  EXPECT_EQ(tensor.numel(), 0u);
  EXPECT_TRUE(tensor.is_cuda());
}

// =============================================================================
// empty_like() CUDA Tests
// =============================================================================

TEST(EmptyLikeCUDATest, CopiesMetadata) {
  std::vector<int64_t> sizes = {2, 3, 4};
  std::vector<int64_t> strides = {12, 4, 1};

  SlimTensor original = empty_strided(
      makeArrayRef(sizes),
      makeArrayRef(strides),
      c10::ScalarType::Float,
      DEFAULT_CUDA_DEVICE);
  SlimTensor copy = empty_like(original);

  EXPECT_EQ(copy.dim(), original.dim());
  EXPECT_EQ(copy.numel(), original.numel());
  EXPECT_EQ(copy.dtype(), original.dtype());
  EXPECT_EQ(copy.is_cuda(), original.is_cuda());
  EXPECT_EQ(copy.is_contiguous(), original.is_contiguous());

  for (size_t i = 0; i < copy.dim(); i++) {
    EXPECT_EQ(copy.size(i), original.size(i));
    EXPECT_EQ(copy.stride(i), original.stride(i));
  }
}

TEST(EmptyLikeCUDATest, HasDifferentStorage) {
  SlimTensor original =
      empty({2, 3}, c10::ScalarType::Float, DEFAULT_CUDA_DEVICE);
  SlimTensor copy = empty_like(original);

  EXPECT_NE(original.data_ptr(), copy.data_ptr());
  EXPECT_TRUE(copy.is_cuda());
}

TEST(EmptyLikeCUDATest, NonContiguousTensor) {
  std::vector<int64_t> sizes = {3, 2};
  std::vector<int64_t> strides = {1, 3};

  SlimTensor original = empty_strided(
      makeArrayRef(sizes),
      makeArrayRef(strides),
      c10::ScalarType::Float,
      DEFAULT_CUDA_DEVICE);
  SlimTensor copy = empty_like(original);

  EXPECT_FALSE(copy.is_contiguous());
  EXPECT_EQ(copy.stride(0), 1);
  EXPECT_EQ(copy.stride(1), 3);
  EXPECT_TRUE(copy.is_cuda());
}

// =============================================================================
// CUDA Data Access Tests
// =============================================================================

TEST(EmptyCUDATest, DataPtrIsValid) {
  SlimTensor tensor =
      empty({2, 3}, c10::ScalarType::Float, DEFAULT_CUDA_DEVICE);

  void* data = tensor.data_ptr();
  EXPECT_NE(data, nullptr);
}

TEST(EmptyCUDATest, DeviceIndex) {
  SlimTensor tensor =
      empty({2, 3}, c10::ScalarType::Float, DEFAULT_CUDA_DEVICE);

  EXPECT_EQ(tensor.device().index(), 0);
}

#endif // CUDA_AVAILABLE

} // namespace executorch::backends::aoti::slim
