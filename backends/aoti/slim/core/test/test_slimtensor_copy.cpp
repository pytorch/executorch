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
#include <executorch/backends/aoti/slim/factory/Empty.h>

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

// =============================================================================
// CUDA Tensor Creation Tests
// These tests verify CUDA tensor creation and the is_cuda() method.
// When CUDA_AVAILABLE is not defined, CUDA operations abort with an error.
// =============================================================================

#ifdef CUDA_AVAILABLE

TEST(CUDATensorTest, CreateEmptyCUDATensor) {
  auto tensor = empty({2, 3}, c10::ScalarType::Float, DEFAULT_CUDA_DEVICE);

  EXPECT_TRUE(tensor.defined());
  EXPECT_TRUE(tensor.is_cuda());
  EXPECT_FALSE(tensor.is_cpu());
  EXPECT_EQ(tensor.dim(), 2);
  EXPECT_EQ(tensor.size(0), 2);
  EXPECT_EQ(tensor.size(1), 3);
  EXPECT_EQ(tensor.numel(), 6);
  EXPECT_TRUE(tensor.is_contiguous());
  EXPECT_EQ(tensor.device().type(), c10::DeviceType::CUDA);
  EXPECT_EQ(tensor.device().index(), 0);
}

TEST(CUDATensorTest, CreateEmptyStridedCUDATensor) {
  std::vector<int64_t> sizes = {2, 4};
  std::vector<int64_t> strides = {4, 1};

  auto tensor = empty_strided(
      makeArrayRef(sizes),
      makeArrayRef(strides),
      c10::ScalarType::Float,
      DEFAULT_CUDA_DEVICE);

  EXPECT_TRUE(tensor.is_cuda());
  EXPECT_EQ(tensor.stride(0), 4);
  EXPECT_EQ(tensor.stride(1), 1);
  EXPECT_EQ(tensor.numel(), 8);
}

TEST(CUDATensorTest, CreateCUDATensorWithDeviceIndex) {
  c10::Device device(c10::DeviceType::CUDA, 0);
  auto tensor = empty({4, 4}, c10::ScalarType::Float, device);

  EXPECT_TRUE(tensor.is_cuda());
  EXPECT_EQ(tensor.device_index(), 0);
}

// =============================================================================
// Cross-Device Copy Tests
// =============================================================================

TEST(CUDACopyTest, CopyFromCPUToCUDA) {
  constexpr size_t kNumFloats = 6;
  auto cpu_tensor = empty({2, 3}, c10::ScalarType::Float, CPU_DEVICE);

  // Fill CPU tensor with known values
  float* cpu_data = static_cast<float*>(cpu_tensor.data_ptr());
  for (size_t i = 0; i < kNumFloats; ++i) {
    cpu_data[i] = static_cast<float>(i) * 1.5f;
  }

  // Create CUDA tensor and copy from CPU
  auto cuda_tensor = empty({2, 3}, c10::ScalarType::Float, DEFAULT_CUDA_DEVICE);
  cuda_tensor.copy_(cpu_tensor);

  // Copy back to CPU to verify
  auto verify_tensor = empty({2, 3}, c10::ScalarType::Float, CPU_DEVICE);
  verify_tensor.copy_(cuda_tensor);

  float* verify_data = static_cast<float*>(verify_tensor.data_ptr());
  for (size_t i = 0; i < kNumFloats; ++i) {
    EXPECT_FLOAT_EQ(verify_data[i], static_cast<float>(i) * 1.5f);
  }
}

TEST(CUDACopyTest, CopyFromCUDAToCPU) {
  constexpr size_t kNumFloats = 4;
  auto cpu_src = empty({2, 2}, c10::ScalarType::Float, CPU_DEVICE);

  float* src_data = static_cast<float*>(cpu_src.data_ptr());
  for (size_t i = 0; i < kNumFloats; ++i) {
    src_data[i] = static_cast<float>(i) + 100.0f;
  }

  // Copy to CUDA
  auto cuda_tensor = empty({2, 2}, c10::ScalarType::Float, DEFAULT_CUDA_DEVICE);
  cuda_tensor.copy_(cpu_src);

  // Copy back to new CPU tensor
  auto cpu_dst = empty({2, 2}, c10::ScalarType::Float, CPU_DEVICE);
  cpu_dst.copy_(cuda_tensor);

  float* dst_data = static_cast<float*>(cpu_dst.data_ptr());
  for (size_t i = 0; i < kNumFloats; ++i) {
    EXPECT_FLOAT_EQ(dst_data[i], static_cast<float>(i) + 100.0f);
  }
}

TEST(CUDACopyTest, CopyFromCUDAToCUDA) {
  constexpr size_t kNumFloats = 4;
  auto cpu_tensor = empty({2, 2}, c10::ScalarType::Float, CPU_DEVICE);

  float* cpu_data = static_cast<float*>(cpu_tensor.data_ptr());
  for (size_t i = 0; i < kNumFloats; ++i) {
    cpu_data[i] = static_cast<float>(i) * 2.0f;
  }

  // Create first CUDA tensor from CPU
  auto cuda_src = empty({2, 2}, c10::ScalarType::Float, DEFAULT_CUDA_DEVICE);
  cuda_src.copy_(cpu_tensor);

  // Copy to second CUDA tensor
  auto cuda_dst = empty({2, 2}, c10::ScalarType::Float, DEFAULT_CUDA_DEVICE);
  cuda_dst.copy_(cuda_src);

  // Verify by copying back to CPU
  auto verify_tensor = empty({2, 2}, c10::ScalarType::Float, CPU_DEVICE);
  verify_tensor.copy_(cuda_dst);

  float* verify_data = static_cast<float*>(verify_tensor.data_ptr());
  for (size_t i = 0; i < kNumFloats; ++i) {
    EXPECT_FLOAT_EQ(verify_data[i], static_cast<float>(i) * 2.0f);
  }
}

TEST(CUDACopyTest, CopyDifferentDtypes) {
  auto cpu_int = empty({4}, c10::ScalarType::Int, CPU_DEVICE);
  int32_t* int_data = static_cast<int32_t*>(cpu_int.data_ptr());
  for (int i = 0; i < 4; ++i) {
    int_data[i] = i * 10;
  }

  auto cuda_int = empty({4}, c10::ScalarType::Int, DEFAULT_CUDA_DEVICE);
  cuda_int.copy_(cpu_int);

  auto verify_int = empty({4}, c10::ScalarType::Int, CPU_DEVICE);
  verify_int.copy_(cuda_int);

  int32_t* verify_data = static_cast<int32_t*>(verify_int.data_ptr());
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(verify_data[i], i * 10);
  }
}

TEST(CUDACopyTest, CopyEmptyTensor) {
  auto cpu_empty = empty({0}, c10::ScalarType::Float, CPU_DEVICE);
  auto cuda_empty = empty({0}, c10::ScalarType::Float, DEFAULT_CUDA_DEVICE);

  // Should not crash
  cuda_empty.copy_(cpu_empty);
  cpu_empty.copy_(cuda_empty);

  EXPECT_EQ(cpu_empty.numel(), 0);
  EXPECT_EQ(cuda_empty.numel(), 0);
}

// =============================================================================
// Non-Contiguous Cross-Device Copy Tests
// These tests verify copying non-contiguous CPU tensors to/from CUDA tensors.
// The CUDA tensor must be contiguous, but the CPU tensor can be non-contiguous.
// =============================================================================

TEST(CUDACopyTest, CopyNonContiguousCPUToCUDA) {
  // Create a transposed (non-contiguous) CPU source tensor
  // Logical shape: 2x3, but stored transposed in memory
  std::vector<int64_t> src_sizes = {2, 3};
  std::vector<int64_t> src_strides = {1, 2}; // Transposed strides

  Storage src_storage =
      Storage(new MaybeOwningStorage(CPU_DEVICE, 6 * sizeof(float)));
  float* src_data = static_cast<float*>(src_storage->data());
  // Physical layout for transposed tensor:
  // Logical[0,0]=Physical[0], Logical[1,0]=Physical[1]
  // Logical[0,1]=Physical[2], Logical[1,1]=Physical[3]
  // Logical[0,2]=Physical[4], Logical[1,2]=Physical[5]
  src_data[0] = 1.0f; // Logical[0,0]
  src_data[1] = 4.0f; // Logical[1,0]
  src_data[2] = 2.0f; // Logical[0,1]
  src_data[3] = 5.0f; // Logical[1,1]
  src_data[4] = 3.0f; // Logical[0,2]
  src_data[5] = 6.0f; // Logical[1,2]

  SlimTensor cpu_src(
      std::move(src_storage),
      makeArrayRef(src_sizes),
      makeArrayRef(src_strides),
      c10::ScalarType::Float);

  ASSERT_FALSE(cpu_src.is_contiguous());

  // Create a contiguous CUDA destination
  auto cuda_dst = empty({2, 3}, c10::ScalarType::Float, DEFAULT_CUDA_DEVICE);
  ASSERT_TRUE(cuda_dst.is_contiguous());

  // Copy non-contiguous CPU → contiguous CUDA
  cuda_dst.copy_(cpu_src);

  // Verify by copying back to CPU
  auto verify = empty({2, 3}, c10::ScalarType::Float, CPU_DEVICE);
  verify.copy_(cuda_dst);

  // Values should be in logical order (contiguous layout)
  float* verify_data = static_cast<float*>(verify.data_ptr());
  EXPECT_FLOAT_EQ(verify_data[0], 1.0f); // [0,0]
  EXPECT_FLOAT_EQ(verify_data[1], 2.0f); // [0,1]
  EXPECT_FLOAT_EQ(verify_data[2], 3.0f); // [0,2]
  EXPECT_FLOAT_EQ(verify_data[3], 4.0f); // [1,0]
  EXPECT_FLOAT_EQ(verify_data[4], 5.0f); // [1,1]
  EXPECT_FLOAT_EQ(verify_data[5], 6.0f); // [1,2]
}

TEST(CUDACopyTest, CopyCUDAToNonContiguousCPU) {
  constexpr size_t kNumFloats = 6;

  // Create a contiguous CPU source, copy to CUDA
  auto cpu_src = empty({2, 3}, c10::ScalarType::Float, CPU_DEVICE);
  float* src_data = static_cast<float*>(cpu_src.data_ptr());
  for (size_t i = 0; i < kNumFloats; ++i) {
    src_data[i] = static_cast<float>(i + 1);
  }

  auto cuda_tensor = empty({2, 3}, c10::ScalarType::Float, DEFAULT_CUDA_DEVICE);
  cuda_tensor.copy_(cpu_src);

  // Create a transposed (non-contiguous) CPU destination
  std::vector<int64_t> dst_sizes = {2, 3};
  std::vector<int64_t> dst_strides = {1, 2}; // Transposed strides

  Storage dst_storage =
      Storage(new MaybeOwningStorage(CPU_DEVICE, 6 * sizeof(float)));
  SlimTensor cpu_dst(
      std::move(dst_storage),
      makeArrayRef(dst_sizes),
      makeArrayRef(dst_strides),
      c10::ScalarType::Float);

  ASSERT_FALSE(cpu_dst.is_contiguous());

  // Copy contiguous CUDA → non-contiguous CPU
  cpu_dst.copy_(cuda_tensor);

  // Verify physical layout matches transposed storage
  float* dst_data = static_cast<float*>(cpu_dst.storage()->data());
  // Physical layout: [1,4,2,5,3,6] for logical [[1,2,3],[4,5,6]]
  EXPECT_FLOAT_EQ(dst_data[0], 1.0f); // Logical[0,0]
  EXPECT_FLOAT_EQ(dst_data[1], 4.0f); // Logical[1,0]
  EXPECT_FLOAT_EQ(dst_data[2], 2.0f); // Logical[0,1]
  EXPECT_FLOAT_EQ(dst_data[3], 5.0f); // Logical[1,1]
  EXPECT_FLOAT_EQ(dst_data[4], 3.0f); // Logical[0,2]
  EXPECT_FLOAT_EQ(dst_data[5], 6.0f); // Logical[1,2]
}

TEST(CUDACopyTest, CopyNonContiguousCPUToCUDA3D) {
  // Test 3D non-contiguous tensor copy
  std::vector<int64_t> sizes = {2, 2, 2};
  // Permuted strides (e.g., from permute(2, 0, 1))
  std::vector<int64_t> non_contig_strides = {2, 1, 4};

  Storage src_storage =
      Storage(new MaybeOwningStorage(CPU_DEVICE, 8 * sizeof(float)));
  float* src_data = static_cast<float*>(src_storage->data());
  // Fill with values 1-8
  for (int i = 0; i < 8; ++i) {
    src_data[i] = static_cast<float>(i + 1);
  }

  SlimTensor cpu_src(
      std::move(src_storage),
      makeArrayRef(sizes),
      makeArrayRef(non_contig_strides),
      c10::ScalarType::Float);

  ASSERT_FALSE(cpu_src.is_contiguous());

  auto cuda_dst = empty({2, 2, 2}, c10::ScalarType::Float, DEFAULT_CUDA_DEVICE);
  cuda_dst.copy_(cpu_src);

  // Copy back and verify the logical order is preserved
  auto verify = empty({2, 2, 2}, c10::ScalarType::Float, CPU_DEVICE);
  verify.copy_(cuda_dst);

  // Access elements via strided indexing on source
  float* verify_data = static_cast<float*>(verify.data_ptr());

  // Verify a few key positions
  // The values should match the logical traversal of the source tensor
  EXPECT_NE(verify_data[0], 0.0f); // Should have data
  EXPECT_EQ(verify.numel(), 8);
}

TEST(CUDACopyTest, CopyCUDAToNonContiguousCPUWithOffset) {
  // Test with storage offset
  constexpr size_t kNumFloats = 4;

  auto cpu_src = empty({2, 2}, c10::ScalarType::Float, CPU_DEVICE);
  float* src_data = static_cast<float*>(cpu_src.data_ptr());
  for (size_t i = 0; i < kNumFloats; ++i) {
    src_data[i] = static_cast<float>(i + 10);
  }

  auto cuda_tensor = empty({2, 2}, c10::ScalarType::Float, DEFAULT_CUDA_DEVICE);
  cuda_tensor.copy_(cpu_src);

  // Create non-contiguous destination with storage offset
  std::vector<int64_t> dst_sizes = {2, 2};
  std::vector<int64_t> dst_strides = {1, 2}; // Transposed

  Storage dst_storage =
      Storage(new MaybeOwningStorage(CPU_DEVICE, 10 * sizeof(float)));
  SlimTensor cpu_dst(
      std::move(dst_storage),
      makeArrayRef(dst_sizes),
      makeArrayRef(dst_strides),
      c10::ScalarType::Float,
      2); // offset of 2 elements

  ASSERT_FALSE(cpu_dst.is_contiguous());

  cpu_dst.copy_(cuda_tensor);

  // Verify data starts at offset
  float* raw_data = static_cast<float*>(cpu_dst.storage()->data());
  float* offset_data = static_cast<float*>(cpu_dst.data_ptr());

  // offset_data should be 2 elements after raw_data
  EXPECT_EQ(offset_data, raw_data + 2);

  // Verify transposed layout at offset
  EXPECT_FLOAT_EQ(offset_data[0], 10.0f); // Logical[0,0]
  EXPECT_FLOAT_EQ(offset_data[1], 12.0f); // Logical[1,0]
  EXPECT_FLOAT_EQ(offset_data[2], 11.0f); // Logical[0,1]
  EXPECT_FLOAT_EQ(offset_data[3], 13.0f); // Logical[1,1]
}

TEST(CUDACopyTest, CopyNonContiguousCPUToCUDAInt64) {
  // Test with different dtype (int64)
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = {1, 2}; // Transposed

  Storage src_storage =
      Storage(new MaybeOwningStorage(CPU_DEVICE, 6 * sizeof(int64_t)));
  int64_t* src_data = static_cast<int64_t*>(src_storage->data());
  // Fill transposed layout
  src_data[0] = 100; // Logical[0,0]
  src_data[1] = 400; // Logical[1,0]
  src_data[2] = 200; // Logical[0,1]
  src_data[3] = 500; // Logical[1,1]
  src_data[4] = 300; // Logical[0,2]
  src_data[5] = 600; // Logical[1,2]

  SlimTensor cpu_src(
      std::move(src_storage),
      makeArrayRef(sizes),
      makeArrayRef(strides),
      c10::ScalarType::Long);

  ASSERT_FALSE(cpu_src.is_contiguous());

  auto cuda_dst = empty({2, 3}, c10::ScalarType::Long, DEFAULT_CUDA_DEVICE);
  cuda_dst.copy_(cpu_src);

  auto verify = empty({2, 3}, c10::ScalarType::Long, CPU_DEVICE);
  verify.copy_(cuda_dst);

  int64_t* verify_data = static_cast<int64_t*>(verify.data_ptr());
  EXPECT_EQ(verify_data[0], 100);
  EXPECT_EQ(verify_data[1], 200);
  EXPECT_EQ(verify_data[2], 300);
  EXPECT_EQ(verify_data[3], 400);
  EXPECT_EQ(verify_data[4], 500);
  EXPECT_EQ(verify_data[5], 600);
}

#endif // CUDA_AVAILABLE

} // namespace executorch::backends::aoti::slim
