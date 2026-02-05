/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/aoti/slim/core/storage.h>
#include <executorch/backends/aoti/slim/factory/empty.h>
#include <executorch/backends/aoti/slim/factory/from_blob.h>

#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

namespace executorch::backends::aoti::slim {

// =============================================================================
// from_blob Basic Tests
// =============================================================================

TEST(FromBlobTest, BasicConstruction) {
  constexpr size_t kNumFloats = 24;
  float external_data[kNumFloats];

  // Initialize external data
  for (size_t i = 0; i < kNumFloats; ++i) {
    external_data[i] = static_cast<float>(i) * 1.5f;
  }

  SlimTensor tensor =
      from_blob(external_data, {2, 3, 4}, c10::ScalarType::Float);

  // Verify tensor properties
  EXPECT_EQ(tensor.numel(), kNumFloats);
  EXPECT_EQ(tensor.dim(), 3);
  EXPECT_EQ(tensor.size(0), 2);
  EXPECT_EQ(tensor.size(1), 3);
  EXPECT_EQ(tensor.size(2), 4);
  EXPECT_EQ(tensor.dtype(), c10::ScalarType::Float);
  EXPECT_TRUE(tensor.is_cpu());
  EXPECT_TRUE(tensor.is_contiguous());
  EXPECT_EQ(tensor.storage_offset(), 0);

  // Verify data pointer points to external data
  EXPECT_EQ(tensor.data_ptr(), static_cast<void*>(external_data));

  // Verify data is accessible through tensor
  float* data = static_cast<float*>(tensor.data_ptr());
  for (size_t i = 0; i < kNumFloats; ++i) {
    EXPECT_FLOAT_EQ(data[i], static_cast<float>(i) * 1.5f);
  }
}

TEST(FromBlobTest, ModifyThroughTensor) {
  constexpr size_t kNumFloats = 16;
  float external_data[kNumFloats];

  for (size_t i = 0; i < kNumFloats; ++i) {
    external_data[i] = 0.0f;
  }

  SlimTensor tensor = from_blob(external_data, {4, 4}, c10::ScalarType::Float);

  // Modify through tensor
  float* data = static_cast<float*>(tensor.data_ptr());
  for (size_t i = 0; i < kNumFloats; ++i) {
    data[i] = static_cast<float>(i) * 10.0f;
  }

  // Verify external data was modified
  for (size_t i = 0; i < kNumFloats; ++i) {
    EXPECT_FLOAT_EQ(external_data[i], static_cast<float>(i) * 10.0f);
  }
}

TEST(FromBlobTest, ExternalDataSurvivesTensorDestruction) {
  constexpr size_t kNumFloats = 8;
  float external_data[kNumFloats];

  for (size_t i = 0; i < kNumFloats; ++i) {
    external_data[i] = static_cast<float>(i) * 2.0f;
  }

  {
    SlimTensor tensor =
        from_blob(external_data, {2, 4}, c10::ScalarType::Float);

    // Modify through tensor
    float* data = static_cast<float*>(tensor.data_ptr());
    data[0] = 999.0f;
  }
  // tensor is destroyed here

  // External data should still be accessible
  EXPECT_FLOAT_EQ(external_data[0], 999.0f);
  for (size_t i = 1; i < kNumFloats; ++i) {
    EXPECT_FLOAT_EQ(external_data[i], static_cast<float>(i) * 2.0f);
  }
}

// =============================================================================
// from_blob with Strides Tests
// =============================================================================

TEST(FromBlobTest, CustomStrides) {
  constexpr size_t kBufferSize = 16;
  float external_data[kBufferSize];

  for (size_t i = 0; i < kBufferSize; ++i) {
    external_data[i] = static_cast<float>(i);
  }

  // Create a 2x3 tensor with custom strides (transpose-like)
  SlimTensor tensor = from_blob(
      external_data,
      {2, 3},
      {1, 4}, // Non-contiguous strides
      c10::ScalarType::Float);

  EXPECT_EQ(tensor.size(0), 2);
  EXPECT_EQ(tensor.size(1), 3);
  EXPECT_EQ(tensor.stride(0), 1);
  EXPECT_EQ(tensor.stride(1), 4);
  EXPECT_FALSE(tensor.is_contiguous());
}

TEST(FromBlobTest, WithStorageOffset) {
  constexpr size_t kBufferSize = 20;
  float external_data[kBufferSize];

  for (size_t i = 0; i < kBufferSize; ++i) {
    external_data[i] = static_cast<float>(i);
  }

  // Create tensor with offset of 5 elements
  SlimTensor tensor = from_blob(
      external_data,
      {3, 4},
      c10::ScalarType::Float,
      CPU_DEVICE,
      5); // storage_offset = 5

  EXPECT_EQ(tensor.storage_offset(), 5);
  EXPECT_EQ(tensor.numel(), 12);

  // data_ptr() should point to external_data + 5 * sizeof(float)
  EXPECT_EQ(tensor.data_ptr(), static_cast<void*>(external_data + 5));

  // Verify first element is external_data[5]
  float* data = static_cast<float*>(tensor.data_ptr());
  EXPECT_FLOAT_EQ(data[0], 5.0f);
}

// =============================================================================
// from_blob with Different DTypes Tests
// =============================================================================

TEST(FromBlobTest, Int64Dtype) {
  constexpr size_t kNumElements = 6;
  int64_t external_data[kNumElements] = {10, 20, 30, 40, 50, 60};

  SlimTensor tensor = from_blob(external_data, {2, 3}, c10::ScalarType::Long);

  EXPECT_EQ(tensor.dtype(), c10::ScalarType::Long);
  EXPECT_EQ(tensor.itemsize(), sizeof(int64_t));
  EXPECT_EQ(tensor.numel(), kNumElements);

  int64_t* data = static_cast<int64_t*>(tensor.data_ptr());
  EXPECT_EQ(data[0], 10);
  EXPECT_EQ(data[5], 60);
}

TEST(FromBlobTest, Int8Dtype) {
  constexpr size_t kNumElements = 10;
  int8_t external_data[kNumElements];

  for (size_t i = 0; i < kNumElements; ++i) {
    external_data[i] = static_cast<int8_t>(i);
  }

  SlimTensor tensor = from_blob(external_data, {10}, c10::ScalarType::Char);

  EXPECT_EQ(tensor.dtype(), c10::ScalarType::Char);
  EXPECT_EQ(tensor.itemsize(), sizeof(int8_t));
  EXPECT_EQ(tensor.dim(), 1);

  int8_t* data = static_cast<int8_t*>(tensor.data_ptr());
  for (size_t i = 0; i < kNumElements; ++i) {
    EXPECT_EQ(data[i], static_cast<int8_t>(i));
  }
}

TEST(FromBlobTest, BoolDtype) {
  bool external_data[] = {true, false, true, false, true, true};

  SlimTensor tensor = from_blob(external_data, {2, 3}, c10::ScalarType::Bool);

  EXPECT_EQ(tensor.dtype(), c10::ScalarType::Bool);
  EXPECT_EQ(tensor.numel(), 6);

  bool* data = static_cast<bool*>(tensor.data_ptr());
  EXPECT_TRUE(data[0]);
  EXPECT_FALSE(data[1]);
  EXPECT_TRUE(data[2]);
}

// =============================================================================
// from_blob Copy Tests
// =============================================================================

TEST(FromBlobTest, CopyToOwnedTensor) {
  constexpr size_t kNumFloats = 12;
  float external_data[kNumFloats];

  for (size_t i = 0; i < kNumFloats; ++i) {
    external_data[i] = static_cast<float>(i) * 3.0f;
  }

  SlimTensor src = from_blob(external_data, {3, 4}, c10::ScalarType::Float);
  SlimTensor dst = empty({3, 4}, c10::ScalarType::Float);

  dst.copy_(src);

  // Verify dst has the data
  float* dst_data = static_cast<float*>(dst.data_ptr());
  for (size_t i = 0; i < kNumFloats; ++i) {
    EXPECT_FLOAT_EQ(dst_data[i], static_cast<float>(i) * 3.0f);
  }

  // Verify dst is independent of src
  external_data[0] = 999.0f;
  EXPECT_FLOAT_EQ(dst_data[0], 0.0f);
}

TEST(FromBlobTest, TensorCopyToFromBlob) {
  constexpr size_t kNumFloats = 6;
  float src_data[kNumFloats];
  float dst_data[kNumFloats];

  for (size_t i = 0; i < kNumFloats; ++i) {
    src_data[i] = static_cast<float>(i) * 5.0f;
    dst_data[i] = 0.0f;
  }

  SlimTensor src = from_blob(src_data, {2, 3}, c10::ScalarType::Float);
  SlimTensor dst = from_blob(dst_data, {2, 3}, c10::ScalarType::Float);

  dst.copy_(src);

  // Verify dst_data was modified
  for (size_t i = 0; i < kNumFloats; ++i) {
    EXPECT_FLOAT_EQ(dst_data[i], static_cast<float>(i) * 5.0f);
  }
}

// =============================================================================
// from_blob Shared Storage Tests
// =============================================================================

TEST(FromBlobTest, CopiedTensorSharesStorage) {
  constexpr size_t kNumFloats = 8;
  float external_data[kNumFloats];

  for (size_t i = 0; i < kNumFloats; ++i) {
    external_data[i] = static_cast<float>(i);
  }

  SlimTensor tensor1 = from_blob(external_data, {2, 4}, c10::ScalarType::Float);
  SlimTensor tensor2 = tensor1; // Copy constructor

  // Both should point to same storage
  EXPECT_EQ(tensor1.data_ptr(), tensor2.data_ptr());
  EXPECT_EQ(tensor1.storage().get(), tensor2.storage().get());

  // Modification through tensor2 affects tensor1
  float* data2 = static_cast<float*>(tensor2.data_ptr());
  data2[0] = 100.0f;

  float* data1 = static_cast<float*>(tensor1.data_ptr());
  EXPECT_FLOAT_EQ(data1[0], 100.0f);

  // And external data
  EXPECT_FLOAT_EQ(external_data[0], 100.0f);
}

// =============================================================================
// from_blob with ArrayRef Tests
// =============================================================================

TEST(FromBlobTest, WithArrayRef) {
  constexpr size_t kNumFloats = 6;
  float external_data[kNumFloats];

  for (size_t i = 0; i < kNumFloats; ++i) {
    external_data[i] = static_cast<float>(i);
  }

  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = {3, 1};

  SlimTensor tensor = from_blob(
      external_data,
      makeArrayRef(sizes),
      makeArrayRef(strides),
      c10::ScalarType::Float);

  EXPECT_EQ(tensor.size(0), 2);
  EXPECT_EQ(tensor.size(1), 3);
  EXPECT_EQ(tensor.stride(0), 3);
  EXPECT_EQ(tensor.stride(1), 1);
  EXPECT_TRUE(tensor.is_contiguous());
}

// =============================================================================
// Dtype and Device Type Validation Tests
// =============================================================================

TEST(FromBlobTest, InvalidDtypeUndefined) {
  constexpr size_t kNumFloats = 6;
  float external_data[kNumFloats];

  EXPECT_DEATH(
      from_blob(external_data, {2, 3}, c10::ScalarType::Undefined), "");
}

TEST(FromBlobTest, InvalidDtypeDouble) {
  constexpr size_t kNumFloats = 6;
  float external_data[kNumFloats];

  EXPECT_DEATH(
      from_blob(external_data, {2, 3}, static_cast<c10::ScalarType>(7)), "");
}

TEST(FromBlobTest, InvalidDeviceType) {
  constexpr size_t kNumFloats = 6;
  float external_data[kNumFloats];

  c10::Device invalid_device(static_cast<c10::DeviceType>(100), 0);

  EXPECT_DEATH(
      from_blob(external_data, {2, 3}, c10::ScalarType::Float, invalid_device),
      "");
}

// =============================================================================
// CUDA from_blob Tests
// Tests are skipped at runtime if CUDA hardware is not available.
// =============================================================================

#ifdef CUDA_AVAILABLE

// =============================================================================
// from_blob CUDA Basic Tests
// =============================================================================

TEST(FromBlobCUDATest, BasicConstruction) {
  constexpr size_t kNumFloats = 24;
  constexpr size_t kNbytes = kNumFloats * sizeof(float);

  // Allocate CUDA memory
  float* cuda_data =
      static_cast<float*>(DeviceTraits<c10::DeviceType::CUDA>::allocate(
          kNbytes, DEFAULT_CUDA_DEVICE));

  // Initialize via CPU buffer
  float* cpu_buffer = static_cast<float*>(
      DeviceTraits<c10::DeviceType::CPU>::allocate(kNbytes));
  for (size_t i = 0; i < kNumFloats; ++i) {
    cpu_buffer[i] = static_cast<float>(i) * 1.5f;
  }
  DeviceTraits<c10::DeviceType::CUDA>::memcpy(
      cuda_data, cpu_buffer, kNbytes, DEFAULT_CUDA_DEVICE, CPU_DEVICE);

  SlimTensor tensor = from_blob(
      cuda_data, {2, 3, 4}, c10::ScalarType::Float, DEFAULT_CUDA_DEVICE);

  // Verify tensor properties
  EXPECT_EQ(tensor.numel(), kNumFloats);
  EXPECT_EQ(tensor.dim(), 3);
  EXPECT_EQ(tensor.size(0), 2);
  EXPECT_EQ(tensor.size(1), 3);
  EXPECT_EQ(tensor.size(2), 4);
  EXPECT_EQ(tensor.dtype(), c10::ScalarType::Float);
  EXPECT_TRUE(tensor.is_cuda());
  EXPECT_FALSE(tensor.is_cpu());
  EXPECT_TRUE(tensor.is_contiguous());
  EXPECT_EQ(tensor.storage_offset(), 0);

  // Verify data pointer points to CUDA data
  EXPECT_EQ(tensor.data_ptr(), static_cast<void*>(cuda_data));

  // Verify data is accessible by copying back to CPU
  float* verify_buffer = static_cast<float*>(
      DeviceTraits<c10::DeviceType::CPU>::allocate(kNbytes));
  DeviceTraits<c10::DeviceType::CUDA>::memcpy(
      verify_buffer, cuda_data, kNbytes, CPU_DEVICE, DEFAULT_CUDA_DEVICE);
  for (size_t i = 0; i < kNumFloats; ++i) {
    EXPECT_FLOAT_EQ(verify_buffer[i], static_cast<float>(i) * 1.5f);
  }

  // Clean up
  DeviceTraits<c10::DeviceType::CPU>::free(cpu_buffer);
  DeviceTraits<c10::DeviceType::CPU>::free(verify_buffer);
  DeviceTraits<c10::DeviceType::CUDA>::free(cuda_data);
}

TEST(FromBlobCUDATest, ExternalDataSurvivesTensorDestruction) {
  constexpr size_t kNumFloats = 8;
  constexpr size_t kNbytes = kNumFloats * sizeof(float);

  // Allocate CUDA memory
  float* cuda_data =
      static_cast<float*>(DeviceTraits<c10::DeviceType::CUDA>::allocate(
          kNbytes, DEFAULT_CUDA_DEVICE));

  // Initialize via CPU buffer
  float* cpu_buffer = static_cast<float*>(
      DeviceTraits<c10::DeviceType::CPU>::allocate(kNbytes));
  for (size_t i = 0; i < kNumFloats; ++i) {
    cpu_buffer[i] = static_cast<float>(i) * 2.0f;
  }
  DeviceTraits<c10::DeviceType::CUDA>::memcpy(
      cuda_data, cpu_buffer, kNbytes, DEFAULT_CUDA_DEVICE, CPU_DEVICE);

  {
    SlimTensor tensor = from_blob(
        cuda_data, {2, 4}, c10::ScalarType::Float, DEFAULT_CUDA_DEVICE);

    // Modify first element via CPU buffer and copy back
    cpu_buffer[0] = 999.0f;
    DeviceTraits<c10::DeviceType::CUDA>::memcpy(
        cuda_data, cpu_buffer, sizeof(float), DEFAULT_CUDA_DEVICE, CPU_DEVICE);
  }
  // tensor is destroyed here

  // External CUDA data should still be accessible
  float* verify_buffer = static_cast<float*>(
      DeviceTraits<c10::DeviceType::CPU>::allocate(kNbytes));
  DeviceTraits<c10::DeviceType::CUDA>::memcpy(
      verify_buffer, cuda_data, kNbytes, CPU_DEVICE, DEFAULT_CUDA_DEVICE);
  EXPECT_FLOAT_EQ(verify_buffer[0], 999.0f);
  for (size_t i = 1; i < kNumFloats; ++i) {
    EXPECT_FLOAT_EQ(verify_buffer[i], static_cast<float>(i) * 2.0f);
  }

  // Clean up
  DeviceTraits<c10::DeviceType::CPU>::free(cpu_buffer);
  DeviceTraits<c10::DeviceType::CPU>::free(verify_buffer);
  DeviceTraits<c10::DeviceType::CUDA>::free(cuda_data);
}

// =============================================================================
// from_blob CUDA with Strides Tests
// =============================================================================

TEST(FromBlobCUDATest, CustomStrides) {
  constexpr size_t kBufferSize = 16;
  constexpr size_t kNbytes = kBufferSize * sizeof(float);

  float* cuda_data =
      static_cast<float*>(DeviceTraits<c10::DeviceType::CUDA>::allocate(
          kNbytes, DEFAULT_CUDA_DEVICE));

  // Create a 2x3 tensor with custom strides (transpose-like)
  SlimTensor tensor = from_blob(
      cuda_data,
      {2, 3},
      {1, 4}, // Non-contiguous strides
      c10::ScalarType::Float,
      DEFAULT_CUDA_DEVICE);

  EXPECT_EQ(tensor.size(0), 2);
  EXPECT_EQ(tensor.size(1), 3);
  EXPECT_EQ(tensor.stride(0), 1);
  EXPECT_EQ(tensor.stride(1), 4);
  EXPECT_FALSE(tensor.is_contiguous());
  EXPECT_TRUE(tensor.is_cuda());

  DeviceTraits<c10::DeviceType::CUDA>::free(cuda_data);
}

TEST(FromBlobCUDATest, WithStorageOffset) {
  constexpr size_t kBufferSize = 20;
  constexpr size_t kNbytes = kBufferSize * sizeof(float);

  float* cuda_data =
      static_cast<float*>(DeviceTraits<c10::DeviceType::CUDA>::allocate(
          kNbytes, DEFAULT_CUDA_DEVICE));

  // Initialize via CPU buffer
  float* cpu_buffer = static_cast<float*>(
      DeviceTraits<c10::DeviceType::CPU>::allocate(kNbytes));
  for (size_t i = 0; i < kBufferSize; ++i) {
    cpu_buffer[i] = static_cast<float>(i);
  }
  DeviceTraits<c10::DeviceType::CUDA>::memcpy(
      cuda_data, cpu_buffer, kNbytes, DEFAULT_CUDA_DEVICE, CPU_DEVICE);

  // Create tensor with offset of 5 elements
  SlimTensor tensor = from_blob(
      cuda_data,
      {3, 4},
      c10::ScalarType::Float,
      DEFAULT_CUDA_DEVICE,
      5); // storage_offset = 5

  EXPECT_EQ(tensor.storage_offset(), 5);
  EXPECT_EQ(tensor.numel(), 12);
  EXPECT_TRUE(tensor.is_cuda());

  // data_ptr() should point to cuda_data + 5 * sizeof(float)
  EXPECT_EQ(tensor.data_ptr(), static_cast<void*>(cuda_data + 5));

  // Verify first element is cuda_data[5] by copying back
  float first_elem = 0.0f;
  DeviceTraits<c10::DeviceType::CUDA>::memcpy(
      &first_elem,
      cuda_data + 5,
      sizeof(float),
      CPU_DEVICE,
      DEFAULT_CUDA_DEVICE);
  EXPECT_FLOAT_EQ(first_elem, 5.0f);

  // Clean up
  DeviceTraits<c10::DeviceType::CPU>::free(cpu_buffer);
  DeviceTraits<c10::DeviceType::CUDA>::free(cuda_data);
}

// =============================================================================
// from_blob CUDA with Different DTypes Tests
// =============================================================================

TEST(FromBlobCUDATest, Int64Dtype) {
  constexpr size_t kNumElements = 6;
  constexpr size_t kNbytes = kNumElements * sizeof(int64_t);

  int64_t* cuda_data =
      static_cast<int64_t*>(DeviceTraits<c10::DeviceType::CUDA>::allocate(
          kNbytes, DEFAULT_CUDA_DEVICE));

  int64_t cpu_buffer[kNumElements] = {10, 20, 30, 40, 50, 60};
  DeviceTraits<c10::DeviceType::CUDA>::memcpy(
      cuda_data, cpu_buffer, kNbytes, DEFAULT_CUDA_DEVICE, CPU_DEVICE);

  SlimTensor tensor =
      from_blob(cuda_data, {2, 3}, c10::ScalarType::Long, DEFAULT_CUDA_DEVICE);

  EXPECT_EQ(tensor.dtype(), c10::ScalarType::Long);
  EXPECT_EQ(tensor.itemsize(), sizeof(int64_t));
  EXPECT_EQ(tensor.numel(), kNumElements);
  EXPECT_TRUE(tensor.is_cuda());

  // Verify by copying back
  int64_t verify_buffer[kNumElements];
  DeviceTraits<c10::DeviceType::CUDA>::memcpy(
      verify_buffer, cuda_data, kNbytes, CPU_DEVICE, DEFAULT_CUDA_DEVICE);
  EXPECT_EQ(verify_buffer[0], 10);
  EXPECT_EQ(verify_buffer[5], 60);

  DeviceTraits<c10::DeviceType::CUDA>::free(cuda_data);
}

TEST(FromBlobCUDATest, Int8Dtype) {
  constexpr size_t kNumElements = 10;
  constexpr size_t kNbytes = kNumElements * sizeof(int8_t);

  int8_t* cuda_data =
      static_cast<int8_t*>(DeviceTraits<c10::DeviceType::CUDA>::allocate(
          kNbytes, DEFAULT_CUDA_DEVICE));

  int8_t cpu_buffer[kNumElements];
  for (size_t i = 0; i < kNumElements; ++i) {
    cpu_buffer[i] = static_cast<int8_t>(i);
  }
  DeviceTraits<c10::DeviceType::CUDA>::memcpy(
      cuda_data, cpu_buffer, kNbytes, DEFAULT_CUDA_DEVICE, CPU_DEVICE);

  SlimTensor tensor =
      from_blob(cuda_data, {10}, c10::ScalarType::Char, DEFAULT_CUDA_DEVICE);

  EXPECT_EQ(tensor.dtype(), c10::ScalarType::Char);
  EXPECT_EQ(tensor.itemsize(), sizeof(int8_t));
  EXPECT_EQ(tensor.dim(), 1);
  EXPECT_TRUE(tensor.is_cuda());

  // Verify by copying back
  int8_t verify_buffer[kNumElements];
  DeviceTraits<c10::DeviceType::CUDA>::memcpy(
      verify_buffer, cuda_data, kNbytes, CPU_DEVICE, DEFAULT_CUDA_DEVICE);
  for (size_t i = 0; i < kNumElements; ++i) {
    EXPECT_EQ(verify_buffer[i], static_cast<int8_t>(i));
  }

  DeviceTraits<c10::DeviceType::CUDA>::free(cuda_data);
}

// =============================================================================
// from_blob CUDA Cross-Device Copy Tests
// =============================================================================

TEST(FromBlobCUDATest, CopyCPUFromBlobToCUDAFromBlob) {
  constexpr size_t kNumFloats = 6;
  constexpr size_t kNbytes = kNumFloats * sizeof(float);

  // Create CPU source with from_blob
  float cpu_src_data[kNumFloats];
  for (size_t i = 0; i < kNumFloats; ++i) {
    cpu_src_data[i] = static_cast<float>(i) * 3.0f;
  }
  SlimTensor cpu_src =
      from_blob(cpu_src_data, {2, 3}, c10::ScalarType::Float, CPU_DEVICE);

  // Create CUDA destination with from_blob
  float* cuda_dst_data =
      static_cast<float*>(DeviceTraits<c10::DeviceType::CUDA>::allocate(
          kNbytes, DEFAULT_CUDA_DEVICE));
  SlimTensor cuda_dst = from_blob(
      cuda_dst_data, {2, 3}, c10::ScalarType::Float, DEFAULT_CUDA_DEVICE);

  // Copy CPU -> CUDA
  cuda_dst.copy_(cpu_src);

  // Verify by copying back to CPU
  float verify_buffer[kNumFloats];
  DeviceTraits<c10::DeviceType::CUDA>::memcpy(
      verify_buffer, cuda_dst_data, kNbytes, CPU_DEVICE, DEFAULT_CUDA_DEVICE);
  for (size_t i = 0; i < kNumFloats; ++i) {
    EXPECT_FLOAT_EQ(verify_buffer[i], static_cast<float>(i) * 3.0f);
  }

  DeviceTraits<c10::DeviceType::CUDA>::free(cuda_dst_data);
}

TEST(FromBlobCUDATest, CopyCUDAFromBlobToCPUFromBlob) {
  constexpr size_t kNumFloats = 4;
  constexpr size_t kNbytes = kNumFloats * sizeof(float);

  // Create and initialize CUDA source with from_blob
  float* cuda_src_data =
      static_cast<float*>(DeviceTraits<c10::DeviceType::CUDA>::allocate(
          kNbytes, DEFAULT_CUDA_DEVICE));
  float cpu_init[kNumFloats];
  for (size_t i = 0; i < kNumFloats; ++i) {
    cpu_init[i] = static_cast<float>(i) + 100.0f;
  }
  DeviceTraits<c10::DeviceType::CUDA>::memcpy(
      cuda_src_data, cpu_init, kNbytes, DEFAULT_CUDA_DEVICE, CPU_DEVICE);
  SlimTensor cuda_src = from_blob(
      cuda_src_data, {2, 2}, c10::ScalarType::Float, DEFAULT_CUDA_DEVICE);

  // Create CPU destination with from_blob
  float cpu_dst_data[kNumFloats] = {0.0f, 0.0f, 0.0f, 0.0f};
  SlimTensor cpu_dst =
      from_blob(cpu_dst_data, {2, 2}, c10::ScalarType::Float, CPU_DEVICE);

  // Copy CUDA -> CPU
  cpu_dst.copy_(cuda_src);

  // Verify CPU destination data
  for (size_t i = 0; i < kNumFloats; ++i) {
    EXPECT_FLOAT_EQ(cpu_dst_data[i], static_cast<float>(i) + 100.0f);
  }

  DeviceTraits<c10::DeviceType::CUDA>::free(cuda_src_data);
}

TEST(FromBlobCUDATest, CopyCUDAFromBlobToCUDAFromBlob) {
  constexpr size_t kNumFloats = 4;
  constexpr size_t kNbytes = kNumFloats * sizeof(float);

  // Create and initialize CUDA source with from_blob
  float* cuda_src_data =
      static_cast<float*>(DeviceTraits<c10::DeviceType::CUDA>::allocate(
          kNbytes, DEFAULT_CUDA_DEVICE));
  float cpu_init[kNumFloats];
  for (size_t i = 0; i < kNumFloats; ++i) {
    cpu_init[i] = static_cast<float>(i) * 5.0f;
  }
  DeviceTraits<c10::DeviceType::CUDA>::memcpy(
      cuda_src_data, cpu_init, kNbytes, DEFAULT_CUDA_DEVICE, CPU_DEVICE);
  SlimTensor cuda_src = from_blob(
      cuda_src_data, {2, 2}, c10::ScalarType::Float, DEFAULT_CUDA_DEVICE);

  // Create CUDA destination with from_blob
  float* cuda_dst_data =
      static_cast<float*>(DeviceTraits<c10::DeviceType::CUDA>::allocate(
          kNbytes, DEFAULT_CUDA_DEVICE));
  SlimTensor cuda_dst = from_blob(
      cuda_dst_data, {2, 2}, c10::ScalarType::Float, DEFAULT_CUDA_DEVICE);

  // Copy CUDA -> CUDA
  cuda_dst.copy_(cuda_src);

  // Verify by copying back to CPU
  float verify_buffer[kNumFloats];
  DeviceTraits<c10::DeviceType::CUDA>::memcpy(
      verify_buffer, cuda_dst_data, kNbytes, CPU_DEVICE, DEFAULT_CUDA_DEVICE);
  for (size_t i = 0; i < kNumFloats; ++i) {
    EXPECT_FLOAT_EQ(verify_buffer[i], static_cast<float>(i) * 5.0f);
  }

  DeviceTraits<c10::DeviceType::CUDA>::free(cuda_src_data);
  DeviceTraits<c10::DeviceType::CUDA>::free(cuda_dst_data);
}

// =============================================================================
// from_blob CUDA to empty() Copy Tests
// =============================================================================

TEST(FromBlobCUDATest, CopyCUDAFromBlobToOwnedCUDATensor) {
  constexpr size_t kNumFloats = 12;
  constexpr size_t kNbytes = kNumFloats * sizeof(float);

  // Create CUDA source with from_blob
  float* cuda_src_data =
      static_cast<float*>(DeviceTraits<c10::DeviceType::CUDA>::allocate(
          kNbytes, DEFAULT_CUDA_DEVICE));
  float cpu_init[kNumFloats];
  for (size_t i = 0; i < kNumFloats; ++i) {
    cpu_init[i] = static_cast<float>(i) * 7.0f;
  }
  DeviceTraits<c10::DeviceType::CUDA>::memcpy(
      cuda_src_data, cpu_init, kNbytes, DEFAULT_CUDA_DEVICE, CPU_DEVICE);
  SlimTensor cuda_src = from_blob(
      cuda_src_data, {3, 4}, c10::ScalarType::Float, DEFAULT_CUDA_DEVICE);

  // Create owned CUDA destination with empty()
  SlimTensor cuda_dst =
      empty({3, 4}, c10::ScalarType::Float, DEFAULT_CUDA_DEVICE);

  cuda_dst.copy_(cuda_src);

  // Verify by copying back to CPU
  float verify_buffer[kNumFloats];
  DeviceTraits<c10::DeviceType::CUDA>::memcpy(
      verify_buffer,
      cuda_dst.data_ptr(),
      kNbytes,
      CPU_DEVICE,
      DEFAULT_CUDA_DEVICE);
  for (size_t i = 0; i < kNumFloats; ++i) {
    EXPECT_FLOAT_EQ(verify_buffer[i], static_cast<float>(i) * 7.0f);
  }

  DeviceTraits<c10::DeviceType::CUDA>::free(cuda_src_data);
}

TEST(FromBlobCUDATest, CopyOwnedCUDATensorToCUDAFromBlob) {
  constexpr size_t kNumFloats = 6;
  constexpr size_t kNbytes = kNumFloats * sizeof(float);

  // Create owned CUDA source with empty() and initialize via CPU
  SlimTensor cuda_src =
      empty({2, 3}, c10::ScalarType::Float, DEFAULT_CUDA_DEVICE);
  float cpu_init[kNumFloats];
  for (size_t i = 0; i < kNumFloats; ++i) {
    cpu_init[i] = static_cast<float>(i) * 11.0f;
  }
  DeviceTraits<c10::DeviceType::CUDA>::memcpy(
      cuda_src.data_ptr(), cpu_init, kNbytes, DEFAULT_CUDA_DEVICE, CPU_DEVICE);

  // Create CUDA destination with from_blob
  float* cuda_dst_data =
      static_cast<float*>(DeviceTraits<c10::DeviceType::CUDA>::allocate(
          kNbytes, DEFAULT_CUDA_DEVICE));
  SlimTensor cuda_dst = from_blob(
      cuda_dst_data, {2, 3}, c10::ScalarType::Float, DEFAULT_CUDA_DEVICE);

  cuda_dst.copy_(cuda_src);

  // Verify by copying back to CPU
  float verify_buffer[kNumFloats];
  DeviceTraits<c10::DeviceType::CUDA>::memcpy(
      verify_buffer, cuda_dst_data, kNbytes, CPU_DEVICE, DEFAULT_CUDA_DEVICE);
  for (size_t i = 0; i < kNumFloats; ++i) {
    EXPECT_FLOAT_EQ(verify_buffer[i], static_cast<float>(i) * 11.0f);
  }

  DeviceTraits<c10::DeviceType::CUDA>::free(cuda_dst_data);
}

// =============================================================================
// from_blob CUDA Shared Storage Tests
// =============================================================================

TEST(FromBlobCUDATest, CopiedTensorSharesStorage) {
  constexpr size_t kNumFloats = 8;
  constexpr size_t kNbytes = kNumFloats * sizeof(float);

  float* cuda_data =
      static_cast<float*>(DeviceTraits<c10::DeviceType::CUDA>::allocate(
          kNbytes, DEFAULT_CUDA_DEVICE));

  SlimTensor tensor1 =
      from_blob(cuda_data, {2, 4}, c10::ScalarType::Float, DEFAULT_CUDA_DEVICE);
  SlimTensor tensor2 = tensor1; // Copy constructor

  // Both should point to same storage
  EXPECT_EQ(tensor1.data_ptr(), tensor2.data_ptr());
  EXPECT_EQ(tensor1.storage().get(), tensor2.storage().get());
  EXPECT_TRUE(tensor1.is_cuda());
  EXPECT_TRUE(tensor2.is_cuda());

  DeviceTraits<c10::DeviceType::CUDA>::free(cuda_data);
}

#endif // CUDA_AVAILABLE

} // namespace executorch::backends::aoti::slim
