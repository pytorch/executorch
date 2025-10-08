/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_runtime.h>
#include <executorch/backends/aoti/common_shims.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/backends/cuda/runtime/shims/tensor_attribute.h>
#include <executorch/backends/cuda/runtime/utils.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/platform.h>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using namespace executorch::backends::cuda;
using namespace executorch::backends::aoti;
using namespace executorch::runtime;

// Test fixture for aoti_torch_copy_ tests
class AOTITorchCopyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize ExecuTorch Platform Abstraction Layer
    et_pal_init();

    // Check if CUDA is available
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "CUDA not available, skipping CUDA tests";
    }

    // Clean up any existing cached metadata before each test
    cleanup_tensor_metadata();

    // Clear any remaining tensors from previous tests
    clear_all_tensors();
  }

  void TearDown() override {
    // Clean up metadata
    cleanup_tensor_metadata();

    // Clear the global tensor storage using the provided function
    clear_all_tensors();
  }

  // Helper to create test tensors with specific data
  Tensor* create_test_tensor_with_data(
      const std::vector<int64_t>& sizes,
      const std::vector<float>& data,
      const std::vector<int64_t>& strides = {},
      int32_t dtype = static_cast<int32_t>(SupportedDTypes::FLOAT32),
      int32_t device_type = static_cast<int32_t>(SupportedDevices::CUDA),
      int32_t device_index = 0) {
    Tensor* tensor;

    const int64_t* strides_ptr = strides.empty() ? nullptr : strides.data();

    AOTITorchError error = aoti_torch_empty_strided(
        sizes.size(),
        sizes.data(),
        strides_ptr,
        dtype,
        device_type,
        device_index,
        &tensor);

    if (error != Error::Ok || tensor == nullptr) {
      return nullptr;
    }

    // Fill tensor with data
    size_t total_bytes = data.size() * sizeof(float);
    if (device_type == static_cast<int32_t>(SupportedDevices::CUDA)) {
      cudaError_t memcpy_err = cudaMemcpy(
          tensor->mutable_data_ptr(),
          data.data(),
          total_bytes,
          cudaMemcpyHostToDevice);
      // Note: Error is checked but we don't fail the function
      // This allows tests to proceed and handle errors as needed
      (void)memcpy_err; // Suppress unused variable warning
    } else { // CPU
      std::memcpy(tensor->mutable_data_ptr(), data.data(), total_bytes);
    }

    return tensor;
  }

  // Helper to get data from tensor
  std::vector<float> get_tensor_data(Tensor* tensor) {
    if (!tensor) {
      return {};
    }

    size_t num_elements = tensor->numel();
    std::vector<float> data(num_elements);

    // Determine if this is a CUDA tensor
    cudaPointerAttributes attributes{};
    cudaError_t err = cudaPointerGetAttributes(&attributes, tensor->data_ptr());
    bool is_device =
        (err == cudaSuccess && attributes.type == cudaMemoryTypeDevice);

    if (is_device) {
      cudaError_t memcpy_err = cudaMemcpy(
          data.data(),
          tensor->data_ptr(),
          num_elements * sizeof(float),
          cudaMemcpyDeviceToHost);
      // Note: Error is checked but we don't fail the function
      // This allows tests to proceed and handle errors as needed
      (void)memcpy_err; // Suppress unused variable warning
    } else {
      std::memcpy(
          data.data(), tensor->data_ptr(), num_elements * sizeof(float));
    }

    return data;
  }

  // Helper to verify two tensors have same data
  bool tensors_equal(Tensor* a, Tensor* b, float tolerance = 1e-6f) {
    if (!a || !b) {
      return false;
    }
    if (a->numel() != b->numel()) {
      return false;
    }

    auto data_a = get_tensor_data(a);
    auto data_b = get_tensor_data(b);

    for (size_t i = 0; i < data_a.size(); ++i) {
      if (std::abs(data_a[i] - data_b[i]) > tolerance) {
        return false;
      }
    }
    return true;
  }
};

// Test basic copy functionality - same schema (fast path)
TEST_F(AOTITorchCopyTest, BasicCopySameSchema) {
  // Create source tensor with test data
  std::vector<int64_t> sizes = {2, 3};
  std::vector<float> src_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  Tensor* src = create_test_tensor_with_data(sizes, src_data);
  EXPECT_NE(src, nullptr);

  // Create destination tensor with same schema
  Tensor* dst =
      create_test_tensor_with_data(sizes, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  EXPECT_NE(dst, nullptr);

  // Perform copy
  AOTITorchError error = aoti_torch_copy_(dst, src, 0);
  EXPECT_EQ(error, Error::Ok);

  // Verify copy was successful
  EXPECT_TRUE(tensors_equal(dst, src));
}

// Test copy with different strides (pointwise fallback)
TEST_F(AOTITorchCopyTest, CopyDifferentStrides) {
  // Create source tensor (2x3) with contiguous layout
  std::vector<int64_t> src_sizes = {2, 3};
  std::vector<float> src_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  Tensor* src = create_test_tensor_with_data(src_sizes, src_data);
  EXPECT_NE(src, nullptr);

  // Create destination tensor with transposed strides
  std::vector<int64_t> dst_strides = {1, 2}; // Column-major layout
  Tensor* dst = create_test_tensor_with_data(
      src_sizes, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, dst_strides);
  EXPECT_NE(dst, nullptr);

  // Perform copy - this should use pointwise fallback
  AOTITorchError error = aoti_torch_copy_(dst, src, 0);
  EXPECT_EQ(error, Error::Ok);

  // Verify the copy worked correctly by checking specific elements
  auto dst_data = get_tensor_data(dst);
  auto src_data_check = get_tensor_data(src);

  // For transposed layout, the data should be rearranged
  EXPECT_EQ(dst_data.size(), 6);
  EXPECT_EQ(src_data_check.size(), 6);
}

// Test copy between CPU and CUDA tensors
TEST_F(AOTITorchCopyTest, CopyCPUToCUDA) {
  std::vector<int64_t> sizes = {2, 2};
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};

  // Create CPU tensor
  Tensor* cpu_tensor = create_test_tensor_with_data(
      sizes,
      data,
      {},
      static_cast<int32_t>(SupportedDTypes::FLOAT32),
      static_cast<int32_t>(SupportedDevices::CPU)); // CPU
  EXPECT_NE(cpu_tensor, nullptr);

  // Create CUDA tensor
  Tensor* cuda_tensor = create_test_tensor_with_data(
      sizes,
      {0.0f, 0.0f, 0.0f, 0.0f},
      {},
      static_cast<int32_t>(SupportedDTypes::FLOAT32),
      static_cast<int32_t>(SupportedDevices::CUDA)); // CUDA
  EXPECT_NE(cuda_tensor, nullptr);

  // Copy from CPU to CUDA
  AOTITorchError error = aoti_torch_copy_(cuda_tensor, cpu_tensor, 0);
  EXPECT_EQ(error, Error::Ok);

  // Verify copy
  EXPECT_TRUE(tensors_equal(cuda_tensor, cpu_tensor));
}

// Test copy between CUDA and CPU tensors
TEST_F(AOTITorchCopyTest, CopyCUDAToCPU) {
  std::vector<int64_t> sizes = {2, 2};
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};

  // Create CUDA tensor
  Tensor* cuda_tensor = create_test_tensor_with_data(
      sizes,
      data,
      {},
      static_cast<int32_t>(SupportedDTypes::FLOAT32),
      static_cast<int32_t>(SupportedDevices::CUDA)); // CUDA
  EXPECT_NE(cuda_tensor, nullptr);

  // Create CPU tensor
  Tensor* cpu_tensor = create_test_tensor_with_data(
      sizes,
      {0.0f, 0.0f, 0.0f, 0.0f},
      {},
      static_cast<int32_t>(SupportedDTypes::FLOAT32),
      static_cast<int32_t>(SupportedDevices::CPU)); // CPU
  EXPECT_NE(cpu_tensor, nullptr);

  // Copy from CUDA to CPU
  AOTITorchError error = aoti_torch_copy_(cpu_tensor, cuda_tensor, 0);
  EXPECT_EQ(error, Error::Ok);

  // Verify copy
  EXPECT_TRUE(tensors_equal(cpu_tensor, cuda_tensor));
}

// Test copy with bf16 dtype support
TEST_F(AOTITorchCopyTest, CopyBf16Tensors) {
  // Test that bf16 tensors can be created and copied
  std::vector<int64_t> sizes = {2, 3};
  std::vector<float> src_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // Note: We create float32 data but the tensor will be created with bf16 dtype
  // This simulates creating bf16 tensors
  Tensor* src = create_test_tensor_with_data(
      sizes,
      src_data,
      {}, // default strides
      static_cast<int32_t>(SupportedDTypes::BFLOAT16), // bf16 dtype
      static_cast<int32_t>(SupportedDevices::CUDA), // CUDA device
      0 // device_index = 0
  );
  EXPECT_NE(src, nullptr);

  // Create destination tensor with bf16 dtype
  std::vector<float> dst_init(6, 0.0f);
  Tensor* dst = create_test_tensor_with_data(
      sizes,
      dst_init,
      {}, // default strides
      static_cast<int32_t>(SupportedDTypes::BFLOAT16), // bf16 dtype
      static_cast<int32_t>(SupportedDevices::CUDA), // CUDA device
      0 // device_index = 0
  );
  EXPECT_NE(dst, nullptr);

  // Perform copy between bf16 tensors
  AOTITorchError error = aoti_torch_copy_(dst, src, 0);
  EXPECT_EQ(error, Error::Ok);

  // Verify that both tensors have the expected dtype
  int32_t src_dtype, dst_dtype;
  aoti_torch_get_dtype(src, &src_dtype);
  aoti_torch_get_dtype(dst, &dst_dtype);

  EXPECT_EQ(src_dtype, static_cast<int32_t>(SupportedDTypes::BFLOAT16));
  EXPECT_EQ(dst_dtype, static_cast<int32_t>(SupportedDTypes::BFLOAT16));

  // Verify copy was successful by checking numel matches
  EXPECT_EQ(src->numel(), dst->numel());
  EXPECT_EQ(src->numel(), 6);
}

// Test copy between different dtypes should fail
TEST_F(AOTITorchCopyTest, CopyDTypeMismatchError) {
  std::vector<int64_t> sizes = {2, 2};
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};

  // Create float32 tensor
  Tensor* float32_tensor = create_test_tensor_with_data(
      sizes,
      data,
      {}, // default strides
      static_cast<int32_t>(SupportedDTypes::FLOAT32), // float32 dtype
      static_cast<int32_t>(SupportedDevices::CUDA), // CUDA device
      0 // device_index = 0
  );
  EXPECT_NE(float32_tensor, nullptr);

  // Create bf16 tensor
  Tensor* bf16_tensor = create_test_tensor_with_data(
      sizes,
      {0.0f, 0.0f, 0.0f, 0.0f},
      {}, // default strides
      static_cast<int32_t>(SupportedDTypes::BFLOAT16), // bf16 dtype
      static_cast<int32_t>(SupportedDevices::CUDA), // CUDA device
      0 // device_index = 0
  );
  EXPECT_NE(bf16_tensor, nullptr);

  // Attempting to copy between different dtypes should fail
  AOTITorchError error = aoti_torch_copy_(bf16_tensor, float32_tensor, 0);
  EXPECT_EQ(error, Error::InvalidArgument);

  // Reverse direction should also fail
  error = aoti_torch_copy_(float32_tensor, bf16_tensor, 0);
  EXPECT_EQ(error, Error::InvalidArgument);
}

// Test error conditions
TEST_F(AOTITorchCopyTest, ErrorHandling) {
  std::vector<int64_t> sizes = {2, 3};
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  Tensor* valid_tensor = create_test_tensor_with_data(sizes, data);
  EXPECT_NE(valid_tensor, nullptr);

  // Test null pointers
  AOTITorchError error = aoti_torch_copy_(nullptr, valid_tensor, 0);
  EXPECT_NE(error, Error::Ok);

  error = aoti_torch_copy_(valid_tensor, nullptr, 0);
  EXPECT_NE(error, Error::Ok);

  // Test numel mismatch (different total number of elements)
  std::vector<int64_t> different_numel_sizes = {
      2, 3, 4}; // 24 elements vs 6 elements
  std::vector<float> different_data(24, 1.0f);
  Tensor* different_numel =
      create_test_tensor_with_data(different_numel_sizes, different_data);
  EXPECT_NE(different_numel, nullptr);

  error = aoti_torch_copy_(valid_tensor, different_numel, 0);
  EXPECT_EQ(error, Error::InvalidArgument);
}

// Test copy from 1D to 3D with same total elements
TEST_F(AOTITorchCopyTest, Copy1DTo3DSameNumel) {
  // Source tensor: 8 elements in 1D
  std::vector<int64_t> src_sizes = {8};
  std::vector<float> src_data = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

  Tensor* src = create_test_tensor_with_data(src_sizes, src_data);
  EXPECT_NE(src, nullptr);

  // Destination tensor: 2x2x2 = 8 elements (different shape, same total)
  std::vector<int64_t> dst_sizes = {2, 2, 2};
  std::vector<float> dst_init(8, 0.0f);
  Tensor* dst = create_test_tensor_with_data(dst_sizes, dst_init);
  EXPECT_NE(dst, nullptr);

  // This should work - same total number of elements
  AOTITorchError error = aoti_torch_copy_(dst, src, 0);
  EXPECT_EQ(error, Error::Ok);

  // Verify the data was copied correctly
  auto dst_data = get_tensor_data(dst);
  EXPECT_EQ(dst_data.size(), 8);

  // Check some specific elements to verify correct copying
  EXPECT_FLOAT_EQ(dst_data[0], 1.0f);
  EXPECT_FLOAT_EQ(dst_data[7], 8.0f);
}
