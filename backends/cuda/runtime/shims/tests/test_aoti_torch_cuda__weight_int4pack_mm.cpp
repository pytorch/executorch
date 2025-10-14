/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_runtime.h>
#include <executorch/backends/aoti/common_shims.h>
#include <executorch/backends/aoti/utils.h>
#include <executorch/backends/cuda/runtime/shims/int4mm.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/backends/cuda/runtime/shims/tensor_attribute.h>
#include <executorch/backends/cuda/runtime/utils.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/platform.h>
#include <gtest/gtest.h>
#include <vector>

using namespace executorch::backends::cuda;
using namespace executorch::backends::aoti;
using namespace executorch::runtime;

// Test fixture for aoti_torch_cuda__weight_int4pack_mm tests
class AOTITorchInt4MMTest : public ::testing::Test {
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

    // Check if GPU supports sm_80+ (required for int4mm)
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int compute_capability = prop.major * 10 + prop.minor;
    if (compute_capability < 80) {
      GTEST_SKIP() << "GPU compute capability " << compute_capability
                   << " < 80 (Ampere+), int4mm requires sm_80+";
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

  // Helper to create a BFloat16 tensor
  Tensor* create_bfloat16_tensor(const std::vector<int64_t>& sizes) {
    Tensor* tensor;

    AOTITorchError error = aoti_torch_empty_strided(
        sizes.size(),
        sizes.data(),
        nullptr, // default strides
        static_cast<int32_t>(SupportedDTypes::BFLOAT16),
        static_cast<int32_t>(SupportedDevices::CUDA),
        0, // device index
        &tensor);

    return (error == Error::Ok) ? tensor : nullptr;
  }

  // Helper to create an Int32 tensor
  Tensor* create_int32_tensor(const std::vector<int64_t>& sizes) {
    Tensor* tensor;

    AOTITorchError error = aoti_torch_empty_strided(
        sizes.size(),
        sizes.data(),
        nullptr, // default strides
        static_cast<int32_t>(SupportedDTypes::INT32),
        static_cast<int32_t>(SupportedDevices::CUDA),
        0, // device index
        &tensor);

    return (error == Error::Ok) ? tensor : nullptr;
  }
};

// Test basic int4mm functionality with minimal valid inputs
TEST_F(AOTITorchInt4MMTest, BasicFunctionality) {
  // Create input tensor A: [m, k] = [2, 128] in BFloat16
  int64_t m = 2;
  int64_t k = 128;
  int64_t n = 64;
  int64_t qGroupSize = 128;

  Tensor* A = create_bfloat16_tensor({m, k});
  ASSERT_NE(A, nullptr) << "Failed to create input tensor A";

  // Create weight tensor B (int4 packed): [n/8, k/(innerKTiles*16), 32, 4] in
  // Int32 For int4mm, innerKTiles is typically 8, so k/(8*16) = 128/128 = 1
  int64_t B_innerKTiles = 8;
  int64_t B_kTiles = k / (B_innerKTiles * 16);
  Tensor* B = create_int32_tensor({n / 8, B_kTiles, 32, 4});
  ASSERT_NE(B, nullptr) << "Failed to create weight tensor B";

  // Create scale and zeros tensor: [k/qGroupSize, n, 2] in BFloat16
  // For k=128, qGroupSize=128, k/qGroupSize=1
  Tensor* qScaleAndZeros = create_bfloat16_tensor({k / qGroupSize, n, 2});
  ASSERT_NE(qScaleAndZeros, nullptr)
      << "Failed to create qScaleAndZeros tensor";

  // Create output tensor: [m, n] in BFloat16
  Tensor* output = create_bfloat16_tensor({m, n});
  ASSERT_NE(output, nullptr) << "Failed to create output tensor";

  printf("Testing int4mm with shapes:\n");
  printf("  A: [%ldx%ld] BFloat16\n", m, k);
  printf("  B: [%ldx%ldx32x4] Int32\n", n / 8, B_kTiles);
  printf("  qScaleAndZeros: [%ldx%ldx2] BFloat16\n", k / qGroupSize, n);
  printf("  qGroupSize: %ld\n", qGroupSize);
  printf("  Output: [%ldx%ld] BFloat16\n", m, n);

  // Call int4mm
  AOTITorchError error = aoti_torch_cuda__weight_int4pack_mm(
      A, B, qGroupSize, qScaleAndZeros, &output);

  // Check if the function succeeded
  EXPECT_EQ(error, Error::Ok) << "int4mm operation should succeed";

  // Verify output tensor properties
  EXPECT_EQ(output->dim(), 2);
  EXPECT_EQ(output->size(0), m);
  EXPECT_EQ(output->size(1), n);

  printf("int4mm test passed successfully!\n");
}

// Test with different qGroupSize values
TEST_F(AOTITorchInt4MMTest, DifferentQGroupSizes) {
  int64_t m = 4;
  int64_t k = 256;
  int64_t n = 128;
  int64_t B_innerKTiles = 8;

  // Test qGroupSize = 64
  {
    int64_t qGroupSize = 64;

    Tensor* A = create_bfloat16_tensor({m, k});
    ASSERT_NE(A, nullptr);

    Tensor* B = create_int32_tensor({n / 8, k / (B_innerKTiles * 16), 32, 4});
    ASSERT_NE(B, nullptr);

    Tensor* qScaleAndZeros = create_bfloat16_tensor({k / qGroupSize, n, 2});
    ASSERT_NE(qScaleAndZeros, nullptr);

    Tensor* output = create_bfloat16_tensor({m, n});
    ASSERT_NE(output, nullptr);

    printf("Testing int4mm with qGroupSize=64\n");

    AOTITorchError error = aoti_torch_cuda__weight_int4pack_mm(
        A, B, qGroupSize, qScaleAndZeros, &output);
    EXPECT_EQ(error, Error::Ok) << "int4mm with qGroupSize=64 should succeed";
  }

  // Test qGroupSize = 128
  {
    int64_t qGroupSize = 128;

    Tensor* A = create_bfloat16_tensor({m, k});
    ASSERT_NE(A, nullptr);

    Tensor* B = create_int32_tensor({n / 8, k / (B_innerKTiles * 16), 32, 4});
    ASSERT_NE(B, nullptr);

    Tensor* qScaleAndZeros = create_bfloat16_tensor({k / qGroupSize, n, 2});
    ASSERT_NE(qScaleAndZeros, nullptr);

    Tensor* output = create_bfloat16_tensor({m, n});
    ASSERT_NE(output, nullptr);

    printf("Testing int4mm with qGroupSize=128\n");

    AOTITorchError error = aoti_torch_cuda__weight_int4pack_mm(
        A, B, qGroupSize, qScaleAndZeros, &output);
    EXPECT_EQ(error, Error::Ok) << "int4mm with qGroupSize=128 should succeed";
  }

  // Test qGroupSize = 256
  {
    int64_t qGroupSize = 256;

    Tensor* A = create_bfloat16_tensor({m, k});
    ASSERT_NE(A, nullptr);

    Tensor* B = create_int32_tensor({n / 8, k / (B_innerKTiles * 16), 32, 4});
    ASSERT_NE(B, nullptr);

    Tensor* qScaleAndZeros = create_bfloat16_tensor({k / qGroupSize, n, 2});
    ASSERT_NE(qScaleAndZeros, nullptr);

    Tensor* output = create_bfloat16_tensor({m, n});
    ASSERT_NE(output, nullptr);

    printf("Testing int4mm with qGroupSize=256\n");

    AOTITorchError error = aoti_torch_cuda__weight_int4pack_mm(
        A, B, qGroupSize, qScaleAndZeros, &output);
    EXPECT_EQ(error, Error::Ok) << "int4mm with qGroupSize=256 should succeed";
  }
}

// Test error handling with null inputs
TEST_F(AOTITorchInt4MMTest, NullInputHandling) {
  int64_t m = 2;
  int64_t k = 128;
  int64_t n = 64;
  int64_t qGroupSize = 128;
  int64_t B_innerKTiles = 8;

  Tensor* A = create_bfloat16_tensor({m, k});
  Tensor* B = create_int32_tensor({n / 8, k / (B_innerKTiles * 16), 32, 4});
  Tensor* qScaleAndZeros = create_bfloat16_tensor({k / qGroupSize, n, 2});
  Tensor* output = create_bfloat16_tensor({m, n});

  // Test null A
  {
    AOTITorchError error = aoti_torch_cuda__weight_int4pack_mm(
        nullptr, B, qGroupSize, qScaleAndZeros, &output);
    EXPECT_EQ(error, Error::InvalidArgument)
        << "Should fail with null A tensor";
  }

  // Test null B
  {
    AOTITorchError error = aoti_torch_cuda__weight_int4pack_mm(
        A, nullptr, qGroupSize, qScaleAndZeros, &output);
    EXPECT_EQ(error, Error::InvalidArgument)
        << "Should fail with null B tensor";
  }

  // Test null qScaleAndZeros
  {
    AOTITorchError error =
        aoti_torch_cuda__weight_int4pack_mm(A, B, qGroupSize, nullptr, &output);
    EXPECT_EQ(error, Error::InvalidArgument)
        << "Should fail with null qScaleAndZeros tensor";
  }

  // Test null output pointer
  {
    AOTITorchError error = aoti_torch_cuda__weight_int4pack_mm(
        A, B, qGroupSize, qScaleAndZeros, nullptr);
    EXPECT_EQ(error, Error::InvalidArgument)
        << "Should fail with null output pointer";
  }

  // Test null output tensor (ret0 points to null)
  {
    Tensor* null_output = nullptr;
    AOTITorchError error = aoti_torch_cuda__weight_int4pack_mm(
        A, B, qGroupSize, qScaleAndZeros, &null_output);
    EXPECT_EQ(error, Error::InvalidArgument)
        << "Should fail with null output tensor";
  }
}

// Test with larger batch size
TEST_F(AOTITorchInt4MMTest, LargerBatchSize) {
  int64_t m = 16; // Batch size
  int64_t k = 256;
  int64_t n = 128;
  int64_t qGroupSize = 128;
  int64_t B_innerKTiles = 8;

  Tensor* A = create_bfloat16_tensor({m, k});
  ASSERT_NE(A, nullptr);

  Tensor* B = create_int32_tensor({n / 8, k / (B_innerKTiles * 16), 32, 4});
  ASSERT_NE(B, nullptr);

  Tensor* qScaleAndZeros = create_bfloat16_tensor({k / qGroupSize, n, 2});
  ASSERT_NE(qScaleAndZeros, nullptr);

  Tensor* output = create_bfloat16_tensor({m, n});
  ASSERT_NE(output, nullptr);

  printf("Testing int4mm with larger batch: m=%ld\n", m);

  AOTITorchError error = aoti_torch_cuda__weight_int4pack_mm(
      A, B, qGroupSize, qScaleAndZeros, &output);

  EXPECT_EQ(error, Error::Ok) << "int4mm with larger batch should succeed";
  EXPECT_EQ(output->size(0), m);
  EXPECT_EQ(output->size(1), n);
}

// Test with larger tensors
TEST_F(AOTITorchInt4MMTest, LargerTensors) {
  int64_t m = 8;
  int64_t k = 512;
  int64_t n = 256;
  int64_t qGroupSize = 128;
  int64_t B_innerKTiles = 8;

  Tensor* A = create_bfloat16_tensor({m, k});
  ASSERT_NE(A, nullptr);

  Tensor* B = create_int32_tensor({n / 8, k / (B_innerKTiles * 16), 32, 4});
  ASSERT_NE(B, nullptr);

  Tensor* qScaleAndZeros = create_bfloat16_tensor({k / qGroupSize, n, 2});
  ASSERT_NE(qScaleAndZeros, nullptr);

  Tensor* output = create_bfloat16_tensor({m, n});
  ASSERT_NE(output, nullptr);

  printf(
      "Testing int4mm with larger tensors: [%ldx%ld] x [weight] -> [%ldx%ld]\n",
      m,
      k,
      m,
      n);

  AOTITorchError error = aoti_torch_cuda__weight_int4pack_mm(
      A, B, qGroupSize, qScaleAndZeros, &output);

  EXPECT_EQ(error, Error::Ok) << "int4mm with larger tensors should succeed";
  EXPECT_EQ(output->dim(), 2);
  EXPECT_EQ(output->size(0), m);
  EXPECT_EQ(output->size(1), n);
}
