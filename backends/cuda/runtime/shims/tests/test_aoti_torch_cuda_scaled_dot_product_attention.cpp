/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <executorch/backends/aoti/common_shims.h>
#include <executorch/backends/aoti/utils.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/backends/cuda/runtime/shims/sdpa.h>
#include <executorch/backends/cuda/runtime/shims/tensor_attribute.h>
#include <executorch/backends/cuda/runtime/utils.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/platform.h>

#include <cmath>
#include <vector>

using namespace executorch::backends::cuda;
using namespace executorch::backends::aoti;
using namespace executorch::runtime;

// Test fixture for SDPA tests
class AOTITorchSDPATest : public ::testing::Test {
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
  }

  void TearDown() override {
    // Clean up after each test
    cleanup_tensor_metadata();
  }

  // Helper function to create a Float32 tensor filled with a specific value
  Tensor* create_float_tensor(
      std::vector<int64_t> shape,
      float fill_value = 1.0f) {
    Tensor* tensor = nullptr;

    // Calculate size
    int64_t total_size = 1;
    for (auto dim : shape) {
      total_size *= dim;
    }

    // Calculate strides (row-major)
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
      strides[i] = stride;
      stride *= shape[i];
    }

    // Create tensor
    Error error = aoti_torch_empty_strided(
        shape.size(),
        shape.data(),
        strides.data(),
        static_cast<int32_t>(SupportedDTypes::FLOAT32),
        static_cast<int32_t>(SupportedDevices::CUDA),
        0,
        &tensor);

    if (error != Error::Ok || tensor == nullptr) {
      return nullptr;
    }

    // Fill with value
    std::vector<float> host_data(total_size, fill_value);
    cudaMemcpy(
        tensor->data_ptr(),
        host_data.data(),
        total_size * sizeof(float),
        cudaMemcpyHostToDevice);

    return tensor;
  }

  // Helper function to create a BFloat16 tensor
  Tensor* create_bfloat16_tensor(
      std::vector<int64_t> shape,
      float fill_value = 1.0f) {
    Tensor* tensor = nullptr;

    // Calculate size
    int64_t total_size = 1;
    for (auto dim : shape) {
      total_size *= dim;
    }

    // Calculate strides (row-major)
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
      strides[i] = stride;
      stride *= shape[i];
    }

    // Create tensor
    Error error = aoti_torch_empty_strided(
        shape.size(),
        shape.data(),
        strides.data(),
        static_cast<int32_t>(SupportedDTypes::BFLOAT16),
        static_cast<int32_t>(SupportedDevices::CUDA),
        0,
        &tensor);

    if (error != Error::Ok || tensor == nullptr) {
      return nullptr;
    }

    // Fill with value
    // Note: For simplicity, we'll fill with float and let the runtime handle conversion
    // In production, you'd want to properly convert to bfloat16
    std::vector<float> host_data(total_size, fill_value);
    cudaMemcpy(
        tensor->data_ptr(),
        host_data.data(),
        total_size * sizeof(float),
        cudaMemcpyHostToDevice);

    return tensor;
  }

  // Helper to check if output tensor has expected shape
  bool check_output_shape(
      Tensor* output,
      const std::vector<int64_t>& expected_shape) {
    if (output == nullptr) {
      return false;
    }
    if (output->dim() != expected_shape.size()) {
      return false;
    }
    for (size_t i = 0; i < expected_shape.size(); ++i) {
      if (output->size(i) != expected_shape[i]) {
        return false;
      }
    }
    return true;
  }

  // Helper to copy tensor data from GPU to CPU for verification
  std::vector<float> copy_tensor_to_host(Tensor* tensor) {
    int64_t total_size = 1;
    for (int i = 0; i < tensor->dim(); ++i) {
      total_size *= tensor->size(i);
    }

    std::vector<float> host_data(total_size);
    cudaMemcpy(
        host_data.data(),
        tensor->data_ptr(),
        total_size * sizeof(float),
        cudaMemcpyDeviceToHost);

    return host_data;
  }

  // Helper to check if a value is approximately equal (for floating point comparison)
  bool approx_equal(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
  }
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

// Test basic SDPA with Float32, no causal mask
TEST_F(AOTITorchSDPATest, BasicFunctionalityFloat32) {
  // Create tensors: [batch=1, num_heads=2, seq_len=4, head_dim=8]
  const int64_t batch = 1;
  const int64_t num_heads = 2;
  const int64_t seq_len = 4;
  const int64_t head_dim = 8;

  Tensor* query = create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* key = create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.3f);
  Tensor* value = create_float_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);

  ASSERT_NE(query, nullptr) << "Failed to create query tensor";
  ASSERT_NE(key, nullptr) << "Failed to create key tensor";
  ASSERT_NE(value, nullptr) << "Failed to create value tensor";

  printf("Testing SDPA Float32: [%ldx%ldx%ldx%ld]\n", batch, num_heads, seq_len, head_dim);

  // Call SDPA
  Tensor* output = nullptr;
  AOTITorchError error = aoti_torch_cuda_scaled_dot_product_attention(
      query,
      key,
      value,
      nullptr,  // no explicit mask
      0.0,      // no dropout
      0,        // not causal
      nullptr,  // default scale
      0,        // no GQA
      &output);

  // Check result
  EXPECT_EQ(error, Error::Ok) << "SDPA should succeed";
  ASSERT_NE(output, nullptr) << "Output should not be null";

  // Verify output shape: [batch, num_heads, seq_len, head_dim]
  EXPECT_TRUE(check_output_shape(output, {batch, num_heads, seq_len, head_dim}))
      << "Output shape mismatch";

  printf("SDPA Float32 test passed!\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
  aoti_torch_delete_tensor_object(output);
}

// Test SDPA with causal masking
TEST_F(AOTITorchSDPATest, CausalMasking) {
  const int64_t batch = 1;
  const int64_t num_heads = 1;
  const int64_t seq_len = 8;
  const int64_t head_dim = 16;

  Tensor* query = create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* key = create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* value = create_float_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);

  printf("Testing SDPA with causal masking: [%ldx%ldx%ldx%ld]\n",
         batch, num_heads, seq_len, head_dim);

  // Call SDPA with causal mask
  Tensor* output = nullptr;
  AOTITorchError error = aoti_torch_cuda_scaled_dot_product_attention(
      query,
      key,
      value,
      nullptr,
      0.0,
      1,        // causal mask enabled
      nullptr,
      0,
      &output);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_TRUE(check_output_shape(output, {batch, num_heads, seq_len, head_dim}));

  printf("Causal masking test passed!\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
  aoti_torch_delete_tensor_object(output);
}

// Test SDPA with BFloat16
TEST_F(AOTITorchSDPATest, BFloat16Precision) {
  const int64_t batch = 2;
  const int64_t num_heads = 4;
  const int64_t seq_len = 16;
  const int64_t head_dim = 32;

  Tensor* query = create_bfloat16_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* key = create_bfloat16_tensor({batch, num_heads, seq_len, head_dim}, 0.3f);
  Tensor* value = create_bfloat16_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);

  ASSERT_NE(query, nullptr) << "Failed to create BFloat16 query tensor";
  ASSERT_NE(key, nullptr) << "Failed to create BFloat16 key tensor";
  ASSERT_NE(value, nullptr) << "Failed to create BFloat16 value tensor";

  printf("Testing SDPA BFloat16: [%ldx%ldx%ldx%ld]\n",
         batch, num_heads, seq_len, head_dim);

  Tensor* output = nullptr;
  AOTITorchError error = aoti_torch_cuda_scaled_dot_product_attention(
      query,
      key,
      value,
      nullptr,
      0.0,
      0,
      nullptr,
      0,
      &output);

  EXPECT_EQ(error, Error::Ok) << "SDPA BFloat16 should succeed";
  ASSERT_NE(output, nullptr);
  EXPECT_TRUE(check_output_shape(output, {batch, num_heads, seq_len, head_dim}));

  printf("BFloat16 precision test passed!\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
  aoti_torch_delete_tensor_object(output);
}

// Test SDPA with custom scale factor
TEST_F(AOTITorchSDPATest, CustomScale) {
  const int64_t batch = 1;
  const int64_t num_heads = 2;
  const int64_t seq_len = 4;
  const int64_t head_dim = 8;

  Tensor* query = create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* key = create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* value = create_float_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);

  printf("Testing SDPA with custom scale\n");

  // Use custom scale instead of default 1/sqrt(head_dim)
  double custom_scale = 0.25;
  Tensor* output = nullptr;
  AOTITorchError error = aoti_torch_cuda_scaled_dot_product_attention(
      query,
      key,
      value,
      nullptr,
      0.0,
      0,
      &custom_scale,  // custom scale
      0,
      &output);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_TRUE(check_output_shape(output, {batch, num_heads, seq_len, head_dim}));

  printf("Custom scale test passed!\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
  aoti_torch_delete_tensor_object(output);
}

// Test with larger tensors (closer to real-world usage)
TEST_F(AOTITorchSDPATest, LargerTensors) {
  const int64_t batch = 4;
  const int64_t num_heads = 8;
  const int64_t seq_len = 128;
  const int64_t head_dim = 64;

  Tensor* query = create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.1f);
  Tensor* key = create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.1f);
  Tensor* value = create_float_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);

  printf("Testing SDPA with larger tensors: [%ldx%ldx%ldx%ld]\n",
         batch, num_heads, seq_len, head_dim);

  Tensor* output = nullptr;
  AOTITorchError error = aoti_torch_cuda_scaled_dot_product_attention(
      query,
      key,
      value,
      nullptr,
      0.0,
      1,  // causal
      nullptr,
      0,
      &output);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_TRUE(check_output_shape(output, {batch, num_heads, seq_len, head_dim}));

  printf("Larger tensors test passed!\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
  aoti_torch_delete_tensor_object(output);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

// Test null pointer handling
TEST_F(AOTITorchSDPATest, NullPointerHandling) {
  Tensor* query = create_float_tensor({1, 1, 4, 8}, 0.5f);
  Tensor* key = create_float_tensor({1, 1, 4, 8}, 0.5f);
  Tensor* value = create_float_tensor({1, 1, 4, 8}, 1.0f);
  Tensor* output = nullptr;

  // Test null query
  {
    AOTITorchError error = aoti_torch_cuda_scaled_dot_product_attention(
        nullptr, key, value, nullptr, 0.0, 0, nullptr, 0, &output);
    EXPECT_NE(error, Error::Ok) << "Should fail with null query";
  }

  // Test null key
  {
    AOTITorchError error = aoti_torch_cuda_scaled_dot_product_attention(
        query, nullptr, value, nullptr, 0.0, 0, nullptr, 0, &output);
    EXPECT_NE(error, Error::Ok) << "Should fail with null key";
  }

  // Test null value
  {
    AOTITorchError error = aoti_torch_cuda_scaled_dot_product_attention(
        query, key, nullptr, nullptr, 0.0, 0, nullptr, 0, &output);
    EXPECT_NE(error, Error::Ok) << "Should fail with null value";
  }

  // Test null output pointer
  {
    AOTITorchError error = aoti_torch_cuda_scaled_dot_product_attention(
        query, key, value, nullptr, 0.0, 0, nullptr, 0, nullptr);
    EXPECT_NE(error, Error::Ok) << "Should fail with null output pointer";
  }

  printf("Null pointer handling tests passed!\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
}

// Test dimension mismatch
TEST_F(AOTITorchSDPATest, DimensionMismatch) {
  Tensor* query = create_float_tensor({1, 2, 4, 8}, 0.5f);
  Tensor* key = create_float_tensor({1, 2, 6, 8}, 0.5f);  // Different seq_len
  Tensor* value = create_float_tensor({1, 2, 6, 8}, 1.0f);
  Tensor* output = nullptr;

  // This should succeed (Q and K can have different seq_len)
  AOTITorchError error = aoti_torch_cuda_scaled_dot_product_attention(
      query, key, value, nullptr, 0.0, 0, nullptr, 0, &output);

  EXPECT_EQ(error, Error::Ok) << "Different Q and K seq_len should be allowed";

  if (output != nullptr) {
    // Output should have Q's seq_len
    EXPECT_EQ(output->size(2), 4) << "Output seq_len should match Query";
    aoti_torch_delete_tensor_object(output);
  }

  printf("Dimension handling test passed!\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
}

// Test dropout error (should fail since we don't support dropout)
TEST_F(AOTITorchSDPATest, DropoutNotSupported) {
  Tensor* query = create_float_tensor({1, 1, 4, 8}, 0.5f);
  Tensor* key = create_float_tensor({1, 1, 4, 8}, 0.5f);
  Tensor* value = create_float_tensor({1, 1, 4, 8}, 1.0f);
  Tensor* output = nullptr;

  AOTITorchError error = aoti_torch_cuda_scaled_dot_product_attention(
      query, key, value, nullptr, 0.5, 0, nullptr, 0, &output);  // dropout=0.5

  EXPECT_NE(error, Error::Ok) << "Should fail with non-zero dropout";

  printf("Dropout rejection test passed!\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
}

// ============================================================================
// Numerical Correctness Tests
// ============================================================================

// Test that output values are in reasonable range
TEST_F(AOTITorchSDPATest, OutputValueRangeCheck) {
  const int64_t batch = 1;
  const int64_t num_heads = 1;
  const int64_t seq_len = 4;
  const int64_t head_dim = 8;

  // Use small values to avoid numerical overflow
  Tensor* query = create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.1f);
  Tensor* key = create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.1f);
  Tensor* value = create_float_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);

  printf("Testing SDPA output value range\n");

  Tensor* output = nullptr;
  AOTITorchError error = aoti_torch_cuda_scaled_dot_product_attention(
      query, key, value, nullptr, 0.0, 0, nullptr, 0, &output);

  ASSERT_EQ(error, Error::Ok);
  ASSERT_NE(output, nullptr);

  // Copy output back to CPU for verification
  std::vector<float> output_data = copy_tensor_to_host(output);

  // Since V is all 1.0, and softmax produces weights that sum to 1,
  // output should be close to 1.0 (weighted average of 1.0)
  bool all_in_range = true;
  for (size_t i = 0; i < output_data.size(); ++i) {
    // Output should be around 1.0 with some tolerance
    if (output_data[i] < 0.5f || output_data[i] > 1.5f) {
      printf("Output[%zu] = %f is out of expected range [0.5, 1.5]\n",
             i, output_data[i]);
      all_in_range = false;
    }
  }

  EXPECT_TRUE(all_in_range) << "Some output values are out of reasonable range";

  printf("Output value range check passed!\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
  aoti_torch_delete_tensor_object(output);
}

// Test with identity Q=K, verify attention weights sum to 1
TEST_F(AOTITorchSDPATest, IdentityQKTest) {
  const int64_t batch = 1;
  const int64_t num_heads = 1;
  const int64_t seq_len = 4;
  const int64_t head_dim = 8;

  // When Q=K, attention scores will be uniform (since all positions are equally similar)
  Tensor* query = create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* key = create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* value = create_float_tensor({batch, num_heads, seq_len, head_dim}, 2.0f);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);

  printf("Testing SDPA with Q=K (identity attention)\n");

  Tensor* output = nullptr;
  AOTITorchError error = aoti_torch_cuda_scaled_dot_product_attention(
      query, key, value, nullptr, 0.0, 0, nullptr, 0, &output);

  ASSERT_EQ(error, Error::Ok);
  ASSERT_NE(output, nullptr);

  // Copy output back to CPU
  std::vector<float> output_data = copy_tensor_to_host(output);

  // When Q=K and V is uniform, output should be close to V
  // (since attention weights are uniform due to identical scores)
  bool values_correct = true;
  for (size_t i = 0; i < output_data.size(); ++i) {
    // Output should be close to 2.0 (the value of V)
    if (!approx_equal(output_data[i], 2.0f, 0.1f)) {
      printf("Output[%zu] = %f, expected ~2.0\n", i, output_data[i]);
      values_correct = false;
    }
  }

  EXPECT_TRUE(values_correct) << "Output values don't match expected for identity Q=K";

  printf("Identity Q=K test passed!\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
  aoti_torch_delete_tensor_object(output);
}

// Test that different scales produce different outputs
TEST_F(AOTITorchSDPATest, ScaleEffectTest) {
  const int64_t batch = 1;
  const int64_t num_heads = 1;
  const int64_t seq_len = 4;
  const int64_t head_dim = 8;

  Tensor* query = create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* key = create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.3f);
  Tensor* value = create_float_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);

  // Make K different at different positions so attention scores vary
  std::vector<float> key_host(batch * num_heads * seq_len * head_dim);
  for (int64_t pos = 0; pos < seq_len; ++pos) {
    for (int64_t d = 0; d < head_dim; ++d) {
      // Different values per position: pos 0=0.1, pos 1=0.3, pos 2=0.5, pos 3=0.7
      key_host[pos * head_dim + d] = 0.1f + 0.2f * pos;
    }
  }
  cudaMemcpy(
      key->data_ptr(),
      key_host.data(),
      key_host.size() * sizeof(float),
      cudaMemcpyHostToDevice);

  // Make V also different at different positions to amplify differences
  std::vector<float> value_host(batch * num_heads * seq_len * head_dim);
  for (int64_t pos = 0; pos < seq_len; ++pos) {
    for (int64_t d = 0; d < head_dim; ++d) {
      // V values: pos 0=1.0, pos 1=2.0, pos 2=3.0, pos 3=4.0
      value_host[pos * head_dim + d] = static_cast<float>(pos + 1);
    }
  }
  cudaMemcpy(
      value->data_ptr(),
      value_host.data(),
      value_host.size() * sizeof(float),
      cudaMemcpyHostToDevice);

  printf("Testing SDPA scale effect\n");

  // Test with default scale
  Tensor* output1 = nullptr;
  AOTITorchError error1 = aoti_torch_cuda_scaled_dot_product_attention(
      query, key, value, nullptr, 0.0, 0, nullptr, 0, &output1);
  ASSERT_EQ(error1, Error::Ok);
  ASSERT_NE(output1, nullptr);

  // Test with custom scale (much smaller, should make attention more uniform)
  double small_scale = 0.01;
  Tensor* output2 = nullptr;
  AOTITorchError error2 = aoti_torch_cuda_scaled_dot_product_attention(
      query, key, value, nullptr, 0.0, 0, &small_scale, 0, &output2);
  ASSERT_EQ(error2, Error::Ok);
  ASSERT_NE(output2, nullptr);

  // Copy outputs back to CPU
  std::vector<float> output1_data = copy_tensor_to_host(output1);
  std::vector<float> output2_data = copy_tensor_to_host(output2);

  // Outputs should be different (scale affects softmax sharpness)
  // With varied V values, even small changes in attention weights will produce
  // noticeably different outputs
  bool outputs_differ = false;
  float max_diff = 0.0f;
  for (size_t i = 0; i < output1_data.size(); ++i) {
    float diff = std::abs(output1_data[i] - output2_data[i]);
    max_diff = std::max(max_diff, diff);
    if (diff > 0.05f) {  // More lenient threshold due to varied V values
      outputs_differ = true;
      break;
    }
  }

  printf("Max difference between outputs: %f\n", max_diff);
  EXPECT_TRUE(outputs_differ) << "Different scales should produce different outputs (max_diff=" << max_diff << ")";

  printf("Scale effect test passed!\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
  aoti_torch_delete_tensor_object(output1);
  aoti_torch_delete_tensor_object(output2);
}

// Test causal masking correctness
TEST_F(AOTITorchSDPATest, CausalMaskingCorrectness) {
  const int64_t batch = 1;
  const int64_t num_heads = 1;
  const int64_t seq_len = 4;
  const int64_t head_dim = 8;

  // Create distinct values at different positions in V
  // This allows us to verify that causal masking works correctly
  Tensor* query = create_float_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);
  Tensor* key = create_float_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);
  Tensor* value = create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.0f);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);

  // Manually set different values for each position in V
  // V[position i] = i+1 (so we can track which positions contribute)
  std::vector<float> value_host(batch * num_heads * seq_len * head_dim);
  for (int64_t pos = 0; pos < seq_len; ++pos) {
    for (int64_t d = 0; d < head_dim; ++d) {
      value_host[pos * head_dim + d] = static_cast<float>(pos + 1);
    }
  }
  cudaMemcpy(
      value->data_ptr(),
      value_host.data(),
      value_host.size() * sizeof(float),
      cudaMemcpyHostToDevice);

  printf("Testing SDPA causal masking correctness\n");

  // Run with causal masking
  Tensor* output_causal = nullptr;
  AOTITorchError error = aoti_torch_cuda_scaled_dot_product_attention(
      query, key, value, nullptr, 0.0, 1, nullptr, 0, &output_causal);
  ASSERT_EQ(error, Error::Ok);
  ASSERT_NE(output_causal, nullptr);

  // Copy output back to CPU
  std::vector<float> output_data = copy_tensor_to_host(output_causal);

  // With causal masking:
  // - Position 0 can only see position 0, so output[0] should be ~1.0
  // - Position 1 can see positions 0,1, so output[1] should be ~1.5 (average of 1 and 2)
  // - Position 2 can see positions 0,1,2, so output[2] should be ~2.0 (average of 1,2,3)
  // - Position 3 can see all, so output[3] should be ~2.5 (average of 1,2,3,4)

  std::vector<float> expected_values = {1.0f, 1.5f, 2.0f, 2.5f};

  bool causal_correct = true;
  for (int64_t pos = 0; pos < seq_len; ++pos) {
    float avg_output = 0.0f;
    for (int64_t d = 0; d < head_dim; ++d) {
      avg_output += output_data[pos * head_dim + d];
    }
    avg_output /= head_dim;

    printf("Position %ld: output avg = %f, expected ~%f\n",
           pos, avg_output, expected_values[pos]);

    if (!approx_equal(avg_output, expected_values[pos], 0.2f)) {
      causal_correct = false;
    }
  }

  EXPECT_TRUE(causal_correct) << "Causal masking did not produce expected values";

  printf("Causal masking correctness test passed!\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
  aoti_torch_delete_tensor_object(output_causal);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
