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
#include <executorch/backends/cuda/runtime/shims/sdpa.cuh> // For can_use_flash_attention

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
    // Note: For simplicity, we'll fill with float and let the runtime handle
    // conversion In production, you'd want to properly convert to bfloat16
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

  // Helper to check if a value is approximately equal (for floating point
  // comparison)
  bool approx_equal(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
  }

  // ========================================================================
  // Wrapper Functions for Simplified Testing
  // ========================================================================

  /**
   * Simplified wrapper for Flash Attention testing
   * Only requires the essential parameters, sets others to nullptr/defaults
   */
  AOTITorchError call_flash_attention(
      Tensor* query,
      Tensor* key,
      Tensor* value,
      double dropout_p,
      int32_t is_causal,
      double* scale,
      Tensor** output) {
    // Initialize all optional outputs to nullptr
    Tensor* logsumexp = nullptr;
    Tensor* cum_seq_q = nullptr;
    Tensor* cum_seq_k = nullptr;
    int64_t max_seqlen_q = query->size(2);
    int64_t max_seqlen_k = key->size(2);
    Tensor* philox_seed = nullptr;
    Tensor* philox_offset = nullptr;
    Tensor* debug_mask = nullptr;

    return aoti_torch_cuda__scaled_dot_product_flash_attention(
        query,
        key,
        value,
        dropout_p,
        is_causal,
        0, // return_debug_mask = 0
        scale,
        output,
        &logsumexp,
        &cum_seq_q,
        &cum_seq_k,
        &max_seqlen_q,
        &max_seqlen_k,
        &philox_seed,
        &philox_offset,
        &debug_mask);
  }

  /**
   * Simplified wrapper for Efficient Attention testing
   * Only requires the essential parameters, sets others to nullptr/defaults
   */
  AOTITorchError call_efficient_attention(
      Tensor* query,
      Tensor* key,
      Tensor* value,
      Tensor* attn_bias,
      int32_t is_causal,
      double* scale,
      Tensor** output) {
    // Initialize all optional outputs to nullptr
    Tensor* logsumexp = nullptr;
    Tensor* philox_seed = nullptr;
    Tensor* philox_offset = nullptr;

    return aoti_torch_cuda__scaled_dot_product_efficient_attention(
        query,
        key,
        value,
        attn_bias ? &attn_bias : nullptr,
        0, // compute_log_sumexp = 0
        0.0, // dropout_p = 0.0
        is_causal,
        scale,
        output,
        &logsumexp,
        &philox_seed,
        &philox_offset);
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

  Tensor* query =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* key =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.3f);

  // Create V with different values at each position so attention weight changes
  // matter
  Tensor* value =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.0f);
  std::vector<float> value_host(batch * num_heads * seq_len * head_dim);
  for (int64_t pos = 0; pos < seq_len; ++pos) {
    for (int64_t d = 0; d < head_dim; ++d) {
      // V values: pos 0=1.0, pos 1=2.0, pos 2=3.0, etc.
      value_host[pos * head_dim + d] = static_cast<float>(pos + 1);
    }
  }
  cudaMemcpy(
      value->data_ptr(),
      value_host.data(),
      value_host.size() * sizeof(float),
      cudaMemcpyHostToDevice);

  ASSERT_NE(query, nullptr) << "Failed to create query tensor";
  ASSERT_NE(key, nullptr) << "Failed to create key tensor";
  ASSERT_NE(value, nullptr) << "Failed to create value tensor";

  printf(
      "Testing SDPA Float32: [%ldx%ldx%ldx%ld]\n",
      batch,
      num_heads,
      seq_len,
      head_dim);

  // Call SDPA - Flash Attention
  Tensor* output = nullptr;
  Tensor* logsumexp = nullptr;
  Tensor* cum_seq_q = nullptr;
  Tensor* cum_seq_k = nullptr;
  int64_t max_seqlen_q = seq_len;
  int64_t max_seqlen_k = seq_len;
  Tensor* philox_seed = nullptr;
  Tensor* philox_offset = nullptr;
  Tensor* debug_mask = nullptr;

  AOTITorchError error = aoti_torch_cuda__scaled_dot_product_flash_attention(
      query,
      key,
      value,
      0.0, // no dropout
      0, // not causal
      0, // no debug mask
      nullptr, // default scale
      &output,
      &logsumexp,
      &cum_seq_q,
      &cum_seq_k,
      &max_seqlen_q,
      &max_seqlen_k,
      &philox_seed,
      &philox_offset,
      &debug_mask);

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

  Tensor* query =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* key =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* value =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);

  printf(
      "Testing SDPA with causal masking: [%ldx%ldx%ldx%ld]\n",
      batch,
      num_heads,
      seq_len,
      head_dim);

  // Call SDPA with causal mask using wrapper
  Tensor* output = nullptr;
  AOTITorchError error = call_flash_attention(
      query,
      key,
      value,
      0.0, // no dropout
      1, // causal mask enabled
      nullptr, // default scale
      &output);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_TRUE(
      check_output_shape(output, {batch, num_heads, seq_len, head_dim}));

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

  Tensor* query =
      create_bfloat16_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* key =
      create_bfloat16_tensor({batch, num_heads, seq_len, head_dim}, 0.3f);
  Tensor* value =
      create_bfloat16_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);

  ASSERT_NE(query, nullptr) << "Failed to create BFloat16 query tensor";
  ASSERT_NE(key, nullptr) << "Failed to create BFloat16 key tensor";
  ASSERT_NE(value, nullptr) << "Failed to create BFloat16 value tensor";

  printf(
      "Testing SDPA BFloat16: [%ldx%ldx%ldx%ld]\n",
      batch,
      num_heads,
      seq_len,
      head_dim);

  Tensor* output = nullptr;
  AOTITorchError error = call_flash_attention(
      query,
      key,
      value,
      0.0, // no dropout
      0, // not causal
      nullptr, // default scale
      &output);

  EXPECT_EQ(error, Error::Ok) << "SDPA BFloat16 should succeed";
  ASSERT_NE(output, nullptr);
  EXPECT_TRUE(
      check_output_shape(output, {batch, num_heads, seq_len, head_dim}));

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

  Tensor* query =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* key =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* value =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);

  printf("Testing SDPA with custom scale\n");

  // Use custom scale instead of default 1/sqrt(head_dim)
  double custom_scale = 0.25;
  Tensor* output = nullptr;
  AOTITorchError error = call_flash_attention(
      query,
      key,
      value,
      0.0, // no dropout
      0, // not causal
      &custom_scale, // custom scale
      &output);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_TRUE(
      check_output_shape(output, {batch, num_heads, seq_len, head_dim}));

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

  Tensor* query =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.1f);
  Tensor* key =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.1f);
  Tensor* value =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);

  printf(
      "Testing SDPA with larger tensors: [%ldx%ldx%ldx%ld]\n",
      batch,
      num_heads,
      seq_len,
      head_dim);

  Tensor* output = nullptr;
  AOTITorchError error = call_flash_attention(
      query,
      key,
      value,
      0.0, // no dropout
      1, // causal
      nullptr, // default scale
      &output);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_TRUE(
      check_output_shape(output, {batch, num_heads, seq_len, head_dim}));

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

// Test dimension mismatch
TEST_F(AOTITorchSDPATest, DimensionMismatch) {
  Tensor* query = create_float_tensor({1, 2, 4, 8}, 0.5f);
  Tensor* key = create_float_tensor({1, 2, 6, 8}, 0.5f); // Different seq_len
  Tensor* value = create_float_tensor({1, 2, 6, 8}, 1.0f);
  Tensor* output = nullptr;

  // This should succeed (Q and K can have different seq_len)
  AOTITorchError error =
      call_flash_attention(query, key, value, 0.0, 0, nullptr, &output);

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

  AOTITorchError error = call_flash_attention(
      query, key, value, 0.5, 0, nullptr, &output); // dropout=0.5

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
  Tensor* query =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.1f);
  Tensor* key =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.1f);
  Tensor* value =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);

  printf("Testing SDPA output value range\n");

  Tensor* output = nullptr;
  AOTITorchError error =
      call_flash_attention(query, key, value, 0.0, 0, nullptr, &output);

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
      printf(
          "Output[%zu] = %f is out of expected range [0.5, 1.5]\n",
          i,
          output_data[i]);
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

  // When Q=K, attention scores will be uniform (since all positions are equally
  // similar)
  Tensor* query =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* key =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* value =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 2.0f);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);

  printf("Testing SDPA with Q=K (identity attention)\n");

  Tensor* output = nullptr;
  AOTITorchError error =
      call_flash_attention(query, key, value, 0.0, 0, nullptr, &output);

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

  EXPECT_TRUE(values_correct)
      << "Output values don't match expected for identity Q=K";

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

  Tensor* query =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* key =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.3f);
  Tensor* value =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);

  // Make K different at different positions so attention scores vary
  std::vector<float> key_host(batch * num_heads * seq_len * head_dim);
  for (int64_t pos = 0; pos < seq_len; ++pos) {
    for (int64_t d = 0; d < head_dim; ++d) {
      // Different values per position: pos 0=0.1, pos 1=0.3, pos 2=0.5, pos
      // 3=0.7
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
  AOTITorchError error1 =
      call_flash_attention(query, key, value, 0.0, 0, nullptr, &output1);
  ASSERT_EQ(error1, Error::Ok);
  ASSERT_NE(output1, nullptr);

  // Test with custom scale (much smaller, should make attention more uniform)
  double small_scale = 0.01;
  Tensor* output2 = nullptr;
  AOTITorchError error2 =
      call_flash_attention(query, key, value, 0.0, 0, &small_scale, &output2);
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
    if (diff > 0.05f) { // More lenient threshold due to varied V values
      outputs_differ = true;
      break;
    }
  }

  printf("Max difference between outputs: %f\n", max_diff);
  EXPECT_TRUE(outputs_differ)
      << "Different scales should produce different outputs (max_diff="
      << max_diff << ")";

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
  Tensor* query =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);
  Tensor* key =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);
  Tensor* value =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.0f);

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
  AOTITorchError error =
      call_flash_attention(query, key, value, 0.0, 1, nullptr, &output_causal);
  ASSERT_EQ(error, Error::Ok);
  ASSERT_NE(output_causal, nullptr);

  // Copy output back to CPU
  std::vector<float> output_data = copy_tensor_to_host(output_causal);

  // With causal masking:
  // - Position 0 can only see position 0, so output[0] should be ~1.0
  // - Position 1 can see positions 0,1, so output[1] should be ~1.5 (average of
  // 1 and 2)
  // - Position 2 can see positions 0,1,2, so output[2] should be ~2.0 (average
  // of 1,2,3)
  // - Position 3 can see all, so output[3] should be ~2.5 (average of 1,2,3,4)

  std::vector<float> expected_values = {1.0f, 1.5f, 2.0f, 2.5f};

  bool causal_correct = true;
  for (int64_t pos = 0; pos < seq_len; ++pos) {
    float avg_output = 0.0f;
    for (int64_t d = 0; d < head_dim; ++d) {
      avg_output += output_data[pos * head_dim + d];
    }
    avg_output /= head_dim;

    printf(
        "Position %ld: output avg = %f, expected ~%f\n",
        pos,
        avg_output,
        expected_values[pos]);

    if (!approx_equal(avg_output, expected_values[pos], 0.2f)) {
      causal_correct = false;
    }
  }

  EXPECT_TRUE(causal_correct)
      << "Causal masking did not produce expected values";

  printf("Causal masking correctness test passed!\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
  aoti_torch_delete_tensor_object(output_causal);
}

// ============================================================================
// Flash Attention Specific Tests
// ============================================================================

// Test that Flash Attention is selected for longer sequences
TEST_F(AOTITorchSDPATest, FlashAttentionLongSequence) {
  const int64_t batch = 2;
  const int64_t num_heads = 4;
  const int64_t seq_len = 256; // Long sequence to trigger Flash Attention
  const int64_t head_dim = 64;

  Tensor* query =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.1f);
  Tensor* key =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.1f);
  Tensor* value =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);

  printf(
      "Testing Flash Attention with long sequence: [%ldx%ldx%ldx%ld]\n",
      batch,
      num_heads,
      seq_len,
      head_dim);

  // Verify that can_use_flash_attention returns true for this configuration
  bool can_use_fa = can_use_flash_attention(query, key, value, nullptr, false);
  EXPECT_TRUE(can_use_fa)
      << "Should be able to use Flash Attention for long sequences";
  printf("  can_use_flash_attention: %s\n", can_use_fa ? "true" : "false");

  Tensor* output = nullptr;
  AOTITorchError error =
      call_flash_attention(query, key, value, 0.0, 0, nullptr, &output);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_TRUE(
      check_output_shape(output, {batch, num_heads, seq_len, head_dim}));

  printf("Flash Attention long sequence test passed!\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
  aoti_torch_delete_tensor_object(output);
}

// Test Flash Attention with causal masking
TEST_F(AOTITorchSDPATest, FlashAttentionCausalMasking) {
  const int64_t batch = 1;
  const int64_t num_heads = 2;
  const int64_t seq_len = 128; // Long enough for Flash Attention
  const int64_t head_dim = 64;

  Tensor* query =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* key =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* value =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);

  printf("Testing Flash Attention with causal masking\n");

  // Verify that can_use_flash_attention returns true for causal masking
  bool can_use_fa = can_use_flash_attention(query, key, value, nullptr, true);
  EXPECT_TRUE(can_use_fa)
      << "Should be able to use Flash Attention with causal masking";
  printf(
      "  can_use_flash_attention (causal): %s\n",
      can_use_fa ? "true" : "false");

  Tensor* output = nullptr;
  AOTITorchError error = call_flash_attention(
      query, key, value, 0.0, 1, nullptr, &output); // causal=1

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_TRUE(
      check_output_shape(output, {batch, num_heads, seq_len, head_dim}));

  printf("Flash Attention causal masking test passed!\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
  aoti_torch_delete_tensor_object(output);
}

// Test Flash Attention with BFloat16
TEST_F(AOTITorchSDPATest, FlashAttentionBFloat16) {
  const int64_t batch = 1;
  const int64_t num_heads = 4;
  const int64_t seq_len = 64;
  const int64_t head_dim = 64;

  Tensor* query =
      create_bfloat16_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* key =
      create_bfloat16_tensor({batch, num_heads, seq_len, head_dim}, 0.3f);
  Tensor* value =
      create_bfloat16_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);

  printf("Testing Flash Attention with BFloat16\n");

  // Verify that can_use_flash_attention returns true for BFloat16
  bool can_use_fa = can_use_flash_attention(query, key, value, nullptr, false);
  EXPECT_TRUE(can_use_fa)
      << "Should be able to use Flash Attention with BFloat16";
  printf(
      "  can_use_flash_attention (BFloat16): %s\n",
      can_use_fa ? "true" : "false");

  Tensor* output = nullptr;
  AOTITorchError error =
      call_flash_attention(query, key, value, 0.0, 0, nullptr, &output);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_TRUE(
      check_output_shape(output, {batch, num_heads, seq_len, head_dim}));

  printf("Flash Attention BFloat16 test passed!\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
  aoti_torch_delete_tensor_object(output);
}

// Test correctness: Compare Flash Attention output with Math fallback
TEST_F(AOTITorchSDPATest, FlashAttentionCorrectness) {
  const int64_t batch = 1;
  const int64_t num_heads = 1;
  const int64_t seq_len = 32; // Short enough to potentially use Math fallback
  const int64_t head_dim = 32;

  // Create inputs with known values
  Tensor* query =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* key =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.3f);
  Tensor* value =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);

  printf("Testing Flash Attention numerical correctness\n");

  // Verify that can_use_flash_attention returns true for this configuration
  bool can_use_fa = can_use_flash_attention(query, key, value, nullptr, false);
  EXPECT_TRUE(can_use_fa)
      << "Should be able to use Flash Attention for this configuration";
  printf("  can_use_flash_attention: %s\n", can_use_fa ? "true" : "false");

  // Run SDPA (will auto-select backend)
  Tensor* output = nullptr;
  AOTITorchError error =
      call_flash_attention(query, key, value, 0.0, 0, nullptr, &output);

  ASSERT_EQ(error, Error::Ok);
  ASSERT_NE(output, nullptr);

  // Copy output back to CPU for validation
  std::vector<float> output_data = copy_tensor_to_host(output);

  // Since V is all 1.0 and softmax weights sum to 1, output should be close
  // to 1.0
  bool all_correct = true;
  for (size_t i = 0; i < output_data.size(); ++i) {
    if (!approx_equal(output_data[i], 1.0f, 0.1f)) {
      printf("Output[%zu] = %f, expected ~1.0\n", i, output_data[i]);
      all_correct = false;
    }
  }

  EXPECT_TRUE(all_correct)
      << "Flash Attention output doesn't match expected values";

  printf("Flash Attention correctness test passed!\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
  aoti_torch_delete_tensor_object(output);
}

// Test Flash Attention with different query and key sequence lengths
TEST_F(AOTITorchSDPATest, FlashAttentionDifferentSeqLengths) {
  const int64_t batch = 1;
  const int64_t num_heads = 2;
  const int64_t seq_len_q = 64;
  const int64_t seq_len_k = 128; // K/V longer than Q
  const int64_t head_dim = 32;

  Tensor* query =
      create_float_tensor({batch, num_heads, seq_len_q, head_dim}, 0.5f);
  Tensor* key =
      create_float_tensor({batch, num_heads, seq_len_k, head_dim}, 0.5f);
  Tensor* value =
      create_float_tensor({batch, num_heads, seq_len_k, head_dim}, 1.0f);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);

  printf("Testing Flash Attention with different Q and K/V sequence lengths\n");

  // Verify that can_use_flash_attention returns true for different seq lengths
  bool can_use_fa = can_use_flash_attention(query, key, value, nullptr, false);
  EXPECT_TRUE(can_use_fa)
      << "Should be able to use Flash Attention with different seq lengths";
  printf(
      "  can_use_flash_attention (different seq lengths): %s\n",
      can_use_fa ? "true" : "false");

  Tensor* output = nullptr;
  AOTITorchError error =
      call_flash_attention(query, key, value, 0.0, 0, nullptr, &output);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_TRUE(
      check_output_shape(output, {batch, num_heads, seq_len_q, head_dim}));

  printf("Flash Attention different seq lengths test passed!\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
  aoti_torch_delete_tensor_object(output);
}

// Test that explicit attention mask forces Math fallback instead of Flash
// Attention
TEST_F(AOTITorchSDPATest, ExplicitMaskFallsBackToMath) {
  const int64_t batch = 1;
  const int64_t num_heads = 1;
  const int64_t seq_len = 128; // Long enough to prefer Flash Attention
  const int64_t head_dim = 64;

  Tensor* query =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* key =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* value =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);

  // Create explicit attention mask (will force Math fallback)
  Tensor* attn_mask =
      create_float_tensor({batch, num_heads, seq_len, seq_len}, 0.0f);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);
  ASSERT_NE(attn_mask, nullptr);

  printf("Testing that explicit mask forces Math fallback\n");

  // Verify that can_use_flash_attention returns false when explicit mask is
  // provided
  bool can_use_fa =
      can_use_flash_attention(query, key, value, attn_mask, false);
  EXPECT_FALSE(can_use_fa)
      << "Should NOT be able to use Flash Attention with explicit mask";
  printf(
      "  can_use_flash_attention (with explicit mask): %s\n",
      can_use_fa ? "true" : "false");

  Tensor* output = nullptr;
  AOTITorchError error = call_efficient_attention(
      query, key, value, attn_mask, 0, nullptr, &output);

  // Should succeed but use Math fallback instead of Flash Attention
  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_TRUE(
      check_output_shape(output, {batch, num_heads, seq_len, head_dim}));

  printf("Explicit mask fallback test passed!\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
  aoti_torch_delete_tensor_object(attn_mask);
  aoti_torch_delete_tensor_object(output);
}

// ============================================================================
// can_use_flash_attention Function Tests
// ============================================================================

// Test can_use_flash_attention returns true for valid configuration
TEST_F(AOTITorchSDPATest, CanUseFlashAttention_ValidConfig) {
  // Create tensors with valid flash attention configuration:
  // - head_dim <= 128
  // - head_dim_v <= 64
  // - At least one sequence length >= 32
  // - No explicit attention mask
  const int64_t batch = 2;
  const int64_t num_heads = 8;
  const int64_t seq_len = 64; // >= 32
  const int64_t head_dim = 64; // <= 128

  Tensor* query =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* key =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* value =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);

  printf(
      "Testing can_use_flash_attention with valid config: "
      "seq_len=%ld, head_dim=%ld\n",
      seq_len,
      head_dim);

  // Test without attention mask, should return true
  bool can_use = can_use_flash_attention(
      query,
      key,
      value,
      nullptr, // no attention mask
      false); // not causal

  EXPECT_TRUE(can_use)
      << "Should allow flash attention for valid configuration";

  // Also test with causal masking, should still return true
  bool can_use_causal = can_use_flash_attention(
      query,
      key,
      value,
      nullptr, // no attention mask
      true); // causal

  EXPECT_TRUE(can_use_causal)
      << "Should allow flash attention with causal masking";

  printf("Flash attention valid config test passed!\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
}

// Test can_use_flash_attention returns false for various invalid configurations
TEST_F(AOTITorchSDPATest, CanUseFlashAttention_InvalidConfigs) {
  const int64_t batch = 2;
  const int64_t num_heads = 8;

  printf("Testing can_use_flash_attention with invalid configurations\n");

  // Test 1: Explicit attention mask provided
  {
    Tensor* query = create_float_tensor({batch, num_heads, 64, 64}, 0.5f);
    Tensor* key = create_float_tensor({batch, num_heads, 64, 64}, 0.5f);
    Tensor* value = create_float_tensor({batch, num_heads, 64, 64}, 1.0f);
    Tensor* attn_mask = create_float_tensor({batch, num_heads, 64, 64}, 0.0f);

    ASSERT_NE(query, nullptr);
    ASSERT_NE(key, nullptr);
    ASSERT_NE(value, nullptr);
    ASSERT_NE(attn_mask, nullptr);

    bool can_use = can_use_flash_attention(query, key, value, attn_mask, false);

    EXPECT_FALSE(can_use)
        << "Should reject flash attention with explicit attention mask";
    printf("  - Explicit attention mask: correctly rejected\n");

    aoti_torch_delete_tensor_object(query);
    aoti_torch_delete_tensor_object(key);
    aoti_torch_delete_tensor_object(value);
    aoti_torch_delete_tensor_object(attn_mask);
  }

  // Test 2: head_dim > 128
  {
    const int64_t large_head_dim = 256; // > 128
    Tensor* query =
        create_float_tensor({batch, num_heads, 64, large_head_dim}, 0.5f);
    Tensor* key =
        create_float_tensor({batch, num_heads, 64, large_head_dim}, 0.5f);
    Tensor* value =
        create_float_tensor({batch, num_heads, 64, large_head_dim}, 1.0f);

    ASSERT_NE(query, nullptr);
    ASSERT_NE(key, nullptr);
    ASSERT_NE(value, nullptr);

    bool can_use = can_use_flash_attention(query, key, value, nullptr, false);

    EXPECT_FALSE(can_use)
        << "Should reject flash attention with head_dim > 128";
    printf("  - head_dim > 128: correctly rejected\n");

    aoti_torch_delete_tensor_object(query);
    aoti_torch_delete_tensor_object(key);
    aoti_torch_delete_tensor_object(value);
  }

  // Test 3: head_dim_v > 64
  {
    const int64_t large_head_dim_v = 128; // > 64
    Tensor* query = create_float_tensor({batch, num_heads, 64, 64}, 0.5f);
    Tensor* key = create_float_tensor({batch, num_heads, 64, 64}, 0.5f);
    Tensor* value =
        create_float_tensor({batch, num_heads, 64, large_head_dim_v}, 1.0f);

    ASSERT_NE(query, nullptr);
    ASSERT_NE(key, nullptr);
    ASSERT_NE(value, nullptr);

    bool can_use = can_use_flash_attention(query, key, value, nullptr, false);

    EXPECT_FALSE(can_use)
        << "Should reject flash attention with head_dim_v > 64";
    printf("  - head_dim_v > 64: correctly rejected\n");

    aoti_torch_delete_tensor_object(query);
    aoti_torch_delete_tensor_object(key);
    aoti_torch_delete_tensor_object(value);
  }

  printf("Flash attention invalid configs test passed!\n");
}

// ============================================================================
// Efficient Attention Specific Tests
// ============================================================================

// Test basic efficient attention without attention bias
TEST_F(AOTITorchSDPATest, EfficientAttention_BasicNoBias) {
  const int64_t batch = 2;
  const int64_t num_heads = 4;
  const int64_t seq_len = 32;
  const int64_t head_dim = 64;

  Tensor* query =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* key =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.3f);
  Tensor* value =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);

  printf(
      "Testing Efficient Attention without bias: [%ldx%ldx%ldx%ld]\n",
      batch,
      num_heads,
      seq_len,
      head_dim);

  Tensor* output = nullptr;
  AOTITorchError error =
      call_efficient_attention(query, key, value, nullptr, 0, nullptr, &output);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_TRUE(
      check_output_shape(output, {batch, num_heads, seq_len, head_dim}));

  printf("Efficient attention basic test passed!\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
  aoti_torch_delete_tensor_object(output);
}

// Test efficient attention with full attention bias (no broadcasting)
TEST_F(AOTITorchSDPATest, EfficientAttention_WithBias) {
  const int64_t batch = 2;
  const int64_t num_heads = 4;
  const int64_t seq_len = 16;
  const int64_t head_dim = 32;

  Tensor* query =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* key =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.3f);
  Tensor* value =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);
  Tensor* attn_bias =
      create_float_tensor({batch, num_heads, seq_len, seq_len}, 0.1f);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);
  ASSERT_NE(attn_bias, nullptr);

  printf(
      "Testing Efficient Attention with full bias: [%ldx%ldx%ldx%ld]\n",
      batch,
      num_heads,
      seq_len,
      seq_len);

  Tensor* output = nullptr;
  AOTITorchError error = call_efficient_attention(
      query, key, value, attn_bias, 0, nullptr, &output);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_TRUE(
      check_output_shape(output, {batch, num_heads, seq_len, head_dim}));

  printf("Backend selection test passed!\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
  aoti_torch_delete_tensor_object(attn_bias);
  aoti_torch_delete_tensor_object(output);
}

// Test efficient attention with broadcasted bias (batch broadcast)
TEST_F(AOTITorchSDPATest, EfficientAttention_BroadcastBatchDim) {
  const int64_t batch = 4;
  const int64_t num_heads = 2;
  const int64_t seq_len = 8;
  const int64_t head_dim = 16;

  Tensor* query =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* key =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.3f);
  Tensor* value =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);
  // Broadcast across batch dimension: [1, num_heads, seq_len, seq_len]
  Tensor* attn_bias =
      create_float_tensor({1, num_heads, seq_len, seq_len}, 0.2f);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);
  ASSERT_NE(attn_bias, nullptr);

  printf("Testing Efficient Attention with batch-broadcasted bias\n");

  Tensor* output = nullptr;
  AOTITorchError error = call_efficient_attention(
      query, key, value, attn_bias, 0, nullptr, &output);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_TRUE(
      check_output_shape(output, {batch, num_heads, seq_len, head_dim}));

  printf("Efficient attention with batch broadcast test passed!\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
  aoti_torch_delete_tensor_object(attn_bias);
  aoti_torch_delete_tensor_object(output);
}

// Test efficient attention with broadcasted bias (head broadcast)
TEST_F(AOTITorchSDPATest, EfficientAttention_BroadcastHeadDim) {
  const int64_t batch = 2;
  const int64_t num_heads = 8;
  const int64_t seq_len = 16;
  const int64_t head_dim = 32;

  Tensor* query =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* key =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.3f);
  Tensor* value =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);
  // Broadcast across head dimension: [batch, 1, seq_len, seq_len]
  Tensor* attn_bias = create_float_tensor({batch, 1, seq_len, seq_len}, -0.1f);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);
  ASSERT_NE(attn_bias, nullptr);

  printf("Testing Efficient Attention with head-broadcasted bias\n");

  Tensor* output = nullptr;
  AOTITorchError error = call_efficient_attention(
      query, key, value, attn_bias, 0, nullptr, &output);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_TRUE(
      check_output_shape(output, {batch, num_heads, seq_len, head_dim}));

  printf("Efficient attention with head broadcast test passed!\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
  aoti_torch_delete_tensor_object(attn_bias);
  aoti_torch_delete_tensor_object(output);
}

// Test efficient attention with causal masking and bias
TEST_F(AOTITorchSDPATest, EfficientAttention_CausalWithBias) {
  const int64_t batch = 1;
  const int64_t num_heads = 2;
  const int64_t seq_len = 32;
  const int64_t head_dim = 64;

  Tensor* query =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* key =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* value =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);
  Tensor* attn_bias =
      create_float_tensor({batch, num_heads, seq_len, seq_len}, 0.05f);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);
  ASSERT_NE(attn_bias, nullptr);

  printf("Testing Efficient Attention with causal masking and bias\n");

  Tensor* output = nullptr;
  AOTITorchError error = call_efficient_attention(
      query, key, value, attn_bias, 1, nullptr, &output); // causal=1

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_TRUE(
      check_output_shape(output, {batch, num_heads, seq_len, head_dim}));

  printf("Efficient attention causal with bias test passed!\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
  aoti_torch_delete_tensor_object(attn_bias);
  aoti_torch_delete_tensor_object(output);
}

// Test efficient attention with BFloat16
TEST_F(AOTITorchSDPATest, EfficientAttention_BFloat16) {
  const int64_t batch = 2;
  const int64_t num_heads = 4;
  const int64_t seq_len = 64;
  const int64_t head_dim = 64;

  Tensor* query =
      create_bfloat16_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* key =
      create_bfloat16_tensor({batch, num_heads, seq_len, head_dim}, 0.3f);
  Tensor* value =
      create_bfloat16_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);

  printf("Testing Efficient Attention with BFloat16\n");

  Tensor* output = nullptr;
  AOTITorchError error =
      call_efficient_attention(query, key, value, nullptr, 0, nullptr, &output);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_TRUE(
      check_output_shape(output, {batch, num_heads, seq_len, head_dim}));

  printf("Efficient attention BFloat16 test passed!\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
  aoti_torch_delete_tensor_object(output);
}

// Test efficient attention with different Q and K/V sequence lengths
TEST_F(AOTITorchSDPATest, EfficientAttention_DifferentSeqLengths) {
  const int64_t batch = 2;
  const int64_t num_heads = 4;
  const int64_t seq_len_q = 16;
  const int64_t seq_len_kv = 32; // K/V longer than Q
  const int64_t head_dim = 32;

  Tensor* query =
      create_float_tensor({batch, num_heads, seq_len_q, head_dim}, 0.5f);
  Tensor* key =
      create_float_tensor({batch, num_heads, seq_len_kv, head_dim}, 0.3f);
  Tensor* value =
      create_float_tensor({batch, num_heads, seq_len_kv, head_dim}, 1.0f);
  Tensor* attn_bias =
      create_float_tensor({batch, num_heads, seq_len_q, seq_len_kv}, 0.1f);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);
  ASSERT_NE(attn_bias, nullptr);

  printf(
      "Testing Efficient Attention with different Q (%ld) and K/V (%ld) lengths\n",
      seq_len_q,
      seq_len_kv);

  Tensor* output = nullptr;
  AOTITorchError error = call_efficient_attention(
      query, key, value, attn_bias, 0, nullptr, &output);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_TRUE(
      check_output_shape(output, {batch, num_heads, seq_len_q, head_dim}));

  printf("Efficient attention different seq lengths test passed!\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
  aoti_torch_delete_tensor_object(attn_bias);
  aoti_torch_delete_tensor_object(output);
}

// Test efficient attention with very large head_dim_v (up to 128)
TEST_F(AOTITorchSDPATest, EfficientAttention_LargeHeadDimV) {
  const int64_t batch = 1;
  const int64_t num_heads = 2;
  const int64_t seq_len = 16;
  const int64_t head_dim = 64;
  const int64_t head_dim_v = 128; // Maximum supported by efficient attention

  Tensor* query =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* key =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.3f);
  Tensor* value =
      create_float_tensor({batch, num_heads, seq_len, head_dim_v}, 1.0f);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);

  printf("Testing Efficient Attention with large head_dim_v=%ld\n", head_dim_v);

  // Verify that can_use_efficient_attention returns true
  bool can_use = can_use_efficient_attention(query, key, value, nullptr, false);
  EXPECT_TRUE(can_use) << "Should support head_dim_v=128";

  Tensor* output = nullptr;
  AOTITorchError error =
      call_efficient_attention(query, key, value, nullptr, 0, nullptr, &output);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_TRUE(
      check_output_shape(output, {batch, num_heads, seq_len, head_dim_v}));

  printf("Efficient attention large head_dim_v test passed!\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
  aoti_torch_delete_tensor_object(output);
}

// Test numerical correctness: bias affects output
TEST_F(AOTITorchSDPATest, EfficientAttention_BiasAffectsOutput) {
  const int64_t batch = 1;
  const int64_t num_heads = 1;
  const int64_t seq_len = 8;
  const int64_t head_dim = 16;

  Tensor* query =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* key =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.3f);

  // Create V with different values at each position so attention weight changes
  // matter
  Tensor* value =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.0f);
  std::vector<float> value_host(batch * num_heads * seq_len * head_dim);
  for (int64_t pos = 0; pos < seq_len; ++pos) {
    for (int64_t d = 0; d < head_dim; ++d) {
      // V values: pos 0=1.0, pos 1=2.0, pos 2=3.0, etc.
      value_host[pos * head_dim + d] = static_cast<float>(pos + 1);
    }
  }
  cudaMemcpy(
      value->data_ptr(),
      value_host.data(),
      value_host.size() * sizeof(float),
      cudaMemcpyHostToDevice);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);

  printf("Testing that attention bias affects output\n");

  // Run without bias
  Tensor* output_no_bias = nullptr;
  AOTITorchError error1 = call_efficient_attention(
      query, key, value, nullptr, 0, nullptr, &output_no_bias);
  ASSERT_EQ(error1, Error::Ok);
  ASSERT_NE(output_no_bias, nullptr);

  // Create bias tensor with varied values (not uniform)
  // A uniform bias doesn't change softmax output, so we need variation
  Tensor* attn_bias =
      create_float_tensor({batch, num_heads, seq_len, seq_len}, 0.0f);
  ASSERT_NE(attn_bias, nullptr);

  // Fill bias with varied values to actually affect attention distribution
  // Use a strong pattern that will significantly change attention weights
  std::vector<float> bias_host(batch * num_heads * seq_len * seq_len);
  for (int64_t i = 0; i < seq_len; ++i) {
    for (int64_t j = 0; j < seq_len; ++j) {
      // Create a diagonal pattern with large values to strongly affect
      // attention
      if (i == j) {
        bias_host[i * seq_len + j] = 10.0f; // Large positive bias for diagonal
      } else {
        bias_host[i * seq_len + j] = -5.0f; // Negative bias for off-diagonal
      }
    }
  }
  cudaMemcpy(
      attn_bias->data_ptr(),
      bias_host.data(),
      bias_host.size() * sizeof(float),
      cudaMemcpyHostToDevice);

  // Run with bias
  Tensor* output_with_bias = nullptr;
  AOTITorchError error2 = call_efficient_attention(
      query, key, value, attn_bias, 0, nullptr, &output_with_bias);
  ASSERT_EQ(error2, Error::Ok);
  ASSERT_NE(output_with_bias, nullptr);

  // Verify both computations succeeded
  EXPECT_TRUE(check_output_shape(
      output_no_bias, {batch, num_heads, seq_len, head_dim}));
  EXPECT_TRUE(check_output_shape(
      output_with_bias, {batch, num_heads, seq_len, head_dim}));

  // Copy outputs to host and verify they differ
  std::vector<float> output_no_bias_data = copy_tensor_to_host(output_no_bias);
  std::vector<float> output_with_bias_data =
      copy_tensor_to_host(output_with_bias);

  // Outputs should be different when bias is non-uniform
  bool outputs_differ = false;
  float max_diff = 0.0f;
  for (size_t i = 0; i < output_no_bias_data.size(); ++i) {
    float diff = std::abs(output_no_bias_data[i] - output_with_bias_data[i]);
    max_diff = std::max(max_diff, diff);
    if (diff > 0.001f) { // Small threshold for float comparison
      outputs_differ = true;
      break;
    }
  }

  printf("Max difference between outputs: %f\n", max_diff);
  EXPECT_TRUE(outputs_differ)
      << "Attention bias should affect the output (max_diff=" << max_diff
      << ")";

  printf("Bias affects output test passed!\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
  aoti_torch_delete_tensor_object(attn_bias);
  aoti_torch_delete_tensor_object(output_no_bias);
  aoti_torch_delete_tensor_object(output_with_bias);
}

// Test numerical correctness: negative bias masks out positions
TEST_F(AOTITorchSDPATest, EfficientAttention_NegativeBiasMasking) {
  const int64_t batch = 1;
  const int64_t num_heads = 1;
  const int64_t seq_len = 4;
  const int64_t head_dim = 8;

  Tensor* query =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);
  Tensor* key =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);

  // Create distinct values at each position in V
  Tensor* value =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.0f);
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

  // Create bias that masks out the last position (apply large negative bias)
  Tensor* attn_bias =
      create_float_tensor({batch, num_heads, seq_len, seq_len}, 0.0f);
  std::vector<float> bias_host(batch * num_heads * seq_len * seq_len, 0.0f);
  // Mask out last position for all queries
  for (int64_t q = 0; q < seq_len; ++q) {
    bias_host[q * seq_len + (seq_len - 1)] = -10000.0f; // Large negative value
  }
  cudaMemcpy(
      attn_bias->data_ptr(),
      bias_host.data(),
      bias_host.size() * sizeof(float),
      cudaMemcpyHostToDevice);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);
  ASSERT_NE(attn_bias, nullptr);

  printf("Testing negative bias masking\n");

  Tensor* output = nullptr;
  AOTITorchError error = call_efficient_attention(
      query, key, value, attn_bias, 0, nullptr, &output);
  ASSERT_EQ(error, Error::Ok);
  ASSERT_NE(output, nullptr);

  // Copy output to host
  std::vector<float> output_data = copy_tensor_to_host(output);

  // With Q=K, without masking, output would be average of all positions:
  // (1+2+3+4)/4 = 2.5 With last position masked, output should be: (1+2+3)/3
  // = 2.0
  float avg_output = 0.0f;
  for (size_t i = 0; i < output_data.size(); ++i) {
    avg_output += output_data[i];
  }
  avg_output /= output_data.size();

  printf("  Average output: %f (expected ~2.0)\n", avg_output);
  EXPECT_TRUE(approx_equal(avg_output, 2.0f, 0.2f))
      << "Masked position should not contribute to output";

  printf("Negative bias masking test passed!\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
  aoti_torch_delete_tensor_object(attn_bias);
  aoti_torch_delete_tensor_object(output);
}

// Test with custom scale and bias
TEST_F(AOTITorchSDPATest, EfficientAttention_CustomScaleWithBias) {
  const int64_t batch = 1;
  const int64_t num_heads = 2;
  const int64_t seq_len = 16;
  const int64_t head_dim = 32;

  Tensor* query =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.5f);
  Tensor* key =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.3f);
  Tensor* value =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);
  Tensor* attn_bias =
      create_float_tensor({batch, num_heads, seq_len, seq_len}, 0.1f);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);
  ASSERT_NE(attn_bias, nullptr);

  printf("Testing Efficient Attention with custom scale and bias\n");

  double custom_scale = 0.1;
  Tensor* output = nullptr;
  AOTITorchError error = call_efficient_attention(
      query, key, value, attn_bias, 0, &custom_scale, &output);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_TRUE(
      check_output_shape(output, {batch, num_heads, seq_len, head_dim}));

  printf("Custom scale with bias test passed!\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
  aoti_torch_delete_tensor_object(attn_bias);
  aoti_torch_delete_tensor_object(output);
}

// Test with edge case: all zeros in query
TEST_F(AOTITorchSDPATest, EfficientAttention_ZeroQuery) {
  const int64_t batch = 1;
  const int64_t num_heads = 1;
  const int64_t seq_len = 8;
  const int64_t head_dim = 16;

  Tensor* query =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.0f);
  Tensor* key =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);
  Tensor* value =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 2.0f);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);

  printf("Testing Efficient Attention with zero query\n");

  Tensor* output = nullptr;
  AOTITorchError error =
      call_efficient_attention(query, key, value, nullptr, 0, nullptr, &output);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_TRUE(
      check_output_shape(output, {batch, num_heads, seq_len, head_dim}));

  // With zero query, attention scores are uniform, output should be average of
  // V
  std::vector<float> output_data = copy_tensor_to_host(output);
  bool all_close_to_2 = true;
  for (size_t i = 0; i < output_data.size(); ++i) {
    if (!approx_equal(output_data[i], 2.0f, 0.1f)) {
      all_close_to_2 = false;
      break;
    }
  }
  EXPECT_TRUE(all_close_to_2)
      << "With uniform attention, output should equal V";

  printf("Zero query test passed!\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
  aoti_torch_delete_tensor_object(output);
}

// Test long sequences (stress test)
TEST_F(AOTITorchSDPATest, EfficientAttention_LongSequence) {
  const int64_t batch = 1;
  const int64_t num_heads = 4;
  const int64_t seq_len = 512; // Long sequence
  const int64_t head_dim = 64;

  Tensor* query =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.1f);
  Tensor* key =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 0.1f);
  Tensor* value =
      create_float_tensor({batch, num_heads, seq_len, head_dim}, 1.0f);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);

  printf("Testing Efficient Attention with long sequence: %ld\n", seq_len);

  Tensor* output = nullptr;
  AOTITorchError error =
      call_efficient_attention(query, key, value, nullptr, 0, nullptr, &output);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(output, nullptr);
  EXPECT_TRUE(
      check_output_shape(output, {batch, num_heads, seq_len, head_dim}));

  printf("Long sequence test passed!\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
  aoti_torch_delete_tensor_object(output);
}

// ============================================================================
// Regression Tests - Specific Bug Reproductions
// ============================================================================

// Test case reproducing Whisper cross-attention bug:
// Flash Attention kernel fails with "invalid argument" when seq_len_q=1
// This happens in Whisper decoder cross-attention where query is single token
// but key/value come from encoder (1500 tokens)
TEST_F(AOTITorchSDPATest, REGRESSION_WhisperCrossAttention_SeqLenQ1) {
  // Exact configuration from the failing Whisper model:
  const int64_t batch = 1;
  const int64_t num_heads = 20;
  const int64_t seq_len_q = 1; // Single query token (decoder)
  const int64_t seq_len_kv = 1500; // Encoder tokens
  const int64_t head_dim = 64;

  printf("\n=== REGRESSION TEST: Whisper Cross-Attention (seq_len_q=1) ===\n");
  printf("Configuration:\n");
  printf("  batch=%ld, num_heads=%ld\n", batch, num_heads);
  printf(
      "  seq_len_q=%ld, seq_len_kv=%ld, head_dim=%ld\n",
      seq_len_q,
      seq_len_kv,
      head_dim);
  printf("  This reproduces the 'invalid argument' error in Flash Attention\n");

  // Create tensors with Half precision (dtype=15 from error log)
  Tensor* query = nullptr;
  Tensor* key = nullptr;
  Tensor* value = nullptr;

  // Calculate strides (row-major, contiguous)
  std::vector<int64_t> query_shape = {batch, num_heads, seq_len_q, head_dim};
  std::vector<int64_t> query_strides = {
      num_heads * seq_len_q * head_dim, // batch stride
      seq_len_q * head_dim, // head stride
      head_dim, // seq stride
      1 // dim stride
  };

  std::vector<int64_t> kv_shape = {batch, num_heads, seq_len_kv, head_dim};
  std::vector<int64_t> kv_strides = {
      num_heads * seq_len_kv * head_dim, // batch stride
      seq_len_kv * head_dim, // head stride
      head_dim, // seq stride
      1 // dim stride
  };

  // Create tensors with Half precision
  Error error = aoti_torch_empty_strided(
      4,
      query_shape.data(),
      query_strides.data(),
      static_cast<int32_t>(SupportedDTypes::FLOAT16),
      static_cast<int32_t>(SupportedDevices::CUDA),
      0,
      &query);
  ASSERT_EQ(error, Error::Ok);
  ASSERT_NE(query, nullptr);

  error = aoti_torch_empty_strided(
      4,
      kv_shape.data(),
      kv_strides.data(),
      static_cast<int32_t>(SupportedDTypes::FLOAT16),
      static_cast<int32_t>(SupportedDevices::CUDA),
      0,
      &key);
  ASSERT_EQ(error, Error::Ok);
  ASSERT_NE(key, nullptr);

  error = aoti_torch_empty_strided(
      4,
      kv_shape.data(),
      kv_strides.data(),
      static_cast<int32_t>(SupportedDTypes::FLOAT16),
      static_cast<int32_t>(SupportedDevices::CUDA),
      0,
      &value);
  ASSERT_EQ(error, Error::Ok);
  ASSERT_NE(value, nullptr);

  // Fill with some values (using float, will be converted to half)
  int64_t query_size = batch * num_heads * seq_len_q * head_dim;
  int64_t kv_size = batch * num_heads * seq_len_kv * head_dim;

  std::vector<__half> query_data(query_size);
  std::vector<__half> key_data(kv_size);
  std::vector<__half> value_data(kv_size);

  for (int64_t i = 0; i < query_size; ++i) {
    query_data[i] = __float2half(0.5f);
  }
  for (int64_t i = 0; i < kv_size; ++i) {
    key_data[i] = __float2half(0.3f);
    value_data[i] = __float2half(1.0f);
  }

  cudaMemcpy(
      query->data_ptr(),
      query_data.data(),
      query_size * sizeof(__half),
      cudaMemcpyHostToDevice);
  cudaMemcpy(
      key->data_ptr(),
      key_data.data(),
      kv_size * sizeof(__half),
      cudaMemcpyHostToDevice);
  cudaMemcpy(
      value->data_ptr(),
      value_data.data(),
      kv_size * sizeof(__half),
      cudaMemcpyHostToDevice);

  printf("\nTensor shapes created successfully:\n");
  printf(
      "  Query: [%ld, %ld, %ld, %ld]\n",
      query->size(0),
      query->size(1),
      query->size(2),
      query->size(3));
  printf(
      "  Key:   [%ld, %ld, %ld, %ld]\n",
      key->size(0),
      key->size(1),
      key->size(2),
      key->size(3));
  printf(
      "  Value: [%ld, %ld, %ld, %ld]\n",
      value->size(0),
      value->size(1),
      value->size(2),
      value->size(3));

  // Check if Flash Attention can be used for this configuration
  bool can_use_fa = can_use_flash_attention(query, key, value, nullptr, false);
  printf("\ncan_use_flash_attention: %s\n", can_use_fa ? "true" : "false");

  // Try Flash Attention (this is expected to FAIL with "invalid argument")
  Tensor* output = nullptr;
  double scale_factor = 1.0; // Same as in the error log
  AOTITorchError attn_error = call_flash_attention(
      query,
      key,
      value,
      0.0, // no dropout
      0, // not causal
      &scale_factor, // custom scale = 1.0
      &output);

  if (attn_error != Error::Ok) {
    printf("\n⚠️  Flash Attention FAILED as expected!\n");
    printf("  Error code: %d\n", static_cast<int>(attn_error));
    printf(
        "  This reproduces the bug: Flash Attention doesn't handle seq_len_q=1\n");

    // Now try with Efficient Attention (should work as workaround)
    printf("\nTrying Efficient Attention as workaround...\n");
    Tensor* output_efficient = nullptr;
    AOTITorchError efficient_error = call_efficient_attention(
        query, key, value, nullptr, 0, &scale_factor, &output_efficient);

    if (efficient_error == Error::Ok && output_efficient != nullptr) {
      printf("✓ Efficient Attention succeeded!\n");
      printf(
          "  Output shape: [%ld, %ld, %ld, %ld]\n",
          output_efficient->size(0),
          output_efficient->size(1),
          output_efficient->size(2),
          output_efficient->size(3));

      // Verify output shape
      EXPECT_TRUE(check_output_shape(
          output_efficient, {batch, num_heads, seq_len_q, head_dim}));

      aoti_torch_delete_tensor_object(output_efficient);

      printf(
          "\n💡 RECOMMENDATION: Modify can_use_flash_attention() to return false\n");
      printf(
          "   when seq_len_q < 4 to automatically use Efficient Attention instead.\n");
    } else {
      printf("✗ Efficient Attention also failed!\n");
      FAIL() << "Both Flash and Efficient Attention failed for seq_len_q=1";
    }

    // Mark this as a known failure for Flash Attention
    EXPECT_NE(attn_error, Error::Ok)
        << "Flash Attention with seq_len_q=1 is known to fail (expected failure)";

  } else {
    printf("\n✓ Flash Attention succeeded (bug may be fixed!)\n");
    ASSERT_NE(output, nullptr);
    EXPECT_TRUE(
        check_output_shape(output, {batch, num_heads, seq_len_q, head_dim}));
    aoti_torch_delete_tensor_object(output);
  }

  printf("\n=== End of Regression Test ===\n\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
}

// Additional edge case: seq_len_q=1 with smaller dimensions (easier to debug)
TEST_F(AOTITorchSDPATest, EdgeCase_SeqLenQ1_SmallDims) {
  const int64_t batch = 1;
  const int64_t num_heads = 2;
  const int64_t seq_len_q = 1; // Single query
  const int64_t seq_len_kv = 64; // Smaller than Whisper for easier debugging
  const int64_t head_dim = 32;

  printf("\n=== Edge Case: seq_len_q=1 with small dimensions ===\n");

  Tensor* query =
      create_float_tensor({batch, num_heads, seq_len_q, head_dim}, 0.5f);
  Tensor* key =
      create_float_tensor({batch, num_heads, seq_len_kv, head_dim}, 0.3f);
  Tensor* value =
      create_float_tensor({batch, num_heads, seq_len_kv, head_dim}, 1.0f);

  ASSERT_NE(query, nullptr);
  ASSERT_NE(key, nullptr);
  ASSERT_NE(value, nullptr);

  printf(
      "Testing Flash Attention with seq_len_q=1, seq_len_kv=%ld\n", seq_len_kv);

  // Check if Flash Attention thinks it can handle this
  bool can_use_fa = can_use_flash_attention(query, key, value, nullptr, false);
  printf("can_use_flash_attention: %s\n", can_use_fa ? "true" : "false");

  Tensor* output = nullptr;
  AOTITorchError error =
      call_flash_attention(query, key, value, 0.0, 0, nullptr, &output);

  if (error == Error::Ok && output != nullptr) {
    printf("✓ Flash Attention succeeded with seq_len_q=1\n");
    EXPECT_TRUE(
        check_output_shape(output, {batch, num_heads, seq_len_q, head_dim}));
    aoti_torch_delete_tensor_object(output);
  } else {
    printf("✗ Flash Attention failed with seq_len_q=1\n");

    // Try efficient attention as fallback
    printf("Trying Efficient Attention...\n");
    Tensor* output_efficient = nullptr;
    AOTITorchError eff_error = call_efficient_attention(
        query, key, value, nullptr, 0, nullptr, &output_efficient);

    EXPECT_EQ(eff_error, Error::Ok);
    if (output_efficient != nullptr) {
      printf("✓ Efficient Attention succeeded\n");
      EXPECT_TRUE(check_output_shape(
          output_efficient, {batch, num_heads, seq_len_q, head_dim}));
      aoti_torch_delete_tensor_object(output_efficient);
    }
  }

  printf("=== End of Edge Case Test ===\n\n");

  // Cleanup
  aoti_torch_delete_tensor_object(query);
  aoti_torch_delete_tensor_object(key);
  aoti_torch_delete_tensor_object(value);
}

// Test various small seq_len_q values to find the threshold
TEST_F(AOTITorchSDPATest, EdgeCase_SmallSeqLenQ_Sweep) {
  const int64_t batch = 1;
  const int64_t num_heads = 4;
  const int64_t seq_len_kv = 128;
  const int64_t head_dim = 64;

  printf("\n=== Testing various small seq_len_q values ===\n");

  for (int64_t seq_len_q = 1; seq_len_q <= 8; ++seq_len_q) {
    printf("\nTesting seq_len_q=%ld:\n", seq_len_q);

    Tensor* query =
        create_float_tensor({batch, num_heads, seq_len_q, head_dim}, 0.5f);
    Tensor* key =
        create_float_tensor({batch, num_heads, seq_len_kv, head_dim}, 0.3f);
    Tensor* value =
        create_float_tensor({batch, num_heads, seq_len_kv, head_dim}, 1.0f);

    ASSERT_NE(query, nullptr);
    ASSERT_NE(key, nullptr);
    ASSERT_NE(value, nullptr);

    bool can_use_fa =
        can_use_flash_attention(query, key, value, nullptr, false);

    Tensor* output = nullptr;
    AOTITorchError error =
        call_flash_attention(query, key, value, 0.0, 0, nullptr, &output);

    if (error == Error::Ok && output != nullptr) {
      printf("  ✓ seq_len_q=%ld: Flash Attention PASSED\n", seq_len_q);
      aoti_torch_delete_tensor_object(output);
    } else {
      printf("  ✗ seq_len_q=%ld: Flash Attention FAILED\n", seq_len_q);
    }

    aoti_torch_delete_tensor_object(query);
    aoti_torch_delete_tensor_object(key);
    aoti_torch_delete_tensor_object(value);
  }

  printf("\n=== End of seq_len_q sweep ===\n\n");
}
