/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include <executorch/extension/llm/sampler/argmax.cuh>
#include <executorch/runtime/platform/platform.h>

using namespace executorch::extension::llm::cuda;

// Test fixture for argmax tests
class ArgmaxTest : public ::testing::Test {
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

    // Create CUDA stream
    ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);

    // Allocate output buffers on GPU
    ASSERT_EQ(cudaMalloc(&out_token_gpu_, sizeof(int) * max_rows_), cudaSuccess);
    ASSERT_EQ(
        cudaMalloc(&out_maxlogit_gpu_, sizeof(float) * max_rows_), cudaSuccess);
  }

  void TearDown() override {
    if (out_token_gpu_) {
      cudaFree(out_token_gpu_);
    }
    if (out_maxlogit_gpu_) {
      cudaFree(out_maxlogit_gpu_);
    }
    if (stream_) {
      cudaStreamDestroy(stream_);
    }
  }

  // Helper to create and upload float tensor to GPU
  float* create_float_tensor(const std::vector<float>& data) {
    float* gpu_ptr;
    EXPECT_EQ(cudaMalloc(&gpu_ptr, data.size() * sizeof(float)), cudaSuccess);
    EXPECT_EQ(
        cudaMemcpy(
            gpu_ptr,
            data.data(),
            data.size() * sizeof(float),
            cudaMemcpyHostToDevice),
        cudaSuccess);
    return gpu_ptr;
  }

  // Helper to create and upload half tensor to GPU
  half* create_half_tensor(const std::vector<float>& data) {
    std::vector<half> half_data(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
      half_data[i] = __float2half(data[i]);
    }
    half* gpu_ptr;
    EXPECT_EQ(cudaMalloc(&gpu_ptr, data.size() * sizeof(half)), cudaSuccess);
    EXPECT_EQ(
        cudaMemcpy(
            gpu_ptr,
            half_data.data(),
            data.size() * sizeof(half),
            cudaMemcpyHostToDevice),
        cudaSuccess);
    return gpu_ptr;
  }

  // Helper to create and upload bfloat16 tensor to GPU
  nv_bfloat16* create_bfloat16_tensor(const std::vector<float>& data) {
    std::vector<nv_bfloat16> bf16_data(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
      bf16_data[i] = __float2bfloat16(data[i]);
    }
    nv_bfloat16* gpu_ptr;
    EXPECT_EQ(
        cudaMalloc(&gpu_ptr, data.size() * sizeof(nv_bfloat16)), cudaSuccess);
    EXPECT_EQ(
        cudaMemcpy(
            gpu_ptr,
            bf16_data.data(),
            data.size() * sizeof(nv_bfloat16),
            cudaMemcpyHostToDevice),
        cudaSuccess);
    return gpu_ptr;
  }

  // Helper to get CPU argmax for verification
  int cpu_argmax(const std::vector<float>& data) {
    return static_cast<int>(
        std::max_element(data.begin(), data.end()) - data.begin());
  }

  cudaStream_t stream_ = nullptr;
  int* out_token_gpu_ = nullptr;
  float* out_maxlogit_gpu_ = nullptr;
  static constexpr int max_rows_ = 16;
};

// Test basic argmax with float32
TEST_F(ArgmaxTest, BasicFloat32) {
  std::vector<float> logits = {0.1f, 0.5f, 0.8f, 0.3f, 0.2f};
  int vocab_size = static_cast<int>(logits.size());
  int expected_idx = cpu_argmax(logits);

  float* gpu_logits = create_float_tensor(logits);

  int32_t out_token = argmax_cuda(
      gpu_logits,
      vocab_size,
      ::executorch::aten::ScalarType::Float,
      stream_,
      out_token_gpu_);

  EXPECT_EQ(out_token, expected_idx);

  cudaFree(gpu_logits);
}

// Test argmax with half precision
TEST_F(ArgmaxTest, BasicHalf) {
  std::vector<float> logits = {0.1f, 0.2f, 0.9f, 0.4f, 0.5f, 0.6f};
  int vocab_size = static_cast<int>(logits.size());
  int expected_idx = cpu_argmax(logits);

  half* gpu_logits = create_half_tensor(logits);

  int32_t out_token = argmax_cuda(
      gpu_logits,
      vocab_size,
      ::executorch::aten::ScalarType::Half,
      stream_,
      out_token_gpu_);

  EXPECT_EQ(out_token, expected_idx);

  cudaFree(gpu_logits);
}

// Test argmax with bfloat16
TEST_F(ArgmaxTest, BasicBFloat16) {
  std::vector<float> logits = {-1.0f, 2.5f, 1.0f, 0.5f};
  int vocab_size = static_cast<int>(logits.size());
  int expected_idx = cpu_argmax(logits);

  nv_bfloat16* gpu_logits = create_bfloat16_tensor(logits);

  int32_t out_token = argmax_cuda(
      gpu_logits,
      vocab_size,
      ::executorch::aten::ScalarType::BFloat16,
      stream_,
      out_token_gpu_);

  EXPECT_EQ(out_token, expected_idx);

  cudaFree(gpu_logits);
}

// Test with large vocabulary (typical for LLMs)
TEST_F(ArgmaxTest, LargeVocab) {
  int vocab_size = 32000; // Typical LLM vocab size
  std::vector<float> logits(vocab_size);

  // Fill with random values
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
  for (int i = 0; i < vocab_size; ++i) {
    logits[i] = dist(gen);
  }

  // Set a known maximum
  int expected_idx = 12345;
  logits[expected_idx] = 100.0f;

  float* gpu_logits = create_float_tensor(logits);

  int32_t out_token = argmax_cuda(
      gpu_logits,
      vocab_size,
      ::executorch::aten::ScalarType::Float,
      stream_,
      out_token_gpu_);

  EXPECT_EQ(out_token, expected_idx);

  cudaFree(gpu_logits);
}

// Test multiple rows (batch)
TEST_F(ArgmaxTest, MultipleRows) {
  int rows = 4;
  int vocab_size = 10;
  std::vector<float> logits = {
      // Row 0: max at index 2
      0.1f, 0.2f, 0.9f, 0.3f, 0.4f, 0.5f, 0.1f, 0.2f, 0.3f, 0.4f,
      // Row 1: max at index 5
      0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.8f, 0.1f, 0.2f, 0.3f, 0.4f,
      // Row 2: max at index 9
      0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.1f, 0.2f, 0.3f, 0.4f, 0.95f,
      // Row 3: max at index 0
      0.99f, 0.2f, 0.3f, 0.4f, 0.5f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
  };

  std::vector<int> expected_indices = {2, 5, 9, 0};

  float* gpu_logits = create_float_tensor(logits);

  launch_argmax_vocab_rows(
      gpu_logits,
      ::executorch::aten::ScalarType::Float,
      rows,
      vocab_size,
      out_token_gpu_,
      out_maxlogit_gpu_,
      stream_);

  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  std::vector<int> out_tokens(rows);
  ASSERT_EQ(
      cudaMemcpy(
          out_tokens.data(),
          out_token_gpu_,
          rows * sizeof(int),
          cudaMemcpyDeviceToHost),
      cudaSuccess);

  for (int i = 0; i < rows; ++i) {
    EXPECT_EQ(out_tokens[i], expected_indices[i]) << "Row " << i << " failed";
  }

  cudaFree(gpu_logits);
}

// Test tie-breaking (smaller index wins)
TEST_F(ArgmaxTest, TieBreaking) {
  std::vector<float> logits = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
  int vocab_size = static_cast<int>(logits.size());

  float* gpu_logits = create_float_tensor(logits);

  int32_t out_token = argmax_cuda(
      gpu_logits,
      vocab_size,
      ::executorch::aten::ScalarType::Float,
      stream_,
      out_token_gpu_);

  // Smallest index should win on tie
  EXPECT_EQ(out_token, 0);

  cudaFree(gpu_logits);
}

// Test with negative values
TEST_F(ArgmaxTest, NegativeValues) {
  std::vector<float> logits = {-5.0f, -2.0f, -1.0f, -3.0f, -4.0f};
  int vocab_size = static_cast<int>(logits.size());
  int expected_idx = cpu_argmax(logits); // Should be index 2 (-1.0f)

  float* gpu_logits = create_float_tensor(logits);

  int32_t out_token = argmax_cuda(
      gpu_logits,
      vocab_size,
      ::executorch::aten::ScalarType::Float,
      stream_,
      out_token_gpu_);

  EXPECT_EQ(out_token, expected_idx);

  cudaFree(gpu_logits);
}

// Test max at first position
TEST_F(ArgmaxTest, MaxAtFirst) {
  std::vector<float> logits = {10.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  int vocab_size = static_cast<int>(logits.size());

  float* gpu_logits = create_float_tensor(logits);

  int32_t out_token = argmax_cuda(
      gpu_logits,
      vocab_size,
      ::executorch::aten::ScalarType::Float,
      stream_,
      out_token_gpu_);

  EXPECT_EQ(out_token, 0);

  cudaFree(gpu_logits);
}

// Test max at last position
TEST_F(ArgmaxTest, MaxAtLast) {
  std::vector<float> logits = {1.0f, 2.0f, 3.0f, 4.0f, 10.0f};
  int vocab_size = static_cast<int>(logits.size());

  float* gpu_logits = create_float_tensor(logits);

  int32_t out_token = argmax_cuda(
      gpu_logits,
      vocab_size,
      ::executorch::aten::ScalarType::Float,
      stream_,
      out_token_gpu_);

  EXPECT_EQ(out_token, 4);

  cudaFree(gpu_logits);
}

// Test with different thread counts
