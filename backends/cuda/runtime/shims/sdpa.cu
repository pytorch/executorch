/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

#include <executorch/backends/aoti/utils.h>
#include <executorch/backends/cuda/runtime/shims/sdpa.h>
#include <executorch/backends/cuda/runtime/utils.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/backends/cuda/runtime/shims/sdpa.cuh>

namespace executorch::backends::cuda {

using executorch::backends::aoti::AOTITorchError;
using executorch::backends::aoti::Tensor;
using executorch::runtime::Error;

// ============================================================================
// CUDA Kernels for Softmax and Masking
// ============================================================================

// Helper function for max with different types
__device__ __forceinline__ float device_max(float a, float b) {
  return fmaxf(a, b);
}

__device__ __forceinline__ __half device_max(__half a, __half b) {
  return __hgt(a, b) ? a : b;
}

__device__ __forceinline__ __nv_bfloat16
device_max(__nv_bfloat16 a, __nv_bfloat16 b) {
#if __CUDA_ARCH__ >= 800
  return __hgt(a, b) ? a : b;
#else
  return __float2bfloat16(fmaxf(__bfloat162float(a), __bfloat162float(b)));
#endif
}

/**
 * Softmax kernel with optional causal masking and attention bias
 *
 * Computes softmax along the last dimension (seq_len_k) of a 4D tensor.
 * Supports:
 * - Causal masking where positions j > i are masked out
 * - Explicit attention bias (additive mask, 0.0 = allow, -inf = mask)
 *
 * Input: [batch, num_heads, seq_len_q, seq_len_k]
 * Output: [batch, num_heads, seq_len_q, seq_len_k]
 * Bias (optional): [batch, num_heads, seq_len_q, seq_len_k] or broadcastable
 *
 * Each thread processes one row (seq_len_q position).
 *
 * Note: Supports in-place operation (input == output).
 */
template <typename scalar_t>
__global__ void softmax_with_mask_kernel(
    const scalar_t* input,
    scalar_t* output,
    const scalar_t* attn_bias, // Optional attention bias (can be nullptr)
    const int64_t batch,
    const int64_t num_heads,
    const int64_t seq_len_q,
    const int64_t seq_len_k,
    const bool is_causal,
    const float scale) {
  // Each block processes one row of the attention matrix
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total_rows = batch * num_heads * seq_len_q;

  if (idx >= total_rows) {
    return;
  }

  // Decode position - we only need i for causal masking
  const int64_t i = idx % seq_len_q;

  // Pointer to the start of this row
  const int64_t row_offset = idx * seq_len_k;
  const scalar_t* input_row = input + row_offset;
  scalar_t* output_row = output + row_offset;

  // Attention bias for this row (if provided)
  const scalar_t* bias_row = attn_bias ? (attn_bias + row_offset) : nullptr;

  // Find max for numerical stability (two-pass algorithm)
  float max_val = -FLT_MAX;
  for (int64_t j = 0; j < seq_len_k; ++j) {
    // Apply scaling and bias
    float val = static_cast<float>(input_row[j]) * scale;
    if (bias_row) {
      float bias = static_cast<float>(bias_row[j]);
      val += bias; // Additive bias (0.0 = allow, -inf = mask)
    }

    // Apply causal mask if needed
    if (!is_causal || j <= i) {
      max_val = fmaxf(max_val, val);
    }
  }

  // Compute exp and sum
  float sum_exp = 0.0f;
  for (int64_t j = 0; j < seq_len_k; ++j) {
    float val = static_cast<float>(input_row[j]) * scale;
    if (bias_row) {
      float bias = static_cast<float>(bias_row[j]);
      val += bias;
    }

    float exp_val;
    if (!is_causal || j <= i) {
      exp_val = expf(val - max_val);
    } else {
      exp_val = 0.0f;
    }
    output_row[j] = static_cast<scalar_t>(exp_val);
    sum_exp += exp_val;
  }

  // Normalize
  const float inv_sum = 1.0f / (sum_exp + 1e-12f); // Add epsilon for stability
  for (int64_t j = 0; j < seq_len_k; ++j) {
    output_row[j] =
        static_cast<scalar_t>(static_cast<float>(output_row[j]) * inv_sum);
  }
}

/**
 * Scale kernel - multiply all elements by a scalar
 */
template <typename scalar_t>
__global__ void scale_kernel(
    scalar_t* __restrict__ data,
    const int64_t size,
    const float scale) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] = static_cast<scalar_t>(static_cast<float>(data[idx]) * scale);
  }
}

// ============================================================================
// cuBLAS Helper Functions
// ============================================================================

/**
 * Get or create a cuBLAS handle for the current stream
 *
 * Note: In production, this should use a handle pool or be managed
 * by the backend infrastructure. This is a simplified version.
 */
cublasHandle_t get_cublas_handle(cudaStream_t stream) {
  static cublasHandle_t handle = nullptr;
  static bool init_attempted = false;
  static bool init_success = false;

  if (!init_attempted) {
    init_attempted = true;
    cublasStatus_t status = cublasCreate_v2(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      ET_LOG(
          Error,
          "Failed to create cuBLAS handle: %d",
          static_cast<int>(status));
      handle = nullptr;
      init_success = false;
    } else {
      ET_LOG(Info, "cuBLAS handle created successfully");
      init_success = true;
    }
  }

  if (!init_success || handle == nullptr) {
    return nullptr;
  }

  cublasStatus_t status = cublasSetStream_v2(handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    ET_LOG(Error, "Failed to set cuBLAS stream: %d", static_cast<int>(status));
    return nullptr;
  }

  return handle;
}

/**
 * Batched matrix multiplication wrapper for cuBLAS
 *
 * Computes: C = alpha * op(A) @ op(B) + beta * C
 * for a batch of matrices
 */
template <typename scalar_t>
cublasStatus_t batched_gemm(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const scalar_t* alpha,
    const scalar_t* A,
    int lda,
    int64_t strideA,
    const scalar_t* B,
    int ldb,
    int64_t strideB,
    const scalar_t* beta,
    scalar_t* C,
    int ldc,
    int64_t strideC,
    int batchCount);

// Specializations for different types
template <>
cublasStatus_t batched_gemm<float>(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float* alpha,
    const float* A,
    int lda,
    int64_t strideA,
    const float* B,
    int ldb,
    int64_t strideB,
    const float* beta,
    float* C,
    int ldc,
    int64_t strideC,
    int batchCount) {
  return cublasSgemmStridedBatched(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      A,
      lda,
      strideA,
      B,
      ldb,
      strideB,
      beta,
      C,
      ldc,
      strideC,
      batchCount);
}

template <>
cublasStatus_t batched_gemm<__half>(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const __half* alpha,
    const __half* A,
    int lda,
    int64_t strideA,
    const __half* B,
    int ldb,
    int64_t strideB,
    const __half* beta,
    __half* C,
    int ldc,
    int64_t strideC,
    int batchCount) {
  return cublasHgemmStridedBatched(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      A,
      lda,
      strideA,
      B,
      ldb,
      strideB,
      beta,
      C,
      ldc,
      strideC,
      batchCount);
}

// Note: BFloat16 uses compute type float internally
template <>
cublasStatus_t batched_gemm<__nv_bfloat16>(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const __nv_bfloat16* alpha,
    const __nv_bfloat16* A,
    int lda,
    int64_t strideA,
    const __nv_bfloat16* B,
    int ldb,
    int64_t strideB,
    const __nv_bfloat16* beta,
    __nv_bfloat16* C,
    int ldc,
    int64_t strideC,
    int batchCount) {
// cuBLAS BFloat16 GEMM - introduced in CUDA 11+
#if CUDA_VERSION >= 11000
  // For BFloat16, we need to use cublasGemmStridedBatchedEx
  // with compute type CUBLAS_COMPUTE_32F
  float alpha_f = 1.0f;
  float beta_f = 0.0f;

  return cublasGemmStridedBatchedEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      &alpha_f,
      A,
      CUDA_R_16BF,
      lda,
      strideA,
      B,
      CUDA_R_16BF,
      ldb,
      strideB,
      &beta_f,
      C,
      CUDA_R_16BF,
      ldc,
      strideC,
      batchCount,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT);
#else
  ET_LOG(Error, "BFloat16 GEMM requires CUDA 11.0 or later");
  return CUBLAS_STATUS_NOT_SUPPORTED;
#endif
}

// ============================================================================
// Flash Attention Implementation
// ============================================================================

/**
 * Flash Attention kernel - memory-efficient attention computation
 *
 * This kernel implements the Flash Attention algorithm which computes
 * attention in blocks to reduce memory usage from O(N^2) to O(N).
 */
template <typename scalar_t, int BLOCK_SIZE = 64>
__global__ void flash_attention_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ output,
    const int64_t seq_len_q,
    const int64_t seq_len_k,
    const int64_t head_dim,
    const int64_t head_dim_v,
    const float scale,
    const bool is_causal) {
  const int64_t batch_head_idx = blockIdx.z;
  const int64_t q_block_idx = blockIdx.y;
  const int64_t q_start = q_block_idx * BLOCK_SIZE;
  const int64_t q_end = min(q_start + BLOCK_SIZE, seq_len_q);
  const int64_t q_block_size = q_end - q_start;
  const int tid = threadIdx.x;

  extern __shared__ char shared_mem[];
  scalar_t* shared_q = reinterpret_cast<scalar_t*>(shared_mem);
  scalar_t* shared_k = shared_q + BLOCK_SIZE * head_dim;

  const scalar_t* q_base = query + batch_head_idx * seq_len_q * head_dim;
  const scalar_t* k_base = key + batch_head_idx * seq_len_k * head_dim;
  const scalar_t* v_base = value + batch_head_idx * seq_len_k * head_dim_v;
  scalar_t* out_base = output + batch_head_idx * seq_len_q * head_dim_v;

  for (int i = tid; i < q_block_size * head_dim; i += blockDim.x) {
    int q_idx = i / head_dim;
    int d_idx = i % head_dim;
    if (q_start + q_idx < seq_len_q) {
      shared_q[i] = q_base[(q_start + q_idx) * head_dim + d_idx];
    }
  }
  __syncthreads();

  for (int64_t q_local = 0; q_local < q_block_size; ++q_local) {
    if (tid != 0)
      continue;

    const int64_t q_idx = q_start + q_local;
    float max_score = -FLT_MAX;
    float sum_exp = 0.0f;
    float output_acc[64];
    for (int d = 0; d < head_dim_v; ++d) {
      output_acc[d] = 0.0f;
    }

    const int64_t k_blocks = (seq_len_k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int64_t k_block_idx = 0; k_block_idx < k_blocks; ++k_block_idx) {
      const int64_t k_start = k_block_idx * BLOCK_SIZE;
      const int64_t k_end = min(k_start + BLOCK_SIZE, seq_len_k);
      const int64_t k_block_size = k_end - k_start;

      float block_scores[64];
      float block_max = -FLT_MAX;

      for (int64_t k_local = 0; k_local < k_block_size; ++k_local) {
        const int64_t k_idx = k_start + k_local;
        if (is_causal && k_idx > q_idx) {
          block_scores[k_local] = -FLT_MAX;
          continue;
        }

        float score = 0.0f;
        for (int64_t d = 0; d < head_dim; ++d) {
          float q_val = static_cast<float>(shared_q[q_local * head_dim + d]);
          float k_val = static_cast<float>(k_base[k_idx * head_dim + d]);
          score += q_val * k_val;
        }
        score *= scale;
        block_scores[k_local] = score;
        block_max = fmaxf(block_max, score);
      }

      float new_max = fmaxf(max_score, block_max);
      float exp_correction = expf(max_score - new_max);

      for (int d = 0; d < head_dim_v; ++d) {
        output_acc[d] *= exp_correction;
      }
      sum_exp *= exp_correction;

      for (int64_t k_local = 0; k_local < k_block_size; ++k_local) {
        const int64_t k_idx = k_start + k_local;
        if (is_causal && k_idx > q_idx)
          continue;

        float exp_score = expf(block_scores[k_local] - new_max);
        sum_exp += exp_score;

        for (int64_t d = 0; d < head_dim_v; ++d) {
          float v_val = static_cast<float>(v_base[k_idx * head_dim_v + d]);
          output_acc[d] += exp_score * v_val;
        }
      }
      max_score = new_max;
    }

    float inv_sum = 1.0f / sum_exp;
    for (int64_t d = 0; d < head_dim_v; ++d) {
      out_base[q_idx * head_dim_v + d] =
          static_cast<scalar_t>(output_acc[d] * inv_sum);
    }
  }
}

/**
 * Flash Attention implementation dispatcher
 */
template <typename scalar_t>
Tensor* sdpa_flash_attention_impl(
    const Tensor* query,
    const Tensor* key,
    const Tensor* value,
    const Tensor* attn_mask,
    bool is_causal,
    float scale_factor,
    cudaStream_t stream) {
  const int64_t batch = query->size(0);
  const int64_t num_heads = query->size(1);
  const int64_t seq_len_q = query->size(2);
  const int64_t head_dim = query->size(3);
  const int64_t seq_len_k = key->size(2);
  const int64_t head_dim_v = value->size(3);

  Tensor* output = nullptr;
  std::array<int64_t, 4> output_shape = {
      batch, num_heads, seq_len_q, head_dim_v};
  std::array<int64_t, 4> output_stride = {
      num_heads * seq_len_q * head_dim_v,
      seq_len_q * head_dim_v,
      head_dim_v,
      1};

  auto dtype_int = static_cast<int32_t>(query->dtype());
  aoti_torch_empty_strided(
      4,
      output_shape.data(),
      output_stride.data(),
      dtype_int,
      static_cast<int32_t>(SupportedDevices::CUDA),
      0,
      &output);

  if (output == nullptr) {
    ET_LOG(Error, "sdpa_flash_attention: Failed to allocate output tensor");
    return nullptr;
  }

  constexpr int BLOCK_SIZE = 64;
  const int64_t q_blocks = (seq_len_q + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int64_t batch_head_count = batch * num_heads;
  const size_t shared_mem_size = BLOCK_SIZE * head_dim * sizeof(scalar_t) * 2 +
      BLOCK_SIZE * BLOCK_SIZE * sizeof(float);

  dim3 grid(1, q_blocks, batch_head_count);
  dim3 block(256);

  const scalar_t* q_ptr = reinterpret_cast<const scalar_t*>(query->data_ptr());
  const scalar_t* k_ptr = reinterpret_cast<const scalar_t*>(key->data_ptr());
  const scalar_t* v_ptr = reinterpret_cast<const scalar_t*>(value->data_ptr());
  scalar_t* out_ptr = reinterpret_cast<scalar_t*>(output->data_ptr());

  flash_attention_kernel<scalar_t, BLOCK_SIZE>
      <<<grid, block, shared_mem_size, stream>>>(
          q_ptr,
          k_ptr,
          v_ptr,
          out_ptr,
          seq_len_q,
          seq_len_k,
          head_dim,
          head_dim_v,
          scale_factor,
          is_causal);

  cudaError_t cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess) {
    ET_LOG(
        Error,
        "sdpa_flash_attention: Kernel launch failed: %s",
        cudaGetErrorString(cuda_err));
    aoti_torch_delete_tensor_object(output);
    return nullptr;
  }
  return output;
}

/**
 * Flash Attention entry point with dtype dispatch
 */
Tensor* sdpa_flash_attention(
    const Tensor* query,
    const Tensor* key,
    const Tensor* value,
    const Tensor* attn_mask,
    bool is_causal,
    double scale_factor,
    cudaStream_t stream) {
  auto dtype = query->dtype();
  if (dtype == executorch::aten::ScalarType::Float) {
    return sdpa_flash_attention_impl<float>(
        query,
        key,
        value,
        attn_mask,
        is_causal,
        static_cast<float>(scale_factor),
        stream);
  } else if (dtype == executorch::aten::ScalarType::Half) {
    return sdpa_flash_attention_impl<__half>(
        query,
        key,
        value,
        attn_mask,
        is_causal,
        static_cast<float>(scale_factor),
        stream);
  } else if (dtype == executorch::aten::ScalarType::BFloat16) {
    return sdpa_flash_attention_impl<__nv_bfloat16>(
        query,
        key,
        value,
        attn_mask,
        is_causal,
        static_cast<float>(scale_factor),
        stream);
  } else {
    ET_LOG(Error, "sdpa_flash_attention: Unsupported dtype");
    return nullptr;
  }
}

// ============================================================================
// Memory-Efficient Attention Implementation (with attn_bias support)
// ============================================================================

/**
 * Memory-Efficient Attention kernel with attention bias support
 *
 * This kernel computes scaled dot-product attention with full support for
 * attention bias (additive mask). It uses online softmax for numerical
 * stability.
 *
 * Each thread processes one query position independently.
 *
 * Input shapes:
 * - query: [batch, num_heads, seq_len_q, head_dim]
 * - key: [batch, num_heads, seq_len_k, head_dim]
 * - value: [batch, num_heads, seq_len_k, head_dim_v]
 * - attn_bias: [batch, num_heads, seq_len_q, seq_len_k] (optional)
 * - output: [batch, num_heads, seq_len_q, head_dim_v]
 */
template <typename scalar_t, int MAX_HEAD_DIM_V = 128>
__global__ void efficient_attention_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ attn_bias,
    scalar_t* __restrict__ output,
    const int64_t num_heads,
    const int64_t seq_len_q,
    const int64_t seq_len_k,
    const int64_t head_dim,
    const int64_t head_dim_v,
    const float scale,
    const bool is_causal,
    // Query strides [batch, head, seq, dim]
    const int64_t q_stride_batch,
    const int64_t q_stride_head,
    const int64_t q_stride_seq,
    const int64_t q_stride_dim,
    // Key strides [batch, head, seq, dim]
    const int64_t k_stride_batch,
    const int64_t k_stride_head,
    const int64_t k_stride_seq,
    const int64_t k_stride_dim,
    // Value strides [batch, head, seq, dim]
    const int64_t v_stride_batch,
    const int64_t v_stride_head,
    const int64_t v_stride_seq,
    const int64_t v_stride_dim,
    // Output strides [batch, head, seq, dim] - always contiguous
    const int64_t o_stride_batch,
    const int64_t o_stride_head,
    const int64_t o_stride_seq,
    const int64_t o_stride_dim,
    // Bias strides [batch, head, seq_q, seq_k]
    const int64_t bias_stride_batch,
    const int64_t bias_stride_head,
    const int64_t bias_stride_q,
    const int64_t bias_stride_k) {
  const int64_t batch_head_idx = blockIdx.x;
  const int64_t q_idx = blockIdx.y * blockDim.x + threadIdx.x;

  if (q_idx >= seq_len_q) {
    return;
  }

  // Decompose batch_head_idx into batch and head indices
  const int64_t batch_idx = batch_head_idx / num_heads;
  const int64_t head_idx = batch_head_idx % num_heads;

  // Compute base pointers using proper 4D strides
  const scalar_t* q_base =
      query + batch_idx * q_stride_batch + head_idx * q_stride_head;
  const scalar_t* k_base =
      key + batch_idx * k_stride_batch + head_idx * k_stride_head;
  const scalar_t* v_base =
      value + batch_idx * v_stride_batch + head_idx * v_stride_head;
  scalar_t* out_base =
      output + batch_idx * o_stride_batch + head_idx * o_stride_head;

  // Compute bias base pointer using proper 4D indexing with broadcasting
  // support
  const scalar_t* bias_base = nullptr;
  if (attn_bias != nullptr) {
    // Only add stride contribution if the dimension size > 1 (not broadcasting)
    int64_t bias_offset = 0;

    // Note: bias_stride will be 0 for broadcasting dimensions (size=1)
    // This is correct - we want all positions to point to the same element
    bias_offset += batch_idx * bias_stride_batch;
    bias_offset += head_idx * bias_stride_head;
    bias_offset += q_idx * bias_stride_q;

    bias_base = attn_bias + bias_offset;
  }

  float output_acc[MAX_HEAD_DIM_V];
  for (int64_t d = 0; d < head_dim_v && d < MAX_HEAD_DIM_V; ++d) {
    output_acc[d] = 0.0f;
  }

  float max_score = -FLT_MAX;
  float sum_exp = 0.0f;

  for (int64_t k_idx = 0; k_idx < seq_len_k; ++k_idx) {
    if (is_causal && k_idx > q_idx) {
      continue;
    }

    float score = 0.0f;
    for (int64_t d = 0; d < head_dim; ++d) {
      float q_val =
          static_cast<float>(q_base[q_idx * q_stride_seq + d * q_stride_dim]);
      float k_val =
          static_cast<float>(k_base[k_idx * k_stride_seq + d * k_stride_dim]);
      score += q_val * k_val;
    }
    score *= scale;

    // Add bias if provided
    // Note: bias_stride_k should be 1, and we're indexing along the last
    // dimension
    if (bias_base != nullptr) {
      float bias_val = static_cast<float>(bias_base[k_idx * bias_stride_k]);
      score += bias_val;
    }

    float new_max = fmaxf(max_score, score);
    float exp_correction = expf(max_score - new_max);

    for (int64_t d = 0; d < head_dim_v && d < MAX_HEAD_DIM_V; ++d) {
      output_acc[d] *= exp_correction;
    }
    sum_exp = sum_exp * exp_correction + expf(score - new_max);

    float exp_score = expf(score - new_max);
    for (int64_t d = 0; d < head_dim_v && d < MAX_HEAD_DIM_V; ++d) {
      float v_val =
          static_cast<float>(v_base[k_idx * v_stride_seq + d * v_stride_dim]);
      output_acc[d] += exp_score * v_val;
    }

    max_score = new_max;
  }

  float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
  for (int64_t d = 0; d < head_dim_v && d < MAX_HEAD_DIM_V; ++d) {
    out_base[q_idx * o_stride_seq + d * o_stride_dim] =
        static_cast<scalar_t>(output_acc[d] * inv_sum);
  }
}

/**
 * Memory-Efficient Attention implementation dispatcher
 */
template <typename scalar_t>
Tensor* sdpa_efficient_attention_impl(
    const Tensor* query,
    const Tensor* key,
    const Tensor* value,
    const Tensor* attn_bias,
    bool is_causal,
    float scale_factor,
    cudaStream_t stream) {
  const int64_t batch = query->size(0);
  const int64_t num_heads = query->size(1);
  const int64_t seq_len_q = query->size(2);
  const int64_t head_dim = query->size(3);
  const int64_t seq_len_k = key->size(2);
  const int64_t head_dim_v = value->size(3);

  Tensor* output = nullptr;
  std::array<int64_t, 4> output_shape = {
      batch, num_heads, seq_len_q, head_dim_v};
  std::array<int64_t, 4> output_stride = {
      num_heads * seq_len_q * head_dim_v,
      seq_len_q * head_dim_v,
      head_dim_v,
      1};

  auto dtype_int = static_cast<int32_t>(query->dtype());
  aoti_torch_empty_strided(
      4,
      output_shape.data(),
      output_stride.data(),
      dtype_int,
      static_cast<int32_t>(SupportedDevices::CUDA),
      0,
      &output);

  if (output == nullptr) {
    ET_LOG(Error, "sdpa_efficient_attention: Failed to allocate output tensor");
    return nullptr;
  }

  const int64_t batch_head_count = batch * num_heads;
  const int threads_per_block = 128;
  const int64_t q_blocks =
      (seq_len_q + threads_per_block - 1) / threads_per_block;

  dim3 grid(batch_head_count, q_blocks);
  dim3 block(threads_per_block);

  const scalar_t* q_ptr = reinterpret_cast<const scalar_t*>(query->data_ptr());
  const scalar_t* k_ptr = reinterpret_cast<const scalar_t*>(key->data_ptr());
  const scalar_t* v_ptr = reinterpret_cast<const scalar_t*>(value->data_ptr());
  scalar_t* out_ptr = reinterpret_cast<scalar_t*>(output->data_ptr());

  const scalar_t* bias_ptr = nullptr;
  int64_t bias_stride_batch = 0;
  int64_t bias_stride_head = 0;
  int64_t bias_stride_q = 0;
  int64_t bias_stride_k = 1;

  if (attn_bias != nullptr) {
    bias_ptr = reinterpret_cast<const scalar_t*>(attn_bias->data_ptr());

    int64_t bias_dim = attn_bias->dim();
    printf("  attn_bias: ptr=%p, dim=%ld\n", (void*)bias_ptr, bias_dim);
    fflush(stdout);

    if (bias_dim == 4) {
      // Handle attention bias with shape [batch, num_heads, seq_len_q,
      // seq_len_k] or broadcastable variants
      auto bias_strides = attn_bias->strides();
      auto bias_sizes = attn_bias->sizes();

      // Extract sizes safely
      int64_t bias_size_0 = bias_sizes[0];
      int64_t bias_size_1 = bias_sizes[1];
      int64_t bias_size_2 = bias_sizes[2];
      int64_t bias_size_3 = bias_sizes[3];

      // Extract strides safely
      int64_t bias_stride_0 = bias_strides[0];
      int64_t bias_stride_1 = bias_strides[1];
      int64_t bias_stride_2 = bias_strides[2];
      int64_t bias_stride_3 = bias_strides[3];

      printf(
          "  bias dim=4, sizes=[%ld, %ld, %ld, %ld], strides=[%ld, %ld, %ld, %ld]\n",
          bias_size_0,
          bias_size_1,
          bias_size_2,
          bias_size_3,
          bias_stride_0,
          bias_stride_1,
          bias_stride_2,
          bias_stride_3);
      fflush(stdout);

      bias_stride_batch = bias_stride_0;
      bias_stride_head = bias_stride_1;
      bias_stride_q = bias_stride_2;
      bias_stride_k = bias_stride_3;
    } else {
      printf(
          "  WARNING: attn_bias has unexpected dim=%ld (expected 4) tried to print its top 4 size and stride to see the value\n",
          bias_dim);
      fflush(stdout);

      bias_dim = 4;

      // Try to handle 1D or other dimensions
      auto bias_strides = attn_bias->strides();
      auto bias_sizes = attn_bias->sizes();
      printf("  bias_sizes: ");
      for (int64_t i = 0; i < bias_dim; ++i) {
        printf("%d ", bias_sizes[i]);
      }
      printf("\n  bias_strides: ");
      for (int64_t i = 0; i < bias_dim; ++i) {
        printf("%d ", bias_strides[i]);
      }
      printf("\n");
      fflush(stdout);
      exit(1);
    }
  }

  // Debug: Print query tensor info
  auto query_strides = query->strides();
  auto key_strides = key->strides();
  auto value_strides = value->strides();

  printf("Launching efficient_attention_kernel:\n");
  printf(
      "  batch=%ld, num_heads=%ld, seq_len_q=%ld, seq_len_k=%ld, head_dim=%ld, head_dim_v=%ld\n",
      batch,
      num_heads,
      seq_len_q,
      seq_len_k,
      head_dim,
      head_dim_v);
  // printf(
  //     "  query_strides=[%ld, %ld, %ld, %ld]\n",
  //     query_strides[0],
  //     query_strides[1],
  //     query_strides[2],
  //     query_strides[3]);
  // printf(
  //     "  key_strides=[%ld, %ld, %ld, %ld]\n",
  //     key_strides[0],
  //     key_strides[1],
  //     key_strides[2],
  //     key_strides[3]);
  // printf(
  //     "  value_strides=[%ld, %ld, %ld, %ld]\n",
  //     value_strides[0],
  //     value_strides[1],
  // Debug: Print query tensor info
  // auto query_strides = query->strides();
  // auto key_strides = key->strides();
  // auto value_strides = value->strides();

  printf("\n=== Efficient Attention Kernel Launch Details ===\n");
  printf("Tensor Dimensions:\n");
  printf(
      "  batch=%ld, num_heads=%ld, seq_len_q=%ld, seq_len_k=%ld\n",
      batch,
      num_heads,
      seq_len_q,
      seq_len_k);
  printf("  head_dim=%ld, head_dim_v=%ld\n", head_dim, head_dim_v);

  printf(
      "\nQuery Tensor (shape: [%ld, %ld, %ld, %ld]):\n",
      batch,
      num_heads,
      seq_len_q,
      head_dim);
  printf(
      "  strides=[%d, %d, %d, %d]\n",
      query_strides[0],
      query_strides[1],
      query_strides[2],
      query_strides[3]);

  printf(
      "\nKey Tensor (shape: [%ld, %ld, %ld, %ld]):\n",
      batch,
      num_heads,
      seq_len_k,
      head_dim);
  printf(
      "  strides=[%d, %d, %d, %d]\n",
      key_strides[0],
      key_strides[1],
      key_strides[2],
      key_strides[3]);

  printf(
      "\nValue Tensor (shape: [%ld, %ld, %ld, %ld]):\n",
      batch,
      num_heads,
      seq_len_k,
      head_dim_v);
  printf(
      "  strides=[%d, %d, %d, %d]\n",
      value_strides[0],
      value_strides[1],
      value_strides[2],
      value_strides[3]);

  if (attn_bias != nullptr) {
    auto bias_sizes = attn_bias->sizes();
    printf("\nAttention Bias Tensor (shape: [");
    for (int64_t i = 0; i < attn_bias->dim(); ++i) {
      printf("%d", bias_sizes[i]);
      if (i < attn_bias->dim() - 1)
        printf(", ");
    }
    printf("]):\n");
    printf("  bias_ptr=%p\n", (void*)bias_ptr);
    printf(
        "  strides=[batch:%ld, head:%ld, q:%ld, k:%ld]\n",
        bias_stride_batch,
        bias_stride_head,
        bias_stride_q,
        bias_stride_k);
  } else {
    printf("\nAttention Bias: None (nullptr)\n");
  }

  printf("\nKernel Configuration:\n");
  printf("  scale_factor=%.6f\n", scale_factor);
  printf("  is_causal=%d\n", is_causal);
  printf(
      "  grid=(%ld, %ld, 1) [batch_head_count=%ld, q_blocks=%ld]\n",
      batch_head_count,
      q_blocks,
      batch_head_count,
      q_blocks);
  printf(
      "  block=(%d, 1, 1) [threads_per_block=%d]\n",
      threads_per_block,
      threads_per_block);

  // Verify that Q/K/V are contiguous in the batch_head*seq*dim layout
  bool q_is_contiguous =
      (query_strides[0] == num_heads * seq_len_q * head_dim) &&
      (query_strides[1] == seq_len_q * head_dim) &&
      (query_strides[2] == head_dim) && (query_strides[3] == 1);
  bool k_is_contiguous = (key_strides[0] == num_heads * seq_len_k * head_dim) &&
      (key_strides[1] == seq_len_k * head_dim) &&
      (key_strides[2] == head_dim) && (key_strides[3] == 1);
  bool v_is_contiguous =
      (value_strides[0] == num_heads * seq_len_k * head_dim_v) &&
      (value_strides[1] == seq_len_k * head_dim_v) &&
      (value_strides[2] == head_dim_v) && (value_strides[3] == 1);

  printf("\nMemory Layout Check:\n");
  printf("  Q is contiguous: %s\n", q_is_contiguous ? "YES" : "NO");
  printf("  K is contiguous: %s\n", k_is_contiguous ? "YES" : "NO");
  printf("  V is contiguous: %s\n", v_is_contiguous ? "YES" : "NO");

  if (!q_is_contiguous || !k_is_contiguous || !v_is_contiguous) {
    printf(
        "  WARNING: Non-contiguous tensor detected! Kernel will use stride-based indexing.\n");
  }
  printf("==============================================\n\n");
  fflush(stdout);

  // Output strides (always contiguous)
  int64_t o_stride_batch = num_heads * seq_len_q * head_dim_v;
  int64_t o_stride_head = seq_len_q * head_dim_v;
  int64_t o_stride_seq = head_dim_v;
  int64_t o_stride_dim = 1;

  if (head_dim_v <= 64) {
    efficient_attention_kernel<scalar_t, 64><<<grid, block, 0, stream>>>(
        q_ptr,
        k_ptr,
        v_ptr,
        bias_ptr,
        out_ptr,
        num_heads,
        seq_len_q,
        seq_len_k,
        head_dim,
        head_dim_v,
        scale_factor,
        is_causal,
        // Query strides
        query_strides[0],
        query_strides[1],
        query_strides[2],
        query_strides[3],
        // Key strides
        key_strides[0],
        key_strides[1],
        key_strides[2],
        key_strides[3],
        // Value strides
        value_strides[0],
        value_strides[1],
        value_strides[2],
        value_strides[3],
        // Output strides
        o_stride_batch,
        o_stride_head,
        o_stride_seq,
        o_stride_dim,
        // Bias strides
        bias_stride_batch,
        bias_stride_head,
        bias_stride_q,
        bias_stride_k);
  } else {
    efficient_attention_kernel<scalar_t, 128><<<grid, block, 0, stream>>>(
        q_ptr,
        k_ptr,
        v_ptr,
        bias_ptr,
        out_ptr,
        num_heads,
        seq_len_q,
        seq_len_k,
        head_dim,
        head_dim_v,
        scale_factor,
        is_causal,
        // Query strides
        query_strides[0],
        query_strides[1],
        query_strides[2],
        query_strides[3],
        // Key strides
        key_strides[0],
        key_strides[1],
        key_strides[2],
        key_strides[3],
        // Value strides
        value_strides[0],
        value_strides[1],
        value_strides[2],
        value_strides[3],
        // Output strides
        o_stride_batch,
        o_stride_head,
        o_stride_seq,
        o_stride_dim,
        // Bias strides
        bias_stride_batch,
        bias_stride_head,
        bias_stride_q,
        bias_stride_k);
  }

  cudaError_t cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess) {
    ET_LOG(
        Error,
        "sdpa_efficient_attention: Kernel launch failed: %s",
        cudaGetErrorString(cuda_err));
    aoti_torch_delete_tensor_object(output);
    return nullptr;
  }

  // Synchronize to check for kernel execution errors
  cuda_err = cudaStreamSynchronize(stream);
  if (cuda_err != cudaSuccess) {
    ET_LOG(
        Error,
        "sdpa_efficient_attention: Kernel execution failed: %s",
        cudaGetErrorString(cuda_err));
    aoti_torch_delete_tensor_object(output);
    return nullptr;
  }

  printf("efficient_attention_kernel completed successfully\n");
  fflush(stdout);

  return output;
}

/**
 * Memory-Efficient Attention entry point with dtype dispatch
 */
Tensor* sdpa_efficient_attention(
    const Tensor* query,
    const Tensor* key,
    const Tensor* value,
    const Tensor* attn_bias,
    bool is_causal,
    double scale_factor,
    cudaStream_t stream) {
  auto dtype = query->dtype();

  if (dtype == executorch::aten::ScalarType::Float) {
    return sdpa_efficient_attention_impl<float>(
        query,
        key,
        value,
        attn_bias,
        is_causal,
        static_cast<float>(scale_factor),
        stream);
  } else if (dtype == executorch::aten::ScalarType::Half) {
    return sdpa_efficient_attention_impl<__half>(
        query,
        key,
        value,
        attn_bias,
        is_causal,
        static_cast<float>(scale_factor),
        stream);
  } else if (dtype == executorch::aten::ScalarType::BFloat16) {
    return sdpa_efficient_attention_impl<__nv_bfloat16>(
        query,
        key,
        value,
        attn_bias,
        is_causal,
        static_cast<float>(scale_factor),
        stream);
  } else {
    ET_LOG(Error, "sdpa_efficient_attention: Unsupported dtype");
    return nullptr;
  }
}

// ============================================================================
// Math Fallback Implementation
// ============================================================================

/**
 * Math fallback implementation for SDPA
 *
 * This implementation uses cuBLAS for matrix multiplications and custom
 * kernels for softmax. It provides maximum compatibility across all CUDA
 * devices but may not be as optimized as Flash Attention or Memory Efficient
 * Attention.
 *
 * Algorithm:
 * 1. Compute attention scores: S = (Q @ K^T)
 * 2. Apply scaling and compute softmax with optional causal mask
 * 3. Compute output: O = attention_weights @ V
 */
template <typename scalar_t>
Tensor* sdpa_math_fallback_impl(
    const Tensor* query,
    const Tensor* key,
    const Tensor* value,
    const Tensor* attn_mask,
    bool is_causal,
    float scale_factor,
    cudaStream_t stream) {
  printf("Inside sdpa_math_fallback_impl\n");
  fflush(stdout);

  // Get tensor dimensions
  printf("Getting tensor dimensions\n");
  fflush(stdout);
  const int64_t batch = query->size(0);
  const int64_t num_heads = query->size(1);
  const int64_t seq_len_q = query->size(2);
  const int64_t head_dim = query->size(3);
  const int64_t seq_len_k = key->size(2);
  const int64_t head_dim_v = value->size(3);

  printf(
      "Dimensions: batch=%ld, num_heads=%ld, seq_len_q=%ld, head_dim=%ld\n",
      batch,
      num_heads,
      seq_len_q,
      head_dim);
  fflush(stdout);

  // Get cuBLAS handle
  printf("About to get cuBLAS handle\n");
  fflush(stdout);
  cublasHandle_t handle = get_cublas_handle(stream);
  printf("Got cuBLAS handle: %p\n", (void*)handle);
  fflush(stdout);

  // Step 1: Allocate temporary buffer for attention scores
  // Shape: [batch, num_heads, seq_len_q, seq_len_k]
  const int64_t scores_size = batch * num_heads * seq_len_q * seq_len_k;
  scalar_t* scores_ptr = nullptr;
  cudaMalloc(&scores_ptr, scores_size * sizeof(scalar_t));
  if (scores_ptr == nullptr) {
    ET_LOG(Error, "sdpa_math_fallback: Failed to allocate scores buffer");
    return nullptr;
  }

  // Step 2: Compute Q @ K^T using cuBLAS
  // Q: [batch * num_heads, seq_len_q, head_dim]
  // K^T: [batch * num_heads, head_dim, seq_len_k]
  // Output: [batch * num_heads, seq_len_q, seq_len_k]

  const int m = seq_len_q;
  const int n = seq_len_k;
  const int k = head_dim;
  const int batch_count = batch * num_heads;

  const scalar_t alpha = static_cast<scalar_t>(1.0f);
  const scalar_t beta = static_cast<scalar_t>(0.0f);

  const scalar_t* q_ptr = reinterpret_cast<const scalar_t*>(query->data_ptr());
  const scalar_t* k_ptr = reinterpret_cast<const scalar_t*>(key->data_ptr());

  // Strides for batched GEMM
  const int64_t stride_q = seq_len_q * head_dim;
  const int64_t stride_k = seq_len_k * head_dim;
  const int64_t stride_scores = seq_len_q * seq_len_k;

  // Q @ K^T
  cublasStatus_t status = batched_gemm<scalar_t>(
      handle,
      CUBLAS_OP_T, // Transpose K
      CUBLAS_OP_N, // No transpose Q
      n, // seq_len_k
      m, // seq_len_q
      k, // head_dim
      &alpha,
      k_ptr,
      k, // K matrix
      stride_k,
      q_ptr,
      k, // Q matrix
      stride_q,
      &beta,
      scores_ptr,
      n, // Output scores
      stride_scores,
      batch_count);

  if (status != CUBLAS_STATUS_SUCCESS) {
    ET_LOG(Error, "sdpa_math_fallback: cuBLAS GEMM failed for Q @ K^T");
    cudaFree(scores_ptr);
    return nullptr;
  }

  // Step 3: Apply softmax with scaling, optional causal mask, and attention
  // bias
  const int threads_per_block = 256;
  const int64_t total_rows = batch * num_heads * seq_len_q;
  const int num_blocks =
      (total_rows + threads_per_block - 1) / threads_per_block;

  // Get attn_bias pointer if provided
  const scalar_t* bias_ptr = attn_mask
      ? reinterpret_cast<const scalar_t*>(attn_mask->data_ptr())
      : nullptr;

  printf("About to launch softmax kernel with attn_bias=%p\n", (void*)bias_ptr);
  fflush(stdout);

  softmax_with_mask_kernel<scalar_t>
      <<<num_blocks, threads_per_block, 0, stream>>>(
          scores_ptr,
          scores_ptr, // in-place
          bias_ptr, // attention bias (additive)
          batch,
          num_heads,
          seq_len_q,
          seq_len_k,
          is_causal,
          scale_factor);

  cudaError_t cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess) {
    printf(
        "sdpa_math_fallback: Softmax kernel launch failed: %s",
        cudaGetErrorString(cuda_err));
    fflush(stdout);
    ET_LOG(
        Error,
        "sdpa_math_fallback: Softmax kernel launch failed: %s",
        cudaGetErrorString(cuda_err));
    cudaFree(scores_ptr);
    return nullptr;
  }

  // Step 4: Allocate output tensor [batch, num_heads, seq_len_q, head_dim_v]
  Tensor* output = nullptr;
  std::array<int64_t, 4> output_shape = {
      batch, num_heads, seq_len_q, head_dim_v};
  std::array<int64_t, 4> output_stride = {
      num_heads * seq_len_q * head_dim_v,
      seq_len_q * head_dim_v,
      head_dim_v,
      1};

  auto dtype_int = static_cast<int32_t>(query->dtype());
  aoti_torch_empty_strided(
      4,
      output_shape.data(),
      output_stride.data(),
      dtype_int,
      static_cast<int32_t>(SupportedDevices::CUDA),
      0,
      &output);

  if (output == nullptr) {
    printf("sdpa_math_fallback: Failed to allocate output tensor");
    fflush(stdout);
    ET_LOG(Error, "sdpa_math_fallback: Failed to allocate output tensor");
    cudaFree(scores_ptr);
    return nullptr;
  }

  // Step 5: Compute attention_weights @ V
  // attention_weights: [batch * num_heads, seq_len_q, seq_len_k]
  // V: [batch * num_heads, seq_len_k, head_dim_v]
  // Output: [batch * num_heads, seq_len_q, head_dim_v]

  const int m_v = seq_len_q;
  const int n_v = head_dim_v;
  const int k_v = seq_len_k;

  const scalar_t* v_ptr = reinterpret_cast<const scalar_t*>(value->data_ptr());
  scalar_t* out_ptr = reinterpret_cast<scalar_t*>(output->data_ptr());

  const int64_t stride_v = seq_len_k * head_dim_v;
  const int64_t stride_out = seq_len_q * head_dim_v;

  status = batched_gemm<scalar_t>(
      handle,
      CUBLAS_OP_N, // No transpose V
      CUBLAS_OP_N, // No transpose attention_weights
      n_v, // head_dim_v
      m_v, // seq_len_q
      k_v, // seq_len_k
      &alpha,
      v_ptr,
      n_v, // V matrix
      stride_v,
      scores_ptr,
      k_v, // attention_weights
      stride_scores,
      &beta,
      out_ptr,
      n_v, // Output
      stride_out,
      batch_count);

  // Cleanup temporary buffers
  cudaFree(scores_ptr);

  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("sdpa_math_fallback: cuBLAS GEMM failed for attention_weights @ V");
    fflush(stdout);
    ET_LOG(
        Error,
        "sdpa_math_fallback: cuBLAS GEMM failed for attention_weights @ V");
    aoti_torch_delete_tensor_object(output);
    return nullptr;
  }

  return output;
}

Tensor* sdpa_math_fallback(
    const Tensor* query,
    const Tensor* key,
    const Tensor* value,
    const Tensor* attn_mask,
    bool is_causal,
    double scale_factor,
    cudaStream_t stream) {
  printf("Inside sdpa_math_fallback\n");
  fflush(stdout);

  // Dispatch based on dtype
  auto dtype = query->dtype();
  printf("Query dtype: %d\n", static_cast<int>(dtype));
  fflush(stdout);

  if (dtype == executorch::aten::ScalarType::Float) {
    printf("Calling sdpa_math_fallback_impl<float>\n");
    fflush(stdout);
    return sdpa_math_fallback_impl<float>(
        query,
        key,
        value,
        attn_mask,
        is_causal,
        static_cast<float>(scale_factor),
        stream);
  } else if (dtype == executorch::aten::ScalarType::Half) {
    printf("Calling sdpa_math_fallback_impl<half>\n");
    fflush(stdout);
    return sdpa_math_fallback_impl<__half>(
        query,
        key,
        value,
        attn_mask,
        is_causal,
        static_cast<float>(scale_factor),
        stream);
  } else if (dtype == executorch::aten::ScalarType::BFloat16) {
    printf("Calling sdpa_math_fallback_impl<bfloat16>\n");
    fflush(stdout);
    return sdpa_math_fallback_impl<__nv_bfloat16>(
        query,
        key,
        value,
        attn_mask,
        is_causal,
        static_cast<float>(scale_factor),
        stream);
  } else {
    ET_LOG(Error, "sdpa_math_fallback: Unsupported dtype");
    return nullptr;
  }
}

/**
 * Main entry point for SDPA computation
 */
Tensor* scaled_dot_product_attention_cuda(
    const Tensor* query,
    const Tensor* key,
    const Tensor* value,
    const Tensor* attn_mask,
    double dropout_p,
    bool is_causal,
    const double* scale,
    bool enable_gqa,
    cudaStream_t stream) {
  // Select backend
  SDPBackend backend =
      select_sdp_backend(query, key, value, attn_mask, dropout_p, is_causal);

  if (backend == SDPBackend::Error) {
    ET_LOG(
        Error, "scaled_dot_product_attention_cuda: No valid backend selected");
    return nullptr;
  }

  printf("selected backend: %d\n", static_cast<int>(backend));
  fflush(stdout);

  // Calculate scale factor
  printf("About to calculate scale factor\n");
  fflush(stdout);
  double scale_factor = calculate_scale(query, scale);
  printf("Calculated scale factor: %f\n", scale_factor);
  fflush(stdout);

  printf("enable_gqa: %d\n", enable_gqa);
  fflush(stdout);

  // Handle GQA if needed
  printf("About to check GQA configuration\n");
  fflush(stdout);

  if (enable_gqa && is_gqa_configuration(query, key, value)) {
    printf("GQA configuration detected\n");
    fflush(stdout);

    if (!validate_gqa(query, key, value)) {
      ET_LOG(
          Error,
          "scaled_dot_product_attention_cuda: Invalid GQA configuration");
      return nullptr;
    }
    ET_LOG(
        Error,
        "scaled_dot_product_attention_cuda: GQA support not yet implemented. "
        "Need to repeat K/V heads to match Q heads.");
    return nullptr;
  }

  printf("Passed GQA check\n");
  fflush(stdout);

  printf(
      "About to enter switch statement, backend = %d\n",
      static_cast<int>(backend));
  fflush(stdout);

  // Dispatch to appropriate backend
  switch (backend) {
    case SDPBackend::Math:
      printf("In Math case, about to call sdpa_math_fallback\n");
      fflush(stdout);
      return sdpa_math_fallback(
          query, key, value, attn_mask, is_causal, scale_factor, stream);

    case SDPBackend::FlashAttention:
      printf("In FlashAttention case\n");
      fflush(stdout);
      return sdpa_flash_attention(
          query, key, value, attn_mask, is_causal, scale_factor, stream);

    case SDPBackend::MemoryEfficientAttention:
      printf("Memory Efficient Attention backend\n");
      fflush(stdout);
      return sdpa_efficient_attention(
          query, key, value, attn_mask, is_causal, scale_factor, stream);

    case SDPBackend::CuDNN:
      printf("cuDNN backend not yet implemented\n");
      fflush(stdout);
      return nullptr;

    default:
      printf("Unknown SDPA backend\n");
      fflush(stdout);
      return nullptr;
  }
}

// ============================================================================
// C API Implementation
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

AOTITorchError aoti_torch_cuda__scaled_dot_product_flash_attention(
    Tensor* query,
    Tensor* key,
    Tensor* value,
    double dropout_p,
    int32_t is_causal,
    int32_t return_debug_mask,
    double* scale,
    Tensor** ret0, // output
    Tensor** ret1, // logsumexp (nullptr for inference)
    Tensor** ret2, // cum_seq_q (nullptr for inference)
    Tensor** ret3, // cum_seq_k (nullptr for inference)
    int64_t* max_seqlen_q,
    int64_t* max_seqlen_k,
    Tensor** ret4, // philox_seed (nullptr for inference)
    Tensor** ret5, // philox_offset (nullptr for inference)
    Tensor** ret6) { // debug_attn_mask (nullptr for inference)
  // Input validation
  if (!query || !key || !value || !ret0) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_flash_attention: Null pointer input");
    return Error::InvalidArgument;
  }

  // Currently only support dropout_p = 0.0 for inference
  if (dropout_p != 0.0) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_flash_attention: dropout_p != 0.0 is not supported");
    return Error::InvalidArgument;
  }

  // Check tensor dimensions
  if (query->dim() != 4 || key->dim() != 4 || value->dim() != 4) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_flash_attention: Query, Key, Value must be 4D tensors");
    return Error::InvalidArgument;
  }

  // Check that Q, K, V have the same dtype
  if (query->dtype() != key->dtype() || query->dtype() != value->dtype()) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_flash_attention: Query, Key, Value must have the same dtype");
    return Error::InvalidArgument;
  }

  // Check dtype support
  if (!is_supported_dtype(query) || !is_supported_dtype(key) ||
      !is_supported_dtype(value)) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_flash_attention: Unsupported dtype, only Float32/Float16/BFloat16 supported");
    return Error::InvalidArgument;
  }

  // Check tensor shapes
  const int64_t batch = query->size(0);
  const int64_t num_heads = query->size(1);
  const int64_t seq_len_q = query->size(2);
  const int64_t head_dim_q = query->size(3);

  const int64_t num_heads_kv = key->size(1);
  const int64_t seq_len_k = key->size(2);
  const int64_t head_dim_k = key->size(3);

  const int64_t seq_len_v = value->size(2);
  const int64_t head_dim_v = value->size(3);

  // Validate shapes
  if (key->size(0) != batch || value->size(0) != batch) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_flash_attention: Batch size mismatch");
    return Error::InvalidArgument;
  }

  if (seq_len_k != seq_len_v) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_flash_attention: Key and Value sequence length mismatch");
    return Error::InvalidArgument;
  }

  if (head_dim_q != head_dim_k) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_flash_attention: Query and Key head dimension mismatch");
    return Error::InvalidArgument;
  }

  if (value->size(1) != num_heads_kv) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_flash_attention: Key and Value num_heads mismatch");
    return Error::InvalidArgument;
  }

  // Determine if GQA is being used
  bool enable_gqa = (num_heads != num_heads_kv);

  // GQA validation and check
  if (enable_gqa) {
    if (num_heads % num_heads_kv != 0) {
      ET_LOG(
          Error,
          "aoti_torch_cuda__scaled_dot_product_flash_attention: For GQA, num_heads must be divisible by num_heads_kv");
      return Error::InvalidArgument;
    }
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_flash_attention: GQA support not yet implemented");
    return Error::InvalidArgument;
  }

  // Check if flash attention can be used
  if (!supports_flash_attention()) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_flash_attention: Flash Attention not supported on this GPU");
    return Error::InvalidArgument;
  }

  if (!can_use_flash_attention(query, key, value, nullptr, is_causal != 0)) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_flash_attention: Input conditions not suitable for Flash Attention");
    return Error::InvalidArgument;
  }

  // Calculate scale factor
  double scale_factor = calculate_scale(query, scale);

  // Get CUDA stream
  auto stream_result = getCurrentCUDAStream(0);
  if (!stream_result.ok()) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_flash_attention: Failed to get CUDA stream");
    return Error::Internal;
  }
  cudaStream_t stream = stream_result.get();

  // Call flash attention directly
  Tensor* output = sdpa_flash_attention(
      query,
      key,
      value,
      nullptr, // attn_mask - Flash Attention doesn't support it
      is_causal != 0,
      scale_factor,
      stream);

  if (output == nullptr) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_flash_attention: Flash Attention computation failed");
    return Error::Internal;
  }

  // Set the main output
  *ret0 = output;

  // Set all training-related outputs to nullptr (for inference)
  if (ret1)
    *ret1 = nullptr; // logsumexp
  if (ret2)
    *ret2 = nullptr; // cum_seq_q
  if (ret3)
    *ret3 = nullptr; // cum_seq_k
  if (ret4)
    *ret4 = nullptr; // philox_seed
  if (ret5)
    *ret5 = nullptr; // philox_offset
  if (ret6)
    *ret6 = nullptr; // debug_attn_mask

  return Error::Ok;
}

AOTITorchError aoti_torch_cuda__scaled_dot_product_efficient_attention(
    Tensor* query,
    Tensor* key,
    Tensor* value,
    Tensor** attn_bias, // Optional attention bias (can be nullptr)
    int32_t compute_log_sumexp,
    double dropout_p,
    int32_t is_causal,
    double* scale,
    Tensor** ret0, // output
    Tensor** ret1, // logsumexp (nullptr for inference)
    Tensor** ret2, // philox_seed (nullptr for inference)
    Tensor** ret3) { // philox_offset (nullptr for inference)

  // Input validation
  if (!query || !key || !value || !ret0) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_efficient_attention: Null pointer input");
    return Error::InvalidArgument;
  }

  // Currently only support dropout_p = 0.0 for inference
  if (dropout_p != 0.0) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_efficient_attention: dropout_p != 0.0 is not supported");
    return Error::InvalidArgument;
  }

  // Check tensor dimensions
  if (query->dim() != 4 || key->dim() != 4 || value->dim() != 4) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_efficient_attention: Query, Key, Value must be 4D tensors");
    return Error::InvalidArgument;
  }

  // Check that Q, K, V have the same dtype
  if (query->dtype() != key->dtype() || query->dtype() != value->dtype()) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_efficient_attention: Query, Key, Value must have the same dtype");
    return Error::InvalidArgument;
  }

  // Check dtype support
  if (!is_supported_dtype(query) || !is_supported_dtype(key) ||
      !is_supported_dtype(value)) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_efficient_attention: Unsupported dtype, only Float32/Float16/BFloat16 supported");
    return Error::InvalidArgument;
  }

  // Check tensor shapes
  const int64_t batch = query->size(0);
  const int64_t num_heads = query->size(1);
  const int64_t seq_len_q = query->size(2);
  const int64_t head_dim_q = query->size(3);

  const int64_t num_heads_kv = key->size(1);
  const int64_t seq_len_k = key->size(2);
  const int64_t head_dim_k = key->size(3);

  const int64_t seq_len_v = value->size(2);
  const int64_t head_dim_v = value->size(3);

  // Validate shapes
  if (key->size(0) != batch || value->size(0) != batch) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_efficient_attention: Batch size mismatch");
    return Error::InvalidArgument;
  }

  if (seq_len_k != seq_len_v) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_efficient_attention: Key and Value sequence length mismatch");
    return Error::InvalidArgument;
  }

  if (head_dim_q != head_dim_k) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_efficient_attention: Query and Key head dimension mismatch");
    return Error::InvalidArgument;
  }

  if (value->size(1) != num_heads_kv) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_efficient_attention: Key and Value num_heads mismatch");
    return Error::InvalidArgument;
  }

  // Determine if GQA is being used
  bool enable_gqa = (num_heads != num_heads_kv);

  // GQA validation and check
  if (enable_gqa) {
    if (num_heads % num_heads_kv != 0) {
      ET_LOG(
          Error,
          "aoti_torch_cuda__scaled_dot_product_efficient_attention: For GQA, num_heads must be divisible by num_heads_kv");
      return Error::InvalidArgument;
    }
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_efficient_attention: GQA support not yet implemented");
    return Error::InvalidArgument;
  }

  // Extract attn_bias tensor if provided
  Tensor* attn_bias_tensor = (attn_bias && *attn_bias) ? *attn_bias : nullptr;

  // Check if efficient attention can be used
  if (!can_use_efficient_attention(query, key, value, attn_bias_tensor, is_causal != 0)) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_efficient_attention: Input conditions not suitable for Efficient Attention");
    return Error::InvalidArgument;
  }

  // Calculate scale factor
  double scale_factor = calculate_scale(query, scale);

  // Get CUDA stream
  auto stream_result = getCurrentCUDAStream(0);
  if (!stream_result.ok()) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_efficient_attention: Failed to get CUDA stream");
    return Error::Internal;
  }
  cudaStream_t stream = stream_result.get();

  // Call efficient attention directly
  Tensor* output = sdpa_efficient_attention(
      query,
      key,
      value,
      attn_bias_tensor, // Pass attn_bias (can be nullptr)
      is_causal != 0,
      scale_factor,
      stream);

  if (output == nullptr) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_efficient_attention: Efficient Attention computation failed");
    return Error::Internal;
  }

  // Set the main output
  *ret0 = output;

  // Set all training-related outputs to nullptr (for inference)
  if (ret1)
    *ret1 = nullptr; // logsumexp
  if (ret2)
    *ret2 = nullptr; // philox_seed
  if (ret3)
    *ret3 = nullptr; // philox_offset

  return Error::Ok;
}

#ifdef __cplusplus
}
#endif

} // namespace executorch::backends::cuda
