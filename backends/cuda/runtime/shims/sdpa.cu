/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cfloat>
#include <cmath>

#include <executorch/backends/aoti/utils.h>
#include <executorch/backends/cuda/runtime/shims/sdpa.h>
#include <executorch/backends/cuda/runtime/shims/sdpa.cuh>
#include <executorch/backends/cuda/runtime/utils.h>
#include <executorch/runtime/platform/log.h>

namespace executorch::backends::cuda {

using executorch::backends::aoti::Tensor;
using executorch::backends::aoti::AOTITorchError;
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

__device__ __forceinline__ __nv_bfloat16 device_max(__nv_bfloat16 a, __nv_bfloat16 b) {
  #if __CUDA_ARCH__ >= 800
  return __hgt(a, b) ? a : b;
  #else
  return __float2bfloat16(fmaxf(__bfloat162float(a), __bfloat162float(b)));
  #endif
}

/**
 * Softmax kernel with optional causal masking
 *
 * Computes softmax along the last dimension (seq_len_k) of a 4D tensor.
 * Supports causal masking where positions j > i are masked out.
 *
 * Input: [batch, num_heads, seq_len_q, seq_len_k]
 * Output: [batch, num_heads, seq_len_q, seq_len_k]
 *
 * Each thread processes one row (seq_len_q position).
 *
 * Note: Supports in-place operation (input == output).
 */
template<typename scalar_t>
__global__ void softmax_with_causal_mask_kernel(
    const scalar_t* input,
    scalar_t* output,
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

  // Find max for numerical stability (two-pass algorithm)
  float max_val = -FLT_MAX;
  for (int64_t j = 0; j < seq_len_k; ++j) {
    if (!is_causal || j <= i) {
      float val = static_cast<float>(input_row[j]) * scale;
      max_val = fmaxf(max_val, val);
    }
  }

  // Compute exp and sum
  float sum_exp = 0.0f;
  for (int64_t j = 0; j < seq_len_k; ++j) {
    float val;
    if (!is_causal || j <= i) {
      val = expf(static_cast<float>(input_row[j]) * scale - max_val);
    } else {
      val = 0.0f;
    }
    output_row[j] = static_cast<scalar_t>(val);
    sum_exp += val;
  }

  // Normalize
  const float inv_sum = 1.0f / sum_exp;
  for (int64_t j = 0; j < seq_len_k; ++j) {
    output_row[j] = static_cast<scalar_t>(static_cast<float>(output_row[j]) * inv_sum);
  }
}

/**
 * Scale kernel - multiply all elements by a scalar
 */
template<typename scalar_t>
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

  if (handle == nullptr) {
    cublasCreate(&handle);
  }

  cublasSetStream(handle, stream);
  return handle;
}

/**
 * Batched matrix multiplication wrapper for cuBLAS
 *
 * Computes: C = alpha * op(A) @ op(B) + beta * C
 * for a batch of matrices
 */
template<typename scalar_t>
cublasStatus_t batched_gemm(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const scalar_t* alpha,
    const scalar_t* A, int lda, int64_t strideA,
    const scalar_t* B, int ldb, int64_t strideB,
    const scalar_t* beta,
    scalar_t* C, int ldc, int64_t strideC,
    int batchCount);

// Specializations for different types
template<>
cublasStatus_t batched_gemm<float>(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* A, int lda, int64_t strideA,
    const float* B, int ldb, int64_t strideB,
    const float* beta,
    float* C, int ldc, int64_t strideC,
    int batchCount) {
  return cublasSgemmStridedBatched(
      handle, transa, transb, m, n, k,
      alpha, A, lda, strideA, B, ldb, strideB,
      beta, C, ldc, strideC, batchCount);
}

template<>
cublasStatus_t batched_gemm<__half>(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const __half* alpha,
    const __half* A, int lda, int64_t strideA,
    const __half* B, int ldb, int64_t strideB,
    const __half* beta,
    __half* C, int ldc, int64_t strideC,
    int batchCount) {
  return cublasHgemmStridedBatched(
      handle, transa, transb, m, n, k,
      alpha, A, lda, strideA, B, ldb, strideB,
      beta, C, ldc, strideC, batchCount);
}

// Note: BFloat16 uses compute type float internally
template<>
cublasStatus_t batched_gemm<__nv_bfloat16>(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const __nv_bfloat16* alpha,
    const __nv_bfloat16* A, int lda, int64_t strideA,
    const __nv_bfloat16* B, int ldb, int64_t strideB,
    const __nv_bfloat16* beta,
    __nv_bfloat16* C, int ldc, int64_t strideC,
    int batchCount) {

  // cuBLAS BFloat16 GEMM - introduced in CUDA 11+
  #if CUDA_VERSION >= 11000
  // For BFloat16, we need to use cublasGemmStridedBatchedEx
  // with compute type CUBLAS_COMPUTE_32F
  float alpha_f = 1.0f;
  float beta_f = 0.0f;

  return cublasGemmStridedBatchedEx(
      handle,
      transa, transb,
      m, n, k,
      &alpha_f,
      A, CUDA_R_16BF, lda, strideA,
      B, CUDA_R_16BF, ldb, strideB,
      &beta_f,
      C, CUDA_R_16BF, ldc, strideC,
      batchCount,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT);
  #else
  ET_LOG(Error, "BFloat16 GEMM requires CUDA 11.0 or later");
  return CUBLAS_STATUS_NOT_SUPPORTED;
  #endif
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
template<typename scalar_t>
Tensor* sdpa_math_fallback_impl(
    const Tensor* query,
    const Tensor* key,
    const Tensor* value,
    const Tensor* attn_mask,
    bool is_causal,
    float scale_factor,
    cudaStream_t stream) {

  // Get tensor dimensions
  const int64_t batch = query->size(0);
  const int64_t num_heads = query->size(1);
  const int64_t seq_len_q = query->size(2);
  const int64_t head_dim = query->size(3);
  const int64_t seq_len_k = key->size(2);
  const int64_t head_dim_v = value->size(3);

  // Get cuBLAS handle
  cublasHandle_t handle = get_cublas_handle(stream);

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
      CUBLAS_OP_T,  // Transpose K
      CUBLAS_OP_N,  // No transpose Q
      n,  // seq_len_k
      m,  // seq_len_q
      k,  // head_dim
      &alpha,
      k_ptr, k,  // K matrix
      stride_k,
      q_ptr, k,  // Q matrix
      stride_q,
      &beta,
      scores_ptr, n,  // Output scores
      stride_scores,
      batch_count);

  if (status != CUBLAS_STATUS_SUCCESS) {
    ET_LOG(Error, "sdpa_math_fallback: cuBLAS GEMM failed for Q @ K^T");
    cudaFree(scores_ptr);
    return nullptr;
  }

  // Step 3: Apply softmax with scaling and optional causal mask
  const int threads_per_block = 256;
  const int64_t total_rows = batch * num_heads * seq_len_q;
  const int num_blocks = (total_rows + threads_per_block - 1) / threads_per_block;

  softmax_with_causal_mask_kernel<scalar_t><<<num_blocks, threads_per_block, 0, stream>>>(
      scores_ptr,
      scores_ptr,  // in-place
      batch,
      num_heads,
      seq_len_q,
      seq_len_k,
      is_causal,
      scale_factor);

  cudaError_t cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess) {
    ET_LOG(Error, "sdpa_math_fallback: Softmax kernel launch failed: %s",
           cudaGetErrorString(cuda_err));
    cudaFree(scores_ptr);
    return nullptr;
  }

  // Step 4: Allocate output tensor [batch, num_heads, seq_len_q, head_dim_v]
  Tensor* output = nullptr;
  std::array<int64_t, 4> output_shape = {batch, num_heads, seq_len_q, head_dim_v};
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
      CUBLAS_OP_N,  // No transpose V
      CUBLAS_OP_N,  // No transpose attention_weights
      n_v,  // head_dim_v
      m_v,  // seq_len_q
      k_v,  // seq_len_k
      &alpha,
      v_ptr, n_v,  // V matrix
      stride_v,
      scores_ptr, k_v,  // attention_weights
      stride_scores,
      &beta,
      out_ptr, n_v,  // Output
      stride_out,
      batch_count);

  // Cleanup temporary buffers
  cudaFree(scores_ptr);

  if (status != CUBLAS_STATUS_SUCCESS) {
    ET_LOG(Error, "sdpa_math_fallback: cuBLAS GEMM failed for attention_weights @ V");
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

  // Dispatch based on dtype
  auto dtype = query->dtype();

  if (dtype == executorch::aten::ScalarType::Float) {
    return sdpa_math_fallback_impl<float>(
        query, key, value, attn_mask, is_causal,
        static_cast<float>(scale_factor), stream);
  } else if (dtype == executorch::aten::ScalarType::Half) {
    return sdpa_math_fallback_impl<__half>(
        query, key, value, attn_mask, is_causal,
        static_cast<float>(scale_factor), stream);
  } else if (dtype == executorch::aten::ScalarType::BFloat16) {
    return sdpa_math_fallback_impl<__nv_bfloat16>(
        query, key, value, attn_mask, is_causal,
        static_cast<float>(scale_factor), stream);
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
  SDPBackend backend = select_sdp_backend(
      query, key, value, attn_mask, dropout_p, is_causal);

  if (backend == SDPBackend::Error) {
    ET_LOG(Error, "scaled_dot_product_attention_cuda: No valid backend selected");
    return nullptr;
  }

  // Calculate scale factor
  double scale_factor = calculate_scale(query, scale);

  // Handle GQA if needed
  if (enable_gqa && is_gqa_configuration(query, key, value)) {
    if (!validate_gqa(query, key, value)) {
      ET_LOG(Error, "scaled_dot_product_attention_cuda: Invalid GQA configuration");
      return nullptr;
    }
    ET_LOG(
        Error,
        "scaled_dot_product_attention_cuda: GQA support not yet implemented. "
        "Need to repeat K/V heads to match Q heads.");
    return nullptr;
  }

  // Dispatch to appropriate backend
  switch (backend) {
    case SDPBackend::Math:
      return sdpa_math_fallback(
          query, key, value, attn_mask, is_causal, scale_factor, stream);

    case SDPBackend::FlashAttention:
      ET_LOG(Error, "Flash Attention backend not yet implemented");
      return nullptr;

    case SDPBackend::MemoryEfficientAttention:
      ET_LOG(Error, "Memory Efficient Attention backend not yet implemented");
      return nullptr;

    case SDPBackend::CuDNN:
      ET_LOG(Error, "cuDNN backend not yet implemented");
      return nullptr;

    default:
      ET_LOG(Error, "Unknown SDPA backend");
      return nullptr;
  }
}

// ============================================================================
// C API Implementation
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

AOTITorchError aoti_torch_cuda_scaled_dot_product_attention(
    Tensor* query,
    Tensor* key,
    Tensor* value,
    Tensor* attn_mask,
    double dropout_p,
    int32_t is_causal,
    double* scale,
    int32_t enable_gqa,
    Tensor** ret0) {

  // Input validation
  if (!query || !key || !value || !ret0) {
    ET_LOG(Error, "aoti_torch_cuda_scaled_dot_product_attention: Null pointer input");
    return Error::InvalidArgument;
  }

  // Currently only support dropout_p = 0.0 for inference
  if (dropout_p != 0.0) {
    ET_LOG(Error, "aoti_torch_cuda_scaled_dot_product_attention: dropout_p != 0.0 is not supported");
    return Error::InvalidArgument;
  }

  // Check tensor dimensions
  if (query->dim() != 4 || key->dim() != 4 || value->dim() != 4) {
    ET_LOG(Error, "aoti_torch_cuda_scaled_dot_product_attention: Query, Key, Value must be 4D tensors");
    return Error::InvalidArgument;
  }

  // Check that Q, K, V have the same dtype
  if (query->dtype() != key->dtype() || query->dtype() != value->dtype()) {
    ET_LOG(Error, "aoti_torch_cuda_scaled_dot_product_attention: Query, Key, Value must have the same dtype");
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
    ET_LOG(Error, "aoti_torch_cuda_scaled_dot_product_attention: Batch size mismatch");
    return Error::InvalidArgument;
  }

  if (seq_len_k != seq_len_v) {
    ET_LOG(Error, "aoti_torch_cuda_scaled_dot_product_attention: Key and Value sequence length mismatch");
    return Error::InvalidArgument;
  }

  if (head_dim_q != head_dim_k) {
    ET_LOG(Error, "aoti_torch_cuda_scaled_dot_product_attention: Query and Key head dimension mismatch");
    return Error::InvalidArgument;
  }

  if (value->size(1) != num_heads_kv) {
    ET_LOG(Error, "aoti_torch_cuda_scaled_dot_product_attention: Key and Value num_heads mismatch");
    return Error::InvalidArgument;
  }

  // GQA validation
  if (enable_gqa && num_heads % num_heads_kv != 0) {
    ET_LOG(Error, "aoti_torch_cuda_scaled_dot_product_attention: For GQA, num_heads must be divisible by num_heads_kv");
    return Error::InvalidArgument;
  }

  // Validate attn_mask if provided
  if (attn_mask) {
    if (is_causal) {
      ET_LOG(Error, "aoti_torch_cuda_scaled_dot_product_attention: Cannot use both attn_mask and is_causal");
      return Error::InvalidArgument;
    }
  }

  // Get CUDA stream
  auto stream_result = getCurrentCUDAStream(0);
  if (!stream_result.ok()) {
    ET_LOG(Error, "aoti_torch_cuda_scaled_dot_product_attention: Failed to get CUDA stream");
    return Error::Internal;
  }
  cudaStream_t stream = stream_result.get();

  // Call the main SDPA function
  Tensor* output = scaled_dot_product_attention_cuda(
      query,
      key,
      value,
      attn_mask,
      dropout_p,
      is_causal != 0,
      scale,
      enable_gqa != 0,
      stream);

  if (output == nullptr) {
    ET_LOG(Error, "aoti_torch_cuda_scaled_dot_product_attention: SDPA computation failed");
    return Error::Internal;
  }

  *ret0 = output;
  return Error::Ok;
}

#ifdef __cplusplus
}
#endif

} // namespace executorch::backends::cuda
