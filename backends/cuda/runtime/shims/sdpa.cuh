/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// This file implements scaled_dot_product_attention for ExecuTorch.
//
// IMPLEMENTATION NOTES:
// ---------------------
// This is NOT a direct port from PyTorch. Instead, we implemented
// a custom Math Fallback using cuBLAS and custom CUDA kernels.
//
// PyTorch reference implementations (for architecture reference only):
// - CPU/General: aten/src/ATen/native/transformers/attention.cpp
// - CUDA: aten/src/ATen/native/transformers/cuda/attention.cu
//
// Key differences from PyTorch:
// - PyTorch uses high-level ATen ops (at::matmul, at::_safe_softmax)
// - We use direct cuBLAS calls and custom softmax kernels
// - Optimized for inference (no dropout, no backward pass)
// - Simplified memory management
// - No ATen/c10 dependencies
//
// PORTING NOTES:
// --------------
// 1. KERNEL CODE: Adapted from PyTorch attention kernels
//    - Math fallback implementation for maximum compatibility
//    - Supports Float32, Float16, and BFloat16 dtypes
//    - Standard attention computation: softmax(Q @ K^T / scale) @ V
//
// 2. API ADAPTATIONS:
//    - Replaced at::Tensor with executorch::backends::aoti::Tensor
//    - Output returned via pointer-to-pointer instead of by-value
//    - Simplified interface for inference (dropout=0.0 only)
//
// 3. REMOVED FEATURES:
//    - Flash Attention backend (requires external library)
//    - Memory Efficient Attention backend (requires external library)
//    - cuDNN backend (requires cuDNN library)
//    - Dropout support (training-only feature)
//    - Nested tensor support (complex layout)
//    - Backward pass (training-only feature)
//
// 4. INFRASTRUCTURE CHANGES:
//    - Removed c10::cuda::CUDAGuard: Device management handled by AOTI backend
//    - Removed at::cuda::getCurrentCUDAStream(): Stream passed explicitly
//    - Simplified error handling using ExecutorTorch Error codes

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

#include <executorch/backends/aoti/common_shims.h>
#include <executorch/backends/aoti/utils.h>
#include <executorch/backends/cuda/runtime/guard.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/backends/cuda/runtime/utils.h>
#include <executorch/runtime/platform/log.h>

namespace executorch::backends::cuda {

using executorch::backends::aoti::Tensor;
using executorch::runtime::Error;

// ============================================================================
// Utility Functions for SDPA
// ============================================================================

// Calculate the scaling factor for attention scores
inline double calculate_scale(const Tensor* query, const double* scale) {
  if (scale != nullptr) {
    return *scale;
  }
  // Default: 1 / sqrt(head_dim)
  // Query shape: [batch, num_heads, seq_len_q, head_dim]
  // head_dim is at index 3 (0-indexed)
  const int64_t head_dim = query->size(3);
  return 1.0 / std::sqrt(static_cast<double>(head_dim));
}

// Check if tensor dtype is supported for SDPA
inline bool is_supported_dtype(const Tensor* tensor) {
  auto dtype = tensor->dtype();
  return dtype == executorch::aten::ScalarType::Float ||
      dtype == executorch::aten::ScalarType::Half ||
      dtype == executorch::aten::ScalarType::BFloat16;
}

// ============================================================================
// Math Fallback Implementation
// ============================================================================

// This is the basic, portable implementation that works on all CUDA devices.
// It computes attention using explicit matrix multiplications and softmax:
//   1. Compute scores: S = Q @ K^T * scale
//   2. Apply mask if provided
//   3. Compute attention weights: A = softmax(S)
//   4. Compute output: O = A @ V

/**
 * Math fallback kernel for scaled dot product attention
 *
 * This is a basic implementation that performs:
 * output = softmax(query @ key^T / scale) @ value
 *
 * Supports:
 * - Batch processing
 * - Multiple attention heads
 * - Optional causal masking
 * - Optional explicit attention mask
 * - Float32, Float16, BFloat16 dtypes
 *
 * Note: This implementation is for reference and maximum compatibility.
 * For production use, consider using Flash Attention or other optimized
 * backends.
 */
Tensor* sdpa_math_fallback(
    const Tensor* query, // [batch, num_heads, seq_len_q, head_dim]
    const Tensor* key, // [batch, num_heads_kv, seq_len_k, head_dim]
    const Tensor* value, // [batch, num_heads_kv, seq_len_k, head_dim_v]
    const Tensor* attn_mask, // Optional: [batch, num_heads, seq_len_q,
                             // seq_len_k] or broadcastable
    bool is_causal, // Apply causal masking
    double scale_factor, // Scaling factor for attention scores
    cudaStream_t stream); // CUDA stream for execution

// ============================================================================
// Memory-Efficient Attention Implementation
// ============================================================================

/**
 * Memory-Efficient Attention kernel for scaled dot product attention
 *
 * This implementation uses online softmax to compute attention efficiently
 * without materializing the full attention matrix. Supports attention bias.
 *
 * Supports:
 * - Batch processing
 * - Multiple attention heads
 * - Optional causal masking
 * - Optional attention bias (additive)
 * - Float32, Float16, BFloat16 dtypes
 */
Tensor* sdpa_efficient_attention(
    const Tensor* query, // [batch, num_heads, seq_len_q, head_dim]
    const Tensor* key, // [batch, num_heads_kv, seq_len_k, head_dim]
    const Tensor* value, // [batch, num_heads_kv, seq_len_k, head_dim_v]
    const Tensor* attn_bias, // Optional: [batch, num_heads, seq_len_q,
                             // seq_len_k] or broadcastable
    bool is_causal, // Apply causal masking
    double scale_factor, // Scaling factor for attention scores
    cudaStream_t stream); // CUDA stream for execution

// ============================================================================
// Backend Selection
// ============================================================================

enum class SDPBackend {
  Error = -1,
  Math = 0,
  FlashAttention = 1,
  MemoryEfficientAttention = 2,
  CuDNN = 3
};

/**
 * Check if Flash Attention is supported on the current GPU
 */
inline bool supports_flash_attention() {
  int device;
  cudaGetDevice(&device);

  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device);

  // Flash Attention typically requires compute capability >= 7.0 (Volta+)
  // For better performance, prefer Ampere+ (8.0+)
  // We'll use a conservative threshold of 7.0 for compatibility
  return props.major >= 7;
}

/**
 * Check if inputs are suitable for Flash Attention
 */
inline bool can_use_flash_attention(
    const Tensor* query,
    const Tensor* key,
    const Tensor* value,
    const Tensor* attn_mask,
    bool is_causal) {
  // Flash Attention doesn't support explicit attention masks (yet)
  // It only supports causal masking
  if (attn_mask != nullptr) {
    return false;
  }

  // Check head dimensions are reasonable for Flash Attention
  const int64_t head_dim = query->size(3);
  const int64_t head_dim_v = value->size(3);

  // Flash Attention works best with head_dim <= 128
  // Our simple implementation requires head_dim_v <= 64 (output accumulator
  // size) which can be improved in the future
  if (head_dim > 128 || head_dim_v > 64) {
    return false;
  }

  // Flash Attention works for all sequence lengths
  // While it's most beneficial for longer sequences, it still works correctly
  // for short sequences used in testing
  return true;
}

/**
 * Check if inputs are suitable for Memory Efficient Attention
 */
inline bool can_use_efficient_attention(
    const Tensor* query,
    const Tensor* key,
    const Tensor* value,
    const Tensor* attn_mask,
    bool is_causal) {
  // Check head dimensions
  const int64_t head_dim_v = value->size(3);

  // Our efficient attention implementation supports head_dim_v <= 128
  if (head_dim_v > 128) {
    return false;
  }

  // Efficient attention supports attention bias/mask
  return true;
}

/**
 * Select the best available backend for SDPA based on input parameters
 *
 * Selection priority:
 * 1. Memory Efficient Attention - For inputs with attention bias/mask
 * 2. Flash Attention - For supported hardware and suitable inputs (no mask)
 * 3. Math fallback - For maximum compatibility
 */
inline SDPBackend select_sdp_backend(
    const Tensor* query,
    const Tensor* key,
    const Tensor* value,
    const Tensor* attn_mask,
    double dropout_p,
    bool is_causal) {
  // Check for unsupported features
  if (dropout_p > 0.0) {
    ET_LOG(Error, "SDPA: Dropout not supported in inference mode");
    return SDPBackend::Error;
  }

  // Check tensor dimensions
  if (query->dim() != 4 || key->dim() != 4 || value->dim() != 4) {
    ET_LOG(Error, "SDPA: All inputs must be 4D tensors");
    return SDPBackend::Error;
  }

  // Check dtype support
  if (!is_supported_dtype(query) || !is_supported_dtype(key) ||
      !is_supported_dtype(value)) {
    ET_LOG(
        Error,
        "SDPA: Unsupported dtype, only Float32/Float16/BFloat16 supported");
    return SDPBackend::Error;
  }

  // Check dtype consistency
  if (query->dtype() != key->dtype() || query->dtype() != value->dtype()) {
    ET_LOG(Error, "SDPA: Query, Key, Value must have the same dtype");
    return SDPBackend::Error;
  }

  // If attention mask/bias is provided, use Memory Efficient Attention
  if (attn_mask != nullptr &&
      can_use_efficient_attention(query, key, value, attn_mask, is_causal)) {
    printf("Selected backend: MemoryEfficientAttention (has attn_mask)\n");
    return SDPBackend::MemoryEfficientAttention;
  }

  // Try Flash Attention if hardware supports it and inputs are suitable
  if (supports_flash_attention() &&
      can_use_flash_attention(query, key, value, attn_mask, is_causal)) {
    printf("Selected backend: FlashAttention\n");
    return SDPBackend::FlashAttention;
  }

  // Fall back to math implementation
  printf("Selected backend: Math (fallback)\n");
  return SDPBackend::Math;
}

// ============================================================================
// Helper Functions for Causal Mask
// ============================================================================

/**
 * Check if we need to apply causal masking
 */
inline bool needs_causal_mask(bool is_causal, const Tensor* attn_mask) {
  if (!is_causal) {
    return false;
  }
  if (attn_mask != nullptr) {
    ET_LOG(
        Error, "SDPA: Cannot use both is_causal=true and explicit attn_mask");
    return false;
  }
  return true;
}

// ============================================================================
// Grouped Query Attention (GQA) Support
// ============================================================================

/**
 * Check if inputs require GQA handling
 *
 * GQA allows num_heads_q != num_heads_kv, where num_heads_q must be
 * divisible by num_heads_kv. Key and Value heads are repeated to match
 * Query heads.
 */
inline bool is_gqa_configuration(
    const Tensor* query,
    const Tensor* key,
    const Tensor* value) {
  const int64_t num_heads_q = query->size(1);
  const int64_t num_heads_kv = key->size(1);

  return num_heads_q != num_heads_kv;
}

/**
 * Validate GQA configuration
 */
inline bool
validate_gqa(const Tensor* query, const Tensor* key, const Tensor* value) {
  const int64_t num_heads_q = query->size(1);
  const int64_t num_heads_kv = key->size(1);
  const int64_t num_heads_v = value->size(1);

  // Key and Value must have same num_heads
  if (num_heads_kv != num_heads_v) {
    ET_LOG(Error, "SDPA GQA: Key and Value must have same num_heads");
    return false;
  }

  // Query heads must be divisible by Key/Value heads
  if (num_heads_q % num_heads_kv != 0) {
    ET_LOG(
        Error,
        "SDPA GQA: Query num_heads must be divisible by Key/Value num_heads");
    return false;
  }

  return true;
}

// ============================================================================
// Main SDPA Entry Point
// ============================================================================

/**
 * Compute scaled dot product attention
 *
 * This is the main entry point that selects the appropriate backend
 * and dispatches to the corresponding implementation.
 *
 * Currently only Math fallback is implemented. Future versions may add:
 * - Flash Attention
 * - Memory Efficient Attention
 * - cuDNN backend
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
    cudaStream_t stream);

} // namespace executorch::backends::cuda
