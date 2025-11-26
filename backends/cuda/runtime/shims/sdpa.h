/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda_runtime.h>
#include <executorch/backends/aoti/common_shims.h>
#include <executorch/backends/aoti/export.h>

namespace executorch::backends::cuda {

using executorch::backends::aoti::AOTITorchError;
using executorch::backends::aoti::Tensor;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Performs scaled dot-product attention on CUDA.
 *
 * This is a port of PyTorch's scaled_dot_product_attention CUDA implementation
 * (aten/src/ATen/native/transformers/cuda/attention.cu) adapted for the
 * ExecuTorch runtime.
 *
 * Computes attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
 *
 * HARDWARE REQUIREMENTS:
 * - CUDA-capable GPU
 * - Supports Flash Attention if available (Ampere+ GPUs)
 *
 * TENSOR REQUIREMENTS:
 * @param query Query tensor [batch, num_heads, seq_len_q, head_dim]
 *   - Must be Float32, Float16, or BFloat16 dtype
 *   - Must be 4D
 *   - Must be on CUDA device
 *
 * @param key Key tensor [batch, num_heads_kv, seq_len_k, head_dim]
 *   - Must be same dtype as query
 *   - Must be 4D
 *   - Must be on CUDA device
 *   - num_heads_kv can be different from num_heads (for GQA)
 *
 * @param value Value tensor [batch, num_heads_kv, seq_len_k, head_dim_v]
 *   - Must be same dtype as query
 *   - Must be 4D
 *   - Must be on CUDA device
 *
 * @param attn_mask Optional attention mask [batch, num_heads, seq_len_q, seq_len_k]
 *   or broadcastable shape
 *   - Can be nullptr (no mask)
 *   - If provided, must be Float32, BFloat16, or Bool dtype
 *   - Additive mask: positions with large negative values are masked out
 *
 * @param dropout_p Dropout probability (0.0 to 1.0)
 *   - Currently only supports 0.0 (no dropout)
 *   - Must be 0.0 for inference
 *
 * @param is_causal Whether to apply causal masking
 *   - If true, applies lower triangular mask
 *   - Cannot be used together with explicit attn_mask
 *
 * @param scale Optional scaling factor for attention scores
 *   - If nullptr, uses 1/sqrt(head_dim) by default
 *   - If provided, uses the specified value
 *
 * @param enable_gqa Enable grouped query attention support
 *   - Allows num_heads_kv != num_heads
 *   - Query heads must be divisible by key/value heads
 *
 * @param ret0 Output parameter for attention result
 *   [batch, num_heads, seq_len_q, head_dim_v]
 *   - Allocated by this function
 *   - Same dtype as input tensors
 *   - Must not be null
 *   - Caller is responsible for freeing via aoti_torch_delete_tensor_object()
 *
 * @return AOTITorchError error code:
 *   - Error::Ok: Success
 *   - Error::InvalidArgument: Null pointer, wrong dtype, wrong dimensions,
 *     or invalid parameter combination
 *   - Error::Internal: CUDA kernel launch failure
 */
AOTI_SHIM_EXPORT AOTITorchError aoti_torch_cuda_scaled_dot_product_attention(
    Tensor* query,
    Tensor* key,
    Tensor* value,
    Tensor* attn_mask,
    double dropout_p,
    int32_t is_causal,
    double* scale,
    int32_t enable_gqa,
    Tensor** ret0);

#ifdef __cplusplus
}
#endif

} // namespace executorch::backends::cuda
