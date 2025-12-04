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
 * This matches PyTorch's AOTI signature for:
 * torch.ops.aten._scaled_dot_product_flash_attention
 *
 * @param query Query tensor [batch, num_heads, seq_len_q, head_dim]
 * @param key Key tensor [batch, num_heads, seq_len_k, head_dim]
 * @param value Value tensor [batch, num_heads, seq_len_k, head_dim]
 * @param dropout_p Dropout probability (must be 0.0 for inference)
 * @param is_causal Whether to apply causal masking
 * @param return_debug_mask Whether to return debug attention mask (ignored for
 * inference)
 * @param scale Optional scaling factor for attention scores
 * @param ret0 Output: attention result [batch, num_heads, seq_len_q, head_dim]
 * @param ret1 Output: logsumexp (set to nullptr for inference)
 * @param ret2 Output: cumulative sequence length Q (set to nullptr for
 * inference)
 * @param ret3 Output: cumulative sequence length K (set to nullptr for
 * inference)
 * @param max_seqlen_q Maximum sequence length in Q (set to seq_len_q)
 * @param max_seqlen_k Maximum sequence length in K (set to seq_len_k)
 * @param ret4 Output: philox seed (set to nullptr for inference)
 * @param ret5 Output: philox offset (set to nullptr for inference)
 * @param ret6 Output: debug attention mask (set to nullptr for inference)
 *
 * @return AOTITorchError error code
 */
AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_cuda__scaled_dot_product_flash_attention(
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
    Tensor** ret6); // debug_attn_mask (nullptr for inference)

/**
 * Performs scaled dot-product efficient attention on CUDA.
 *
 * This matches PyTorch's AOTI signature for:
 * torch.ops.aten._scaled_dot_product_efficient_attention
 *
 * @param query Query tensor [batch, num_heads, seq_len_q, head_dim]
 * @param key Key tensor [batch, num_heads, seq_len_k, head_dim]
 * @param value Value tensor [batch, num_heads, seq_len_k, head_dim]
 * @param attn_bias Optional attention bias (additive mask)
 * @param compute_log_sumexp Whether to compute logsumexp (ignored for
 * inference)
 * @param dropout_p Dropout probability (must be 0.0 for inference)
 * @param is_causal Whether to apply causal masking
 * @param scale Optional scaling factor for attention scores
 * @param ret0 Output: attention result [batch, num_heads, seq_len_q, head_dim]
 * @param ret1 Output: logsumexp (set to nullptr for inference)
 * @param ret2 Output: philox seed (set to nullptr for inference)
 * @param ret3 Output: philox offset (set to nullptr for inference)
 *
 * @return AOTITorchError error code
 *
 */
AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_cuda__scaled_dot_product_efficient_attention(
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
    Tensor** ret3); // philox_offset (nullptr for inference)

#ifdef __cplusplus
}
#endif

} // namespace executorch::backends::cuda
