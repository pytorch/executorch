/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

// llama::quantized_moe_ffn.out
//
// Single fused MoE feed-forward op for token-choice routing with
// torchao-packed group-quantized INT4 (or up to INT8) experts and INT8
// dynamic-quantized activations.
//
// Implements the same math as
// executorch.examples.models.llama.llama_transformer.MOEFeedForward.forward:
//   - router GEMM on `gate_weight`,
//   - "sigmoid" or "softmax" scoring (with optional `expert_bias` shifting
//     top-k selection but not the gathered routing weights),
//   - top-k expert selection,
//   - permute-then-grouped-GEMM expert evaluation
//     (3 GEMMs per active expert via
//     torchao::ops::linear_8bit_act_xbit_weight),
//   - SwiGLU activation,
//   - weighted scatter-add unpermute.
//
// On aarch64, the torchao kernel selector picks the best available
// ukernel (NEON i8mm, dotprod, or scalar) based on runtime CPU
// feature detection.  On x86, a reference path unpacks the torchao
// blob, dequantizes, and calls cpublas::gemm.
//
// Inputs:
//   x [T, D] fp32 — flattened activations
//   gate_weight [E, D] fp32 — router weight (block_sparse_moe.gate.weight)
//   expert_bias [E] fp32 — additive bias for top-k selection. May be empty
//     (numel == 0) if `moe_gate_bias=False`.
//   packed_w1, packed_w3 [E, packed_bytes] uint8 — torchao opaque packed
//     weight blobs for the up/gate projections (D -> F per expert).
//   packed_w2 [E, packed_bytes] uint8 — torchao packed blob for the down
//     projection (F -> D per expert).
//   num_activated_experts — top-k.
//   num_experts E, hidden_dim F, dim D — shapes.
//   group_size — INT4 group size (e.g. 32).
//   weight_nbit — quantized weight bit width. INT4 (weight_nbit==4) and
//     INT8 (weight_nbit==8) are supported; other values trip a runtime check.
//   score_func — "sigmoid" or "softmax".
//   route_scale — multiplier applied to the gathered (unbiased) routing
//     weights for the sigmoid path. Ignored for softmax.
//
// Output:
//   out [T, D] fp32.
Tensor& quantized_moe_ffn_out(
    KernelRuntimeContext& ctx,
    const Tensor& x,
    const Tensor& gate_weight,
    const Tensor& expert_bias,
    const Tensor& packed_w1,
    const Tensor& packed_w3,
    const Tensor& packed_w2,
    int64_t num_activated_experts,
    int64_t num_experts,
    int64_t hidden_dim,
    int64_t dim,
    int64_t group_size,
    int64_t weight_nbit,
    executorch::aten::string_view score_func,
    double route_scale,
    Tensor& out);

} // namespace native
} // namespace executor
} // namespace torch
