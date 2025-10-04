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

Tensor& sdpa_with_kv_cache_out(
    KernelRuntimeContext& ctx,
    const Tensor& q_projected,
    const Tensor& k_projected,
    const Tensor& v_projected,
    Tensor& key_cache,
    Tensor& value_cache,
    const int64_t start_pos,
    const int64_t seq_len,
    const std::optional<Tensor>& attn_mask,
    const double dropout_p,
    const bool is_causal,
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const std::optional<double> scale,
    Tensor& output);

Tensor& custom_sdpa_out(
    RuntimeContext& ctx,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const int64_t start_pos,
    const std::optional<Tensor>& attn_mask,
    const double dropout_p,
    const bool is_causal,
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const std::optional<double> scale,
    Tensor& output);

Tensor& flash_attention_kernel_out(
    KernelRuntimeContext& ctx,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask,
    const double dropout_p,
    const bool is_causal,
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const std::optional<double> scale,
    Tensor& output);

Tensor& custom_quantized_sdpa_out(
    RuntimeContext& ctx,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const int64_t start_pos,
    const std::optional<Tensor>& attn_mask,
    const double dropout_p,
    const bool is_causal,
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const std::optional<double> scale,
    const std::optional<Tensor>& q_zero_points,
    const std::optional<Tensor>& q_scales,
    const std::optional<Tensor>& k_zero_points,
    const std::optional<Tensor>& k_scales,
    const std::optional<Tensor>& v_zero_points,
    const std::optional<Tensor>& v_scales,
    const bool is_seq_at_dim_1,
    Tensor& output);
} // namespace native
} // namespace executor
} // namespace torch
