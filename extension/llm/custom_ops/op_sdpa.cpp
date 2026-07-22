/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/custom_ops/op_sdpa.h>
#include <executorch/extension/llm/custom_ops/op_sdpa_impl.h>

#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <executorch/kernels/optimized/blas/CPUBlas.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
// @lint-ignore CLANGTIDY facebook-unused-include-check
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#ifdef ET_USE_THREADPOOL
#include <executorch/extension/threadpool/threadpool.h>
#include <executorch/runtime/kernel/thread_parallel_interface.h>
#endif
#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>

namespace torch {
namespace executor {

namespace native {

namespace {

bool validate_flash_attention_args(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const optional<Tensor>& attn_mask) {
  ET_CHECK_OR_RETURN_FALSE(query.dim() == 4, "query must be a 4D tensor");
  ET_CHECK_OR_RETURN_FALSE(key.dim() == 4, "key must be a 4D tensor");
  ET_CHECK_OR_RETURN_FALSE(value.dim() == 4, "value must be a 4D tensor");

  // Sizes
  ET_CHECK_OR_RETURN_FALSE(
      (query.size(3) == value.size(3)) && (key.size(3) == value.size(3)),
      "scaled_dot_product_attention_flash_attention: Q/K/V should have the same head size");

  ET_CHECK_OR_RETURN_FALSE(
      (query.scalar_type() == ScalarType::Float) ||
          (query.scalar_type() == ScalarType::Half) ||
          (query.scalar_type() == ScalarType::BFloat16) ||
          (query.scalar_type() == ScalarType::Char),
      "Query must be Float, Half, BFloat16, or Char type");

  ET_CHECK_OR_RETURN_FALSE(
      (query.scalar_type() == key.scalar_type()) &&
          (query.scalar_type() == value.scalar_type()),
      "Key and Value must have the same data type as Query");

  ET_CHECK_OR_RETURN_FALSE(
      !attn_mask.has_value() || attn_mask.value().dim() == 2,
      "Attention mask must be a 2D tensor");

  ET_CHECK_OR_RETURN_FALSE(
      !attn_mask.has_value() ||
          attn_mask.value().scalar_type() == ScalarType::Float,
      "Attention mask must be a Float tensor");

  ET_CHECK_OR_RETURN_FALSE(
      is_contiguous_dim_order(query.dim_order().data(), query.dim()),
      "key cache must be in contiguous dim order");

  ET_CHECK_OR_RETURN_FALSE(
      is_contiguous_dim_order(key.dim_order().data(), key.dim()),
      "value cache must be in contiguous dim order");

  ET_CHECK_OR_RETURN_FALSE(
      is_contiguous_dim_order(value.dim_order().data(), value.dim()),
      "value cache must be in contiguous dim order");

  if (attn_mask.has_value()) {
    ET_CHECK_OR_RETURN_FALSE(
        is_contiguous_dim_order(
            attn_mask.value().dim_order().data(), attn_mask.value().dim()),
        "value cache must be in contiguous dim order");
  }

  return true;
}

bool validate_cache_quant_params_args(
    const Tensor& t,
    const Tensor& t_zero_points,
    const Tensor& t_scales) {
  ET_CHECK_OR_RETURN_FALSE(
      t.dim() == t_scales.dim(),
      "Quantized tensor and scales must have the same number of dimensions");
  ET_CHECK_OR_RETURN_FALSE(
      t.dim() == t_zero_points.dim(),
      "Quantized tensor and scales must have the same number of dimensions");

  ET_CHECK_OR_RETURN_FALSE(
      (t.scalar_type() == ScalarType::Char), "Tensor must be of int8_t type");

  ET_CHECK_OR_RETURN_FALSE(
      (t_scales.scalar_type() == ScalarType::Float),
      "Scales tensor must be of float type");

  ET_CHECK_OR_RETURN_FALSE(
      (t_zero_points.scalar_type() == ScalarType::Char),
      "Zero points tensor must be of int8_t type");

  // Sizes
  for (int64_t i = 0; i < t.dim() - 1; i++) {
    ET_CHECK_OR_RETURN_FALSE(
        (t.size(i) == t_scales.size(i)),
        "Quantized tensor and scales have different shape"
        "at dim: %" PRId64 ", t: %zd, t_scales: %zd",
        i,
        t.size(i),
        t_scales.size(i));
    ;
    ET_CHECK_OR_RETURN_FALSE(
        (t.size(i) == t_zero_points.size(i)),
        "Quantized tensor and zero points have different shape"
        "at dim: %" PRId64 ", t: %zd, t_scales: %zd",
        i,
        t.size(i),
        t_zero_points.size(i));
    ;
  }

  return true;
}

bool validate_cache_params(
    const Tensor& k_cache,
    const Tensor& v_cache,
    int64_t start_pos,
    int64_t seq_length) {
  ET_CHECK_OR_RETURN_FALSE(k_cache.dim() == 4, "kcache must be a 4D tensor");

  ET_CHECK_OR_RETURN_FALSE(v_cache.dim() == 4, "v_cache must be a 4D tensor");

  ET_CHECK_OR_RETURN_FALSE(
      start_pos < k_cache.size(1),
      "start_pos must be less than key cache at dim 1");

  ET_CHECK_OR_RETURN_FALSE(
      start_pos < v_cache.size(1),
      "start_pos must be less than value cache at dim 1");

  ET_CHECK_OR_RETURN_FALSE(
      (start_pos + seq_length) <= k_cache.size(1),
      "start_post + seq_length must be less than max seq length supported by key cache."
      "start pos: %" PRId64 ", seq_length: %" PRId64
      "."
      "key cache size: %zd",
      start_pos,
      seq_length,
      k_cache.size(1));

  ET_CHECK_OR_RETURN_FALSE(
      (start_pos + seq_length) <= v_cache.size(1),
      "start_post + seq_length must be less than max seq length supported by key cache."
      "start pos: %" PRId64 ", seq_length: %" PRId64
      "."
      "value cache size: %zd",
      start_pos,
      seq_length,
      v_cache.size(1));

  // Make sure they are in contiguous dim order
  ET_CHECK_OR_RETURN_FALSE(
      is_contiguous_dim_order(k_cache.dim_order().data(), k_cache.dim()),
      "key cache must be in contiguous dim order");

  ET_CHECK_OR_RETURN_FALSE(
      is_contiguous_dim_order(v_cache.dim_order().data(), v_cache.dim()),
      "value cache must be in contiguous dim order");

  return true;
}

bool validate_channelwise_gated_delta_rule_args(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& decay,
    const Tensor& beta,
    const Tensor& initial_state) {
  ET_CHECK_OR_RETURN_FALSE(query.dim() == 4, "query must be a 4D tensor");
  ET_CHECK_OR_RETURN_FALSE(key.dim() == 4, "key must be a 4D tensor");
  ET_CHECK_OR_RETURN_FALSE(value.dim() == 4, "value must be a 4D tensor");
  ET_CHECK_OR_RETURN_FALSE(decay.dim() == 4, "decay must be a 4D tensor");
  ET_CHECK_OR_RETURN_FALSE(beta.dim() == 3, "beta must be a 3D tensor");
  ET_CHECK_OR_RETURN_FALSE(
      initial_state.dim() == 4, "initial_state must be a 4D tensor");

  ET_CHECK_OR_RETURN_FALSE(
      query.scalar_type() == ScalarType::Float, "query must be float32");
  ET_CHECK_OR_RETURN_FALSE(
      key.scalar_type() == ScalarType::Float, "key must be float32");
  ET_CHECK_OR_RETURN_FALSE(
      value.scalar_type() == ScalarType::Float, "value must be float32");
  ET_CHECK_OR_RETURN_FALSE(
      decay.scalar_type() == ScalarType::Float, "decay must be float32");
  ET_CHECK_OR_RETURN_FALSE(
      beta.scalar_type() == ScalarType::Float, "beta must be float32");
  ET_CHECK_OR_RETURN_FALSE(
      initial_state.scalar_type() == ScalarType::Float,
      "initial_state must be float32");

  ET_CHECK_OR_RETURN_FALSE(
      query.size(0) == key.size(0) && query.size(1) == key.size(1) &&
          query.size(2) == key.size(2) && query.size(3) == key.size(3),
      "query and key must have matching shapes");
  ET_CHECK_OR_RETURN_FALSE(
      query.size(0) == decay.size(0) && query.size(1) == decay.size(1) &&
          query.size(2) == decay.size(2) && query.size(3) == decay.size(3),
      "query and decay must have matching shapes");
  ET_CHECK_OR_RETURN_FALSE(
      query.size(0) == value.size(0) && query.size(1) == value.size(1) &&
          query.size(2) == value.size(2),
      "query and value must match in batch/head/sequence dims");
  ET_CHECK_OR_RETURN_FALSE(
      beta.size(0) == query.size(0) && beta.size(1) == query.size(1) &&
          beta.size(2) == query.size(2),
      "beta must match query batch/head/sequence dims");
  ET_CHECK_OR_RETURN_FALSE(
      initial_state.size(0) == query.size(0) &&
          initial_state.size(1) == query.size(1) &&
          initial_state.size(2) == query.size(3) &&
          initial_state.size(3) == value.size(3),
      "initial_state shape must match [B, H, K, V]");

  for (const Tensor* tensor :
       {&query, &key, &value, &decay, &beta, &initial_state}) {
    ET_CHECK_OR_RETURN_FALSE(
        is_contiguous_dim_order((*tensor).dim_order().data(), (*tensor).dim()),
        "channelwise gated delta rule expects contiguous inputs");
  }

  return true;
}

bool tensor_memory_ranges_overlap(const Tensor& a, const Tensor& b) {
  if (a.nbytes() == 0 || b.nbytes() == 0) {
    return false;
  }
  const auto a_begin = reinterpret_cast<uintptr_t>(a.const_data_ptr());
  const auto b_begin = reinterpret_cast<uintptr_t>(b.const_data_ptr());
  const auto a_end = a_begin + a.nbytes();
  const auto b_end = b_begin + b.nbytes();
  return a_begin < b_end && b_begin < a_end;
}

// TODO: seq_length is not yet used for copy
void update_cache(
    const Tensor& projected_value,
    const Tensor& cache,
    int64_t start_pos,
    int64_t seq_length) { // NOLINT: unused parameter 'seq_length'
  // 1) Cache shape should be [bs, max_seq_len, num heads, head dim]
  // 2) projected_value shape should be [bs, seq_len, num heads, head dim]
  // 3) We're updating the cache with projected_value, at position start_pos

  ET_CHECK_MSG(
      projected_value.size(0) == cache.size(0),
      "projected_value batch size should be equal to the cache batch size.");
  ET_CHECK_MSG(
      projected_value.size(2) == cache.size(2),
      "projected_value number of heads should be equal to the cache number of heads.");
  ET_CHECK_MSG(
      projected_value.size(3) == cache.size(3),
      "projected_value embedding dimension should be equal to the cache embedding dimension.");
  ET_CHECK_MSG(
      projected_value.element_size() == cache.element_size(),
      "projected_value data type size should be equal to the cache data type size.");

  ET_CHECK_MSG(
      is_contiguous_dim_order(
          projected_value.dim_order().data(), projected_value.dim()),
      "projected value must be in contiguous dim order");
  const void* projected_value_data = projected_value.const_data_ptr();
  void* cache_data = cache.mutable_data_ptr();

  ET_CHECK_MSG(projected_value_data != nullptr, "projected_value data is null");
  ET_CHECK_MSG(cache_data, "cache data is null");

  auto cache_strides = cache.strides();
  ::executorch::aten::StridesType cache_batch_dim_stride = cache_strides[0];
  ::executorch::aten::StridesType cache_seq_dim_stride = cache_strides[1];

  auto value_strides = projected_value.strides();
  ::executorch::aten::StridesType value_batch_dim_stride = value_strides[0];

  ::executorch::aten::SizesType num_bytes_to_copy =
      (projected_value.numel() / projected_value.size(0)) *
      projected_value.element_size();

  for (int64_t batch_line = 0; batch_line < projected_value.size(0);
       ++batch_line) {
    ::executorch::aten::SizesType cache_pos_offset =
        (batch_line * cache_batch_dim_stride +
         start_pos * cache_seq_dim_stride) *
        cache.element_size();
    ::executorch::aten::SizesType value_pos_offset =
        (batch_line * value_batch_dim_stride) * cache.element_size();

    std::memcpy(
        (uint8_t*)cache_data + cache_pos_offset,
        (uint8_t*)projected_value_data + value_pos_offset,
        num_bytes_to_copy);
  }
}

} // anonymous namespace

Tensor& flash_attention_kernel_out(
    RuntimeContext& ctx,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const optional<Tensor>& attn_mask,
    const double dropout_p,
    const bool is_causal,
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const optional<double> scale,
    Tensor& output) {
  (void)ctx;
  ET_KERNEL_CHECK(
      ctx,
      validate_flash_attention_args(query, key, value, attn_mask),
      InvalidArgument,
      output);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(output, query.sizes()) == Error::Ok,
      InvalidArgument,
      output);

  auto seq_len = query.size(2);

  ET_SWITCH_FLOATHBF16_TYPES(
      query.scalar_type(), ctx, "flash_attention", CTYPE, [&] {
        // TODO we need to re-evaluate this for ARM CPUs
        // And there can be many so instead of templatizing
        // we might consider another appraoch
        if (seq_len >= 768) {
          sdpa::impl::cpu_flash_attention<CTYPE, 256, 512>(
              ctx,
              output,
              query,
              key,
              value,
              dropout_p,
              is_causal,
              attn_mask,
              scale,
              nullopt,
              nullopt,
              nullopt,
              nullopt,
              nullopt,
              nullopt);
        } else if (seq_len >= 192) {
          sdpa::impl::cpu_flash_attention<CTYPE, 64, 512>(
              ctx,
              output,
              query,
              key,
              value,
              dropout_p,
              is_causal,
              attn_mask,
              scale,
              nullopt,
              nullopt,
              nullopt,
              nullopt,
              nullopt,
              nullopt);
        } else {
          sdpa::impl::cpu_flash_attention<CTYPE, 32, 512>(
              ctx,
              output,
              query,
              key,
              value,
              dropout_p,
              is_causal,
              attn_mask,
              scale,
              nullopt,
              nullopt,
              nullopt,
              nullopt,
              nullopt,
              nullopt);
        }
      });
  return output;
}

Tensor& custom_sdpa_out_impl(
    RuntimeContext& ctx,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const int64_t start_pos,
    const optional<Tensor>& attn_mask,
    const double dropout_p,
    const bool is_causal,
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const optional<double> scale,
    Tensor& output,
    const optional<Tensor>& q_zero_points = nullopt,
    const optional<Tensor>& q_scales = nullopt,
    const optional<Tensor>& k_zero_points = nullopt,
    const optional<Tensor>& k_scales = nullopt,
    const optional<Tensor>& v_zero_points = nullopt,
    const optional<Tensor>& v_scales = nullopt,
    bool is_seq_at_dim_2 = false) {
  ET_KERNEL_CHECK_MSG(
      ctx,
      !attn_mask.has_value() || !is_causal,
      InvalidArgument,
      output,
      "attn_mask and is_causal cannot be set at the same time");

  ET_KERNEL_CHECK_MSG(
      ctx,
      validate_flash_attention_args(q, k, v, attn_mask),
      InvalidArgument,
      output,
      "Invalid arguments");

  int64_t seq_len = q.size(1);
  SeqDim seq_dim{SeqDim::TWO};
  if (!is_seq_at_dim_2) {
    seq_dim = SeqDim::ONE;
  }

  if (q.scalar_type() == ScalarType::Char) {
    if (seq_dim == SeqDim::TWO) {
      seq_len = q.size(2);
    }
    ET_KERNEL_CHECK_MSG(
        ctx,
        q_scales.has_value() && q_zero_points.has_value() &&
            k_scales.has_value() && k_zero_points.has_value() &&
            q_scales.has_value() && q_zero_points.has_value(),
        InvalidArgument,
        output,
        "If q is quantized, k and v must be quantized as well");
    ET_KERNEL_CHECK_MSG(
        ctx,
        validate_cache_quant_params_args(
            q, q_zero_points.value(), q_scales.value()),
        InvalidArgument,
        output,
        "Invalid arguments for quantized query");
    ET_KERNEL_CHECK_MSG(
        ctx,
        validate_cache_quant_params_args(
            k, k_zero_points.value(), k_scales.value()),
        InvalidArgument,
        output,
        "Invalid arguments for quantized key");
    ET_KERNEL_CHECK_MSG(
        ctx,
        validate_cache_quant_params_args(
            v, v_zero_points.value(), v_scales.value()),
        InvalidArgument,
        output,
        "Invalid arguments for quantized value");
  }

  ET_CHECK_MSG(q.dim() == 4, "query must be a 4D tensor");

  const int64_t num_keys_for_causal_attention =
      attn_mask.has_value() ? -1 : start_pos + seq_len;

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(output, q.sizes()) == Error::Ok,
      InvalidArgument,
      output);

  // TODO(task): replace the template param selection logic
  // with whatever apprpriately makes more sense for
  ET_SWITCH_FLOATHBF16_TYPES(
      output.scalar_type(), ctx, "flash_attention", CTYPE, [&] {
        // TODO we need to re-evaluate this for ARM CPUs
        // And there can be many so instead of templatizing
        // we might consider another appraoch
        if (seq_len >= 768) {
          sdpa::impl::cpu_flash_attention<CTYPE, 256, 512>(
              ctx,
              output,
              q,
              k,
              v,
              dropout_p,
              is_causal,
              attn_mask,
              scale,
              q_zero_points, // q_zero_points
              q_scales, // q_scales
              k_zero_points, // k_zero_points
              k_scales, // k_scales
              v_zero_points, // v_zero_points
              v_scales, // v_scales
              seq_dim, /* seq_dim */
              start_pos,
              num_keys_for_causal_attention);
        } else if (seq_len >= 192) {
          sdpa::impl::cpu_flash_attention<CTYPE, 64, 512>(
              ctx,
              output,
              q,
              k,
              v,
              dropout_p,
              is_causal,
              attn_mask,
              scale,
              q_zero_points, // q_zero_points
              q_scales, // q_scales
              k_zero_points, // k_zero_points
              k_scales, // k_scales
              v_zero_points, // v_zero_points
              v_scales, // v_scales
              seq_dim, /* seq_dim */
              start_pos,
              num_keys_for_causal_attention);
        } else {
          sdpa::impl::cpu_flash_attention<CTYPE, 32, 512>(
              ctx,
              output,
              q,
              k,
              v,
              dropout_p,
              is_causal,
              attn_mask,
              scale,
              q_zero_points, // q_zero_points
              q_scales, // q_scales
              k_zero_points, // k_zero_points
              k_scales, // k_scales
              v_zero_points, // v_zero_points
              v_scales, // v_scales
              seq_dim, /* seq_dim */
              start_pos,
              num_keys_for_causal_attention);
        }
      });
  return output;
}

Tensor& custom_quantized_sdpa_out(
    RuntimeContext& ctx,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const int64_t start_pos,
    const optional<Tensor>& attn_mask,
    const double dropout_p,
    const bool is_causal,
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const optional<double> scale,
    const optional<Tensor>& q_zero_points,
    const optional<Tensor>& q_scales,
    const optional<Tensor>& k_zero_points,
    const optional<Tensor>& k_scales,
    const optional<Tensor>& v_zero_points,
    const optional<Tensor>& v_scales,
    const bool is_seq_at_dim_2,
    Tensor& output) {
  return custom_sdpa_out_impl(
      ctx,
      q,
      k,
      v,
      start_pos,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      output,
      q_zero_points,
      q_scales,
      k_zero_points,
      k_scales,
      v_zero_points,
      v_scales,
      is_seq_at_dim_2);
}

/*
  Input params
  @param[in] q_projected Projected query with query weights.
  Format [n_layers, batch size, seq_len, num heads, head dim]
  @param[in] k_projected Projected query with key weights.
  Format [n_layers, batch size, seq_len, num heads, head dim]
  @param[in] v_projected Projected query with value weights.
  Format [n_layers, batch size, seq_len, num heads, head dim]
  @param[in] key_cache Cache of previous k_projected.
  Format [n_layers, batch size, max_seq_len, num heads, head dim]
  @param[in] key_cache Cache of previous v_projected.
  Format [n_layers, batch size, max_seq_len, num heads, head dim]
  ....
  @param[in] start_pos: sequence position
*/
Tensor& custom_sdpa_out(
    RuntimeContext& ctx,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const int64_t start_pos,
    const optional<Tensor>& attn_mask,
    const double dropout_p,
    const bool is_causal,
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const optional<double> scale,
    Tensor& output) {
  return custom_sdpa_out_impl(
      ctx, q, k, v, start_pos, attn_mask, dropout_p, is_causal, scale, output);
}
/*
  Input params
  @param[in] q_projected Projected query with query weights.
  Format [n_layers, batch size, seq_len, num heads, head dim]
  @param[in] k_projected Projected query with key weights.
  Format [n_layers, batch size, seq_len, num heads, head dim]
  @param[in] v_projected Projected query with value weights.
  Format [n_layers, batch size, seq_len, num heads, head dim]
  @param[in] key_cache Cache of previous k_projected.
  Format [n_layers, batch size, max_seq_len, num heads, head dim]
  @param[in] key_cache Cache of previous v_projected.
  Format [n_layers, batch size, max_seq_len, num heads, head dim]
  ....
  @param[in] start_pos: sequence position
  @param[in] seq_len: Seq length. e.g. seq_len dim of q_projected.
*/
Tensor& sdpa_with_kv_cache_out(
    KernelRuntimeContext& ctx,
    const Tensor& q_projected,
    const Tensor& k_projected,
    const Tensor& v_projected,
    Tensor& key_cache,
    Tensor& value_cache,
    const int64_t start_pos,
    const int64_t seq_len,
    const optional<Tensor>& attn_mask,
    const double dropout_p,
    const bool is_causal,
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const optional<double> scale,
    Tensor& output) {
  (void)ctx;
  ET_KERNEL_CHECK(
      ctx,
      validate_cache_params(key_cache, value_cache, start_pos, seq_len),
      InvalidArgument,
      output);

  ET_CHECK_MSG(q_projected.dim() == 4, "query must be a 4D tensor");

  update_cache(k_projected, key_cache, start_pos, seq_len);
  update_cache(v_projected, value_cache, start_pos, seq_len);

  custom_sdpa_out(
      ctx,
      q_projected,
      key_cache,
      value_cache,
      start_pos,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      output);

  return output;
}

std::tuple<Tensor&, Tensor&> channelwise_gated_delta_rule_out(
    RuntimeContext& ctx,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& decay,
    const Tensor& beta,
    const Tensor& initial_state,
    Tensor& out,
    Tensor& final_state_out) {
  std::tuple<Tensor&, Tensor&> ret(out, final_state_out);
  ET_KERNEL_CHECK(
      ctx,
      validate_channelwise_gated_delta_rule_args(
          query, key, value, decay, beta, initial_state),
      InvalidArgument,
      ret);
  ET_KERNEL_CHECK_MSG(
      ctx,
      !tensor_memory_ranges_overlap(initial_state, final_state_out),
      InvalidArgument,
      ret,
      "channelwise_gated_delta_rule final_state_out must not alias initial_state.");
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, value.sizes()) == Error::Ok,
      InvalidArgument,
      ret,
      "Failed to resize channelwise_gated_delta_rule output tensor.");
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(final_state_out, initial_state.sizes()) == Error::Ok,
      InvalidArgument,
      ret,
      "Failed to resize channelwise_gated_delta_rule final_state tensor.");
  ET_KERNEL_CHECK(
      ctx, out.scalar_type() == ScalarType::Float, InvalidArgument, ret);
  ET_KERNEL_CHECK(
      ctx,
      final_state_out.scalar_type() == ScalarType::Float,
      InvalidArgument,
      ret);
  ET_KERNEL_CHECK(
      ctx,
      is_contiguous_dim_order(out.dim_order().data(), out.dim()),
      InvalidArgument,
      ret);
  ET_KERNEL_CHECK(
      ctx,
      is_contiguous_dim_order(
          final_state_out.dim_order().data(), final_state_out.dim()),
      InvalidArgument,
      ret);

  const auto batch_size = query.size(0);
  const auto num_heads = query.size(1);
  const auto sequence_length = query.size(2);
  const auto k_head_dim = query.size(3);
  const auto v_head_dim = value.size(3);

  const auto qk_batch_stride = num_heads * sequence_length * k_head_dim;
  const auto qk_head_stride = sequence_length * k_head_dim;
  const auto qk_seq_stride = k_head_dim;

  const auto value_batch_stride = num_heads * sequence_length * v_head_dim;
  const auto value_head_stride = sequence_length * v_head_dim;
  const auto value_seq_stride = v_head_dim;

  const auto beta_batch_stride = num_heads * sequence_length;
  const auto beta_head_stride = sequence_length;

  const auto state_batch_stride = num_heads * k_head_dim * v_head_dim;
  const auto state_head_stride = k_head_dim * v_head_dim;

  const auto* query_data = query.const_data_ptr<float>();
  const auto* key_data = key.const_data_ptr<float>();
  const auto* value_data = value.const_data_ptr<float>();
  const auto* decay_data = decay.const_data_ptr<float>();
  const auto* beta_data = beta.const_data_ptr<float>();
  const auto* initial_state_data = initial_state.const_data_ptr<float>();
  auto* state_data = final_state_out.mutable_data_ptr<float>();
  auto* output_data = out.mutable_data_ptr<float>();
  std::vector<float> v_pred(v_head_dim);
  std::vector<float> delta(v_head_dim);

  for (int64_t batch = 0; batch < batch_size; ++batch) {
    for (int64_t head = 0; head < num_heads; ++head) {
      const auto qk_offset = batch * qk_batch_stride + head * qk_head_stride;
      const auto value_offset =
          batch * value_batch_stride + head * value_head_stride;
      const auto beta_offset =
          batch * beta_batch_stride + head * beta_head_stride;
      const auto state_offset =
          batch * state_batch_stride + head * state_head_stride;

      const auto* q_head = query_data + qk_offset;
      const auto* k_head = key_data + qk_offset;
      const auto* decay_head = decay_data + qk_offset;
      const auto* value_head = value_data + value_offset;
      const auto* beta_head = beta_data + beta_offset;
      const auto* initial_state_head = initial_state_data + state_offset;
      auto* state_head = state_data + state_offset;
      auto* output_head = output_data + value_offset;

      // Functional: seed the running state from initial_state without mutating
      // the (read-only) input.
      for (int64_t idx = 0; idx < state_head_stride; ++idx) {
        state_head[idx] = initial_state_head[idx];
      }

      for (int64_t token = 0; token < sequence_length; ++token) {
        const auto* q_t = q_head + token * qk_seq_stride;
        const auto* k_t = k_head + token * qk_seq_stride;
        const auto* decay_t = decay_head + token * qk_seq_stride;
        const auto* v_t = value_head + token * value_seq_stride;
        const float beta_t = beta_head[token];
        auto* output_t = output_head + token * value_seq_stride;

        // 1. Per-key-channel decay: scale each state row by decay_t[k].
        for (int64_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
          const float decay_value = decay_t[k_idx];
          auto* state_row = state_head + k_idx * v_head_dim;
          for (int64_t v_idx = 0; v_idx < v_head_dim; ++v_idx) {
            state_row[v_idx] *= decay_value;
          }
        }

        // 2. v_pred = S^T k: the memory's predicted value for this key, read
        // off the decayed state.
        std::fill(v_pred.begin(), v_pred.end(), 0.0f);
        for (int64_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
          const float key_value = k_t[k_idx];
          const auto* state_row = state_head + k_idx * v_head_dim;
          for (int64_t v_idx = 0; v_idx < v_head_dim; ++v_idx) {
            v_pred[v_idx] += state_row[v_idx] * key_value;
          }
        }

        // 3. delta = beta * (v - v_pred).
        for (int64_t v_idx = 0; v_idx < v_head_dim; ++v_idx) {
          delta[v_idx] = (v_t[v_idx] - v_pred[v_idx]) * beta_t;
        }

        // 4. Rank-1 write: S += k (x) delta.
        for (int64_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
          const float key_value = k_t[k_idx];
          auto* state_row = state_head + k_idx * v_head_dim;
          for (int64_t v_idx = 0; v_idx < v_head_dim; ++v_idx) {
            state_row[v_idx] += key_value * delta[v_idx];
          }
        }

        // 5. Output readout: o_t = S^T q.
        std::fill(output_t, output_t + v_head_dim, 0.0f);
        for (int64_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
          const float query_value = q_t[k_idx];
          const auto* state_row = state_head + k_idx * v_head_dim;
          for (int64_t v_idx = 0; v_idx < v_head_dim; ++v_idx) {
            output_t[v_idx] += state_row[v_idx] * query_value;
          }
        }
      }
    }
  }

  return ret;
}
} // namespace native
} // namespace executor
} // namespace torch

EXECUTORCH_LIBRARY(
    llama,
    "sdpa_with_kv_cache.out",
    torch::executor::native::sdpa_with_kv_cache_out);

EXECUTORCH_LIBRARY(
    llama,
    "custom_sdpa.out",
    torch::executor::native::custom_sdpa_out);

EXECUTORCH_LIBRARY(
    llama,
    "custom_quantized_sdpa.out",
    torch::executor::native::custom_quantized_sdpa_out);

namespace {

void channelwise_gated_delta_rule_out_boxed(
    executorch::runtime::KernelRuntimeContext& ctx,
    executorch::runtime::Span<executorch::runtime::EValue*> stack) {
  // Multi-output out variants get a trailing TensorList aggregating the two
  // outputs appended by the emitter, so the boxed stack has 9 entries: 6 inputs
  // + out + final_state_out + [out, final_state_out]. The aggregate (stack[8])
  // duplicates stack[6]/stack[7] and is ignored.
  ET_KERNEL_CHECK_MSG(
      ctx,
      stack.size() == 9,
      InvalidProgram,
      /* void */,
      "Expected %zu args, got %zu",
      static_cast<size_t>(9),
      stack.size());

  auto& query = stack[0]->toTensor();
  auto& key = stack[1]->toTensor();
  auto& value = stack[2]->toTensor();
  auto& decay = stack[3]->toTensor();
  auto& beta = stack[4]->toTensor();
  auto& initial_state = stack[5]->toTensor();
  auto& out = stack[6]->toTensor();
  auto& final_state_out = stack[7]->toTensor();

  (void)torch::executor::native::channelwise_gated_delta_rule_out(
      ctx, query, key, value, decay, beta, initial_state, out, final_state_out);
}

const auto channelwise_gated_delta_rule_out_registration =
    executorch::runtime::register_kernel(executorch::runtime::Kernel(
        "llama::channelwise_gated_delta_rule.out",
        channelwise_gated_delta_rule_out_boxed));

} // namespace
