/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/custom_ops/op_sdpa.h>
#include <executorch/extension/llm/custom_ops/op_sdpa_impl.h>

#include <executorch/kernels/optimized/blas/CPUBlas.h>
#include <executorch/kernels/optimized/vec/functional.h>
#include <executorch/kernels/optimized/vec/vec.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
// @lint-ignore CLANGTIDY facebook-unused-include-check
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

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
      (query.scalar_type() == ScalarType::Float), "Query must be Float type");

  ET_CHECK_OR_RETURN_FALSE(
      (query.scalar_type() == key.scalar_type()) &&
          (query.scalar_type() == value.scalar_type()),
      "Key and Value must have the same data type as Query");

  ET_CHECK_OR_RETURN_FALSE(
      !attn_mask.has_value() || attn_mask.value().dim() == 2,
      "Attention mask must be a 2D tensor");

  ET_CHECK_OR_RETURN_FALSE(
      !attn_mask.has_value() ||
          attn_mask.value().scalar_type() == query.scalar_type(),
      "Attention mask must be a 2D tensor");

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

  auto q_seq_len = query.size(2);

  ET_SWITCH_FLOAT_TYPES(
      query.scalar_type(), ctx, "flash_attention", CTYPE, [&] {
        // TODO we need to re-evaluate this for ARM CPUs
        // And there can be many so instead of templatizing
        // we might consider another appraoch
        if (q_seq_len >= 768) {
          sdpa::impl::cpu_flash_attention<CTYPE, 256, 512>(
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
        } else if (q_seq_len >= 192) {
          sdpa::impl::cpu_flash_attention<CTYPE, 64, 512>(
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
    const optional<Tensor>& v_scales = nullopt) {
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

  bool is_seq_at_dim_1{true};
  if (q.scalar_type() == ScalarType::Char) {
    is_seq_at_dim_1 = false;
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

  const int64_t seq_len = q.size(1);
  auto q_seq_len = q.size(1);

  const int64_t num_keys_for_causal_attention = start_pos + seq_len;

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(output, q.sizes()) == Error::Ok,
      InvalidArgument,
      output);

  // TODO(task): replace the template param selection logic
  // with whatever apprpriately makes more sense for
  ET_SWITCH_FLOAT_TYPES(
      output.scalar_type(), ctx, "flash_attention", CTYPE, [&] {
        // TODO we need to re-evaluate this for ARM CPUs
        // And there can be many so instead of templatizing
        // we might consider another appraoch
        if (q_seq_len >= 768) {
          sdpa::impl::cpu_flash_attention<CTYPE, 256, 512>(
              output,
              q,
              k,
              v,
              dropout_p,
              is_causal,
              attn_mask,
              scale,
              nullopt, // q_zero_points
              nullopt, // q_scales
              nullopt, // k_zero_points
              nullopt, // k_scales
              nullopt, // v_zero_points
              nullopt, // v_scales
              is_seq_at_dim_1, /* is_seq_at_dim_1 */
              start_pos,
              num_keys_for_causal_attention);
        } else if (q_seq_len >= 192) {
          sdpa::impl::cpu_flash_attention<CTYPE, 64, 512>(
              output,
              q,
              k,
              v,
              dropout_p,
              is_causal,
              attn_mask,
              scale,
              nullopt, // q_zero_points
              nullopt, // q_scales
              nullopt, // k_zero_points
              nullopt, // k_scales
              nullopt, // v_zero_points
              nullopt, // v_scales
              is_seq_at_dim_1, /* is_seq_at_dim_1 */
              start_pos,
              num_keys_for_causal_attention);
        } else {
          sdpa::impl::cpu_flash_attention<CTYPE, 32, 512>(
              output,
              q,
              k,
              v,
              dropout_p,
              is_causal,
              attn_mask,
              scale,
              nullopt, // q_zero_points
              nullopt, // q_scales
              nullopt, // k_zero_points
              nullopt, // k_scales
              nullopt, // v_zero_points
              nullopt, // v_scales
              is_seq_at_dim_1, /* is_seq_at_dim_1 */
              start_pos,
              num_keys_for_causal_attention);
        }
      });
  return output;
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
