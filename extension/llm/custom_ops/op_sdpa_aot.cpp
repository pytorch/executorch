/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/aten_util/make_aten_functor_from_et_functor.h>
#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/extension/llm/custom_ops/op_sdpa.h>
#include <executorch/extension/llm/custom_ops/op_update_cache.h>

#include <torch/library.h>

namespace torch {
namespace executor {

namespace native {
namespace {
template <typename EType, typename AType>
auto to_et_arg(AType&& value) {
  return executorch::extension::internal::type_convert<AType, EType>(
      std::forward<AType>(value));
}

at::Tensor& copy_et_result_to_out(Tensor& et_result, at::Tensor& out) {
  auto converted_result =
      executorch::extension::internal::type_convert<Tensor&, at::Tensor>(
          et_result)
          .call();
  at::native::resize_output(out, converted_result.sizes());
  out.copy_(converted_result);
  return out;
}
} // namespace

Tensor& sdpa_with_kv_cache_out_no_context(
    const Tensor& q_projected,
    const Tensor& k_projected,
    const Tensor& v_projected,
    Tensor& key_cache,
    Tensor& value_cache,
    const int64_t start_pos,
    const int64_t seq_len,
    // @lint-ignore CLANGTIDY facebook-hte-ConstantArgumentPassByValue
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const optional<Tensor> attn_mask,
    const double dropout_p,
    const bool is_causal,
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const optional<double> scale,
    Tensor& output);

at::Tensor sdpa_with_kv_cache_aten(
    const at::Tensor& q_projected,
    const at::Tensor& k_projected,
    const at::Tensor& v_projected,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    const int64_t start_pos,
    const int64_t seq_len,
    // @lint-ignore CLANGTIDY facebook-hte-ConstantArgumentPassByValue
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const std::optional<at::Tensor> attn_mask,
    const double dropout_p,
    const bool is_causal,
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const std::optional<double> scale);

at::Tensor& sdpa_with_kv_cache_out_aten(
    const at::Tensor& q_projected,
    const at::Tensor& k_projected,
    const at::Tensor& v_projected,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    const int64_t start_pos,
    const int64_t seq_len,
    const std::optional<at::Tensor> attn_mask,
    const double dropout_p,
    const bool is_causal,
    const std::optional<double> scale,
    at::Tensor& output);

Tensor& custom_sdpa_out_no_context(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const int64_t start_pos,
    // @lint-ignore CLANGTIDY facebook-hte-ConstantArgumentPassByValue
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const optional<Tensor> attn_mask,
    const double dropout_p,
    const bool is_causal,
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const optional<double> scale,
    Tensor& output);

at::Tensor custom_sdpa_aten(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const int64_t start_pos,
    // @lint-ignore CLANGTIDY facebook-hte-ConstantArgumentPassByValue
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const std::optional<at::Tensor> attn_mask,
    const double dropout_p,
    const bool is_causal,
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const std::optional<double> scale);

at::Tensor& custom_sdpa_out_aten(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const int64_t start_pos,
    const std::optional<at::Tensor> attn_mask,
    const double dropout_p,
    const bool is_causal,
    const std::optional<double> scale,
    at::Tensor& output);

Tensor& custom_quantized_sdpa_out_no_context(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const int64_t start_pos,
    // @lint-ignore CLANGTIDY facebook-hte-ConstantArgumentPassByValue
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const optional<Tensor> attn_mask,
    const double dropout_p,
    const bool is_causal,
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const optional<double> scale,
    const optional<Tensor> q_zero_points,
    const optional<Tensor> q_scales,
    const optional<Tensor> k_zero_points,
    const optional<Tensor> k_scales,
    const optional<Tensor> v_zero_points,
    const optional<Tensor> v_scales,
    const bool is_seq_at_dim_2,
    Tensor& output);

at::Tensor custom_quantized_sdpa_aten(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const int64_t start_pos,
    // @lint-ignore CLANGTIDY facebook-hte-ConstantArgumentPassByValue
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const std::optional<at::Tensor> attn_mask,
    const double dropout_p,
    const bool is_causal,
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const std::optional<double> scale,
    const std::optional<at::Tensor>& q_zero_points,
    const std::optional<at::Tensor>& q_scales,
    const std::optional<at::Tensor>& k_zero_points,
    const std::optional<at::Tensor>& k_scales,
    const std::optional<at::Tensor>& v_zero_points,
    const std::optional<at::Tensor>& v_scales,
    const bool is_seq_at_dim_2);

at::Tensor& custom_quantized_sdpa_out_aten(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const int64_t start_pos,
    const std::optional<at::Tensor> attn_mask,
    const double dropout_p,
    const bool is_causal,
    const std::optional<double> scale,
    const std::optional<at::Tensor>& q_zero_points,
    const std::optional<at::Tensor>& q_scales,
    const std::optional<at::Tensor>& k_zero_points,
    const std::optional<at::Tensor>& k_scales,
    const std::optional<at::Tensor>& v_zero_points,
    const std::optional<at::Tensor>& v_scales,
    const bool is_seq_at_dim_2,
    at::Tensor& output);

Tensor& update_cache_out_no_context(
    const Tensor& value,
    Tensor& cache,
    const int64_t start_pos,
    Tensor& output);

at::Tensor update_cache_aten(
    const at::Tensor& value,
    at::Tensor& cache,
    const int64_t start_pos);

at::Tensor& update_cache_out_aten(
    const at::Tensor& value,
    at::Tensor& cache,
    const int64_t start_pos,
    at::Tensor& output);

// New functions for update_cache_with_indices
Tensor& update_cache_with_indices_out_no_context(
    const Tensor& value,
    Tensor& cache,
    const int64_t start_pos,
    const Tensor& indices,
    Tensor& output);

at::Tensor update_cache_with_indices_aten(
    const at::Tensor& value,
    at::Tensor& cache,
    const int64_t start_pos,
    const at::Tensor& indices);

at::Tensor& update_cache_with_indices_out_aten(
    const at::Tensor& value,
    at::Tensor& cache,
    const int64_t start_pos,
    const at::Tensor& indices,
    at::Tensor& output);

Tensor& recurrent_gated_delta_rule_out_no_context(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& g,
    const Tensor& beta,
    Tensor& recurrent_state,
    Tensor& output);

at::Tensor recurrent_gated_delta_rule_aten(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& g,
    const at::Tensor& beta,
    at::Tensor& recurrent_state);

at::Tensor& recurrent_gated_delta_rule_out_aten(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& g,
    const at::Tensor& beta,
    at::Tensor& recurrent_state,
    at::Tensor& output);

Tensor& sdpa_with_kv_cache_out_no_context(
    const Tensor& q_projected,
    const Tensor& k_projected,
    const Tensor& v_projected,
    Tensor& key_cache,
    Tensor& value_cache,
    const int64_t start_pos,
    const int64_t seq_len,
    // @lint-ignore CLANGTIDY facebook-hte-ConstantArgumentPassByValue
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const optional<Tensor> attn_mask,
    const double dropout_p,
    const bool is_causal,
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const optional<double> scale,
    Tensor& output) {
  executorch::runtime::KernelRuntimeContext context{};
  return torch::executor::native::sdpa_with_kv_cache_out(
      context,
      q_projected,
      k_projected,
      v_projected,
      key_cache,
      value_cache,
      start_pos,
      seq_len,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      output);
}

at::Tensor sdpa_with_kv_cache_aten(
    const at::Tensor& q_projected,
    const at::Tensor& k_projected,
    const at::Tensor& v_projected,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    const int64_t start_pos,
    const int64_t seq_len,
    // @lint-ignore CLANGTIDY facebook-hte-ConstantArgumentPassByValue
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const std::optional<at::Tensor> attn_mask,
    const double dropout_p,
    const bool is_causal,
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const std::optional<double> scale) {
  auto output = at::empty_like(q_projected);
  sdpa_with_kv_cache_out_aten(
      q_projected,
      k_projected,
      v_projected,
      key_cache,
      value_cache,
      start_pos,
      seq_len,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      output);
  return output;
}

at::Tensor& sdpa_with_kv_cache_out_aten(
    const at::Tensor& q_projected,
    const at::Tensor& k_projected,
    const at::Tensor& v_projected,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    const int64_t start_pos,
    const int64_t seq_len,
    const std::optional<at::Tensor> attn_mask,
    const double dropout_p,
    const bool is_causal,
    const std::optional<double> scale,
    at::Tensor& output) {
  auto q_et = to_et_arg<Tensor>(q_projected);
  auto k_et = to_et_arg<Tensor>(k_projected);
  auto v_et = to_et_arg<Tensor>(v_projected);
  auto key_cache_et = to_et_arg<Tensor&>(key_cache);
  auto value_cache_et = to_et_arg<Tensor&>(value_cache);
  auto attn_mask_et = to_et_arg<optional<Tensor>>(attn_mask);
  auto scale_et = to_et_arg<optional<double>>(scale);
  auto output_et = to_et_arg<Tensor&>(output);
  auto& et_result = sdpa_with_kv_cache_out_no_context(
      q_et.call(),
      k_et.call(),
      v_et.call(),
      key_cache_et.call(),
      value_cache_et.call(),
      start_pos,
      seq_len,
      attn_mask_et.call(),
      dropout_p,
      is_causal,
      scale_et.call(),
      output_et.call());
  return copy_et_result_to_out(et_result, output);
}

Tensor& custom_sdpa_out_no_context(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const int64_t start_pos,
    // @lint-ignore CLANGTIDY facebook-hte-ConstantArgumentPassByValue
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const optional<Tensor> attn_mask,
    const double dropout_p,
    const bool is_causal,
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const optional<double> scale,
    Tensor& output) {
  executorch::aten::RuntimeContext context{};
  return torch::executor::native::custom_sdpa_out(
      context,
      q,
      k,
      v,
      start_pos,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      output);
}

at::Tensor custom_sdpa_aten(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const int64_t start_pos,
    // @lint-ignore CLANGTIDY facebook-hte-ConstantArgumentPassByValue
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const std::optional<at::Tensor> attn_mask,
    const double dropout_p,
    const bool is_causal,
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const std::optional<double> scale) {
  auto output = at::empty(q.sizes());
  custom_sdpa_out_aten(
      q, k, v, start_pos, attn_mask, dropout_p, is_causal, scale, output);
  return output;
}

at::Tensor& custom_sdpa_out_aten(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const int64_t start_pos,
    const std::optional<at::Tensor> attn_mask,
    const double dropout_p,
    const bool is_causal,
    const std::optional<double> scale,
    at::Tensor& output) {
  auto q_et = to_et_arg<Tensor>(q);
  auto k_et = to_et_arg<Tensor>(k);
  auto v_et = to_et_arg<Tensor>(v);
  auto attn_mask_et = to_et_arg<optional<Tensor>>(attn_mask);
  auto scale_et = to_et_arg<optional<double>>(scale);
  auto output_et = to_et_arg<Tensor&>(output);
  auto& et_result = custom_sdpa_out_no_context(
      q_et.call(),
      k_et.call(),
      v_et.call(),
      start_pos,
      attn_mask_et.call(),
      dropout_p,
      is_causal,
      scale_et.call(),
      output_et.call());
  return copy_et_result_to_out(et_result, output);
}

Tensor& custom_quantized_sdpa_out_no_context(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const int64_t start_pos,
    // @lint-ignore CLANGTIDY facebook-hte-ConstantArgumentPassByValue
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const optional<Tensor> attn_mask,
    const double dropout_p,
    const bool is_causal,
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const optional<double> scale,
    const optional<Tensor> q_zero_points,
    const optional<Tensor> q_scales,
    const optional<Tensor> k_zero_points,
    const optional<Tensor> k_scales,
    const optional<Tensor> v_zero_points,
    const optional<Tensor> v_scales,
    const bool is_seq_at_dim_2,
    Tensor& output) {
  executorch::aten::RuntimeContext context{};
  return torch::executor::native::custom_quantized_sdpa_out(
      context,
      q,
      k,
      v,
      start_pos,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      q_zero_points,
      q_scales,
      k_zero_points,
      k_scales,
      v_zero_points,
      v_scales,
      is_seq_at_dim_2,
      output);
}

at::Tensor custom_quantized_sdpa_aten(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const int64_t start_pos,
    // @lint-ignore CLANGTIDY facebook-hte-ConstantArgumentPassByValue
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const std::optional<at::Tensor> attn_mask,
    const double dropout_p,
    const bool is_causal,
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const std::optional<double> scale,
    const std::optional<at::Tensor>& q_zero_points,
    const std::optional<at::Tensor>& q_scales,
    const std::optional<at::Tensor>& k_zero_points,
    const std::optional<at::Tensor>& k_scales,
    const std::optional<at::Tensor>& v_zero_points,
    const std::optional<at::Tensor>& v_scales,
    const bool is_seq_at_dim_2) {
  auto output = at::empty(q.sizes());
  custom_quantized_sdpa_out_aten(
      q,
      k,
      v,
      start_pos,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      q_zero_points,
      q_scales,
      k_zero_points,
      k_scales,
      v_zero_points,
      v_scales,
      is_seq_at_dim_2,
      output);
  return output;
}

at::Tensor& custom_quantized_sdpa_out_aten(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const int64_t start_pos,
    const std::optional<at::Tensor> attn_mask,
    const double dropout_p,
    const bool is_causal,
    const std::optional<double> scale,
    const std::optional<at::Tensor>& q_zero_points,
    const std::optional<at::Tensor>& q_scales,
    const std::optional<at::Tensor>& k_zero_points,
    const std::optional<at::Tensor>& k_scales,
    const std::optional<at::Tensor>& v_zero_points,
    const std::optional<at::Tensor>& v_scales,
    const bool is_seq_at_dim_2,
    at::Tensor& output) {
  auto q_et = to_et_arg<Tensor>(q);
  auto k_et = to_et_arg<Tensor>(k);
  auto v_et = to_et_arg<Tensor>(v);
  auto attn_mask_et = to_et_arg<optional<Tensor>>(attn_mask);
  auto scale_et = to_et_arg<optional<double>>(scale);
  auto q_zero_points_et = to_et_arg<optional<Tensor>>(q_zero_points);
  auto q_scales_et = to_et_arg<optional<Tensor>>(q_scales);
  auto k_zero_points_et = to_et_arg<optional<Tensor>>(k_zero_points);
  auto k_scales_et = to_et_arg<optional<Tensor>>(k_scales);
  auto v_zero_points_et = to_et_arg<optional<Tensor>>(v_zero_points);
  auto v_scales_et = to_et_arg<optional<Tensor>>(v_scales);
  auto output_et = to_et_arg<Tensor&>(output);
  auto& et_result = custom_quantized_sdpa_out_no_context(
      q_et.call(),
      k_et.call(),
      v_et.call(),
      start_pos,
      attn_mask_et.call(),
      dropout_p,
      is_causal,
      scale_et.call(),
      q_zero_points_et.call(),
      q_scales_et.call(),
      k_zero_points_et.call(),
      k_scales_et.call(),
      v_zero_points_et.call(),
      v_scales_et.call(),
      is_seq_at_dim_2,
      output_et.call());
  return copy_et_result_to_out(et_result, output);
}

Tensor& update_cache_out_no_context(
    const Tensor& value,
    Tensor& cache,
    const int64_t start_pos,
    Tensor& output) {
  executorch::aten::RuntimeContext context{};
  return torch::executor::native::update_cache_out(
      context, value, cache, start_pos, output);
}

at::Tensor update_cache_aten(
    const at::Tensor& value,
    at::Tensor& cache,
    const int64_t start_pos) {
  auto output = at::empty({1});
  update_cache_out_aten(value, cache, start_pos, output);
  return output;
}

at::Tensor& update_cache_out_aten(
    const at::Tensor& value,
    at::Tensor& cache,
    const int64_t start_pos,
    at::Tensor& output) {
  auto value_et = to_et_arg<Tensor>(value);
  auto cache_et = to_et_arg<Tensor&>(cache);
  auto output_et = to_et_arg<Tensor&>(output);
  auto& et_result = update_cache_out_no_context(
      value_et.call(), cache_et.call(), start_pos, output_et.call());
  return copy_et_result_to_out(et_result, output);
}

// Implementations for update_cache_with_indices
Tensor& update_cache_with_indices_out_no_context(
    const Tensor& value,
    Tensor& cache,
    const int64_t start_pos,
    const Tensor& indices,
    Tensor& output) {
  executorch::aten::RuntimeContext context{};
  return torch::executor::native::update_cache_with_indices_out(
      context, value, cache, start_pos, indices, output);
}

at::Tensor update_cache_with_indices_aten(
    const at::Tensor& value,
    at::Tensor& cache,
    const int64_t start_pos,
    const at::Tensor& indices) {
  auto output = at::empty({1});
  update_cache_with_indices_out_aten(value, cache, start_pos, indices, output);
  return output;
}

at::Tensor& update_cache_with_indices_out_aten(
    const at::Tensor& value,
    at::Tensor& cache,
    const int64_t start_pos,
    const at::Tensor& indices,
    at::Tensor& output) {
  auto value_et = to_et_arg<Tensor>(value);
  auto cache_et = to_et_arg<Tensor&>(cache);
  auto indices_et = to_et_arg<Tensor>(indices);
  auto output_et = to_et_arg<Tensor&>(output);
  auto& et_result = update_cache_with_indices_out_no_context(
      value_et.call(),
      cache_et.call(),
      start_pos,
      indices_et.call(),
      output_et.call());
  return copy_et_result_to_out(et_result, output);
}

Tensor& recurrent_gated_delta_rule_out_no_context(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& g,
    const Tensor& beta,
    Tensor& recurrent_state,
    Tensor& output) {
  executorch::aten::RuntimeContext context{};
  return torch::executor::native::recurrent_gated_delta_rule_out(
      context, query, key, value, g, beta, recurrent_state, output);
}

at::Tensor recurrent_gated_delta_rule_aten(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& g,
    const at::Tensor& beta,
    at::Tensor& recurrent_state) {
  auto output = at::empty_like(value);
  recurrent_gated_delta_rule_out_aten(
      query, key, value, g, beta, recurrent_state, output);
  return output;
}

at::Tensor& recurrent_gated_delta_rule_out_aten(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& g,
    const at::Tensor& beta,
    at::Tensor& recurrent_state,
    at::Tensor& output) {
  auto query_et = to_et_arg<Tensor>(query);
  auto key_et = to_et_arg<Tensor>(key);
  auto value_et = to_et_arg<Tensor>(value);
  auto g_et = to_et_arg<Tensor>(g);
  auto beta_et = to_et_arg<Tensor>(beta);
  auto recurrent_state_et = to_et_arg<Tensor&>(recurrent_state);
  auto output_et = to_et_arg<Tensor&>(output);
  auto& et_result = recurrent_gated_delta_rule_out_no_context(
      query_et.call(),
      key_et.call(),
      value_et.call(),
      g_et.call(),
      beta_et.call(),
      recurrent_state_et.call(),
      output_et.call());
  return copy_et_result_to_out(et_result, output);
}

} // namespace native
} // namespace executor
} // namespace torch

TORCH_LIBRARY_FRAGMENT(llama, m) {
  m.def(
      "sdpa_with_kv_cache(Tensor query, Tensor key, Tensor value, Tensor(a!) key_cache, "
      "Tensor(b!) value_cache, SymInt start_pos, SymInt seq_len, Tensor? attn_mask=None, "
      "float drpout_p=0.0, bool is_causal=False, float? scale=None) -> Tensor");
  m.def(
      "sdpa_with_kv_cache.out(Tensor query, Tensor key, Tensor value, Tensor(a!) key_cache, "
      "Tensor(b!) value_cache, SymInt start_pos, SymInt seq_len, Tensor? attn_mask=None, "
      "float drpout_p=0.0, bool is_causal=False, float? scale=None, *, Tensor(c!) out) -> Tensor(c!)");
  m.def(
      "custom_sdpa(Tensor query, Tensor key, Tensor value, SymInt start_pos, "
      "Tensor? attn_mask=None, float drpout_p=0.0, bool is_causal=False, "
      "float? scale=None) -> Tensor");
  m.def(
      "custom_sdpa.out(Tensor query, Tensor key, Tensor value, SymInt start_pos, "
      "Tensor? attn_mask=None, float drpout_p=0.0, bool is_causal=False, "
      "float? scale=None, *, Tensor(a!) out) -> Tensor(a!)");
  m.def(
      "update_cache(Tensor value, Tensor(a!) cache, "
      "SymInt start_pos) -> Tensor");
  m.def(
      "update_cache.out(Tensor value, Tensor(a!) cache, "
      "SymInt start_pos, *, Tensor(b!) out) -> Tensor(b!)");
  m.def(
      "update_cache_with_indices(Tensor value, Tensor(a!) cache, "
      "SymInt start_pos, Tensor indices) -> Tensor");
  m.def(
      "update_cache_with_indices.out(Tensor value, Tensor(a!) cache, "
      "SymInt start_pos, Tensor indices, *, Tensor(b!) out) -> Tensor(b!)");
  m.def(
      "recurrent_gated_delta_rule(Tensor query, Tensor key, Tensor value, Tensor g, "
      "Tensor beta, Tensor(a!) recurrent_state) -> Tensor");
  m.def(
      "recurrent_gated_delta_rule.out(Tensor query, Tensor key, Tensor value, Tensor g, "
      "Tensor beta, Tensor(a!) recurrent_state, *, Tensor(b!) out) -> Tensor(b!)");
  m.def(
      "custom_quantized_sdpa(Tensor query, Tensor key, Tensor value, SymInt start_pos, "
      "Tensor? attn_mask=None, float drpout_p=0.0, bool is_causal=False, "
      "float? scale=None, Tensor? q_zero_points=None, Tensor? q_scales=None, "
      "Tensor? k_zero_points=None, Tensor? k_scales=None, Tensor? v_zero_points=None, "
      "Tensor? v_scales=None, bool is_seq_at_dim_2=False) -> Tensor");
  m.def(
      "custom_quantized_sdpa.out(Tensor query, Tensor key, Tensor value, SymInt start_pos, "
      "Tensor? attn_mask=None, float drpout_p=0.0, bool is_causal=False, "
      "float? scale=None, Tensor? q_zero_points=None, Tensor? q_scales=None, "
      "Tensor? k_zero_points=None, Tensor? k_scales=None, Tensor? v_zero_points=None, "
      "Tensor? v_scales=None, bool is_seq_at_dim_2=False, *, Tensor(a!) out) -> Tensor(a!)");
}

// TODO: Rename this file to op_custom_ops_aot.cpp
TORCH_LIBRARY_IMPL(llama, CompositeExplicitAutograd, m) {
  m.impl(
      "sdpa_with_kv_cache", torch::executor::native::sdpa_with_kv_cache_aten);
  m.impl(
      "sdpa_with_kv_cache.out",
      torch::executor::native::sdpa_with_kv_cache_out_aten);
  m.impl("custom_sdpa", torch::executor::native::custom_sdpa_aten);
  m.impl("custom_sdpa.out", torch::executor::native::custom_sdpa_out_aten);
  m.impl("update_cache", torch::executor::native::update_cache_aten);
  m.impl("update_cache.out", torch::executor::native::update_cache_out_aten);
  m.impl(
      "update_cache_with_indices",
      torch::executor::native::update_cache_with_indices_aten);
  m.impl(
      "update_cache_with_indices.out",
      torch::executor::native::update_cache_with_indices_out_aten);
  m.impl(
      "recurrent_gated_delta_rule",
      torch::executor::native::recurrent_gated_delta_rule_aten);
  m.impl(
      "recurrent_gated_delta_rule.out",
      torch::executor::native::recurrent_gated_delta_rule_out_aten);
  m.impl(
      "custom_quantized_sdpa",
      torch::executor::native::custom_quantized_sdpa_aten);
  m.impl(
      "custom_quantized_sdpa.out",
      torch::executor::native::custom_quantized_sdpa_out_aten);
}
