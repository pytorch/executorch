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
#include <executorch/extension/llm/custom_ops/op_update_quantized_cache.h>

#include <torch/library.h>

namespace torch {
namespace executor {

namespace native {
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
  WRAP_TO_ATEN(sdpa_with_kv_cache_out_no_context, 11)
  (q_projected,
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
  exec_aten::RuntimeContext context{};
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
    const c10::optional<at::Tensor> attn_mask,
    const double dropout_p,
    const bool is_causal,
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const c10::optional<double> scale) {
  auto output = at::empty_like(q);
  WRAP_TO_ATEN(custom_sdpa_out_no_context, 8)
  (q, k, v, start_pos, attn_mask, dropout_p, is_causal, scale, output);
  return output;
}

Tensor& update_quantized_cache_out_no_context(
    const Tensor& value,
    Tensor& cache,
    const int64_t start_pos,
    Tensor& output) {
  exec_aten::RuntimeContext context{};
  return torch::executor::native::update_quantized_cache_out(
      context, value, cache, start_pos, output);
}

at::Tensor update_quantized_cache_aten(
    const at::Tensor& value,
    at::Tensor& cache,
    const int64_t start_pos) {
  auto output = at::empty({1});
  WRAP_TO_ATEN(update_quantized_cache_out_no_context, 3)
  (value, cache, start_pos, output);
  return output;
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
      "update_quantized_cache(Tensor value, Tensor(a!) cache, "
      "SymInt start_pos) -> Tensor");
  m.def(
      "update_quantized_cache.out(Tensor value, Tensor(a!) cache, "
      "SymInt start_pos, *, Tensor(b!) out) -> Tensor(b!)");
}

// TODO: Rename this file to op_custom_ops_aot.cpp
TORCH_LIBRARY_IMPL(llama, CompositeExplicitAutograd, m) {
  m.impl(
      "sdpa_with_kv_cache", torch::executor::native::sdpa_with_kv_cache_aten);
  m.impl(
      "sdpa_with_kv_cache.out",
      WRAP_TO_ATEN(
          torch::executor::native::sdpa_with_kv_cache_out_no_context, 11));
  m.impl("custom_sdpa", torch::executor::native::custom_sdpa_aten);
  m.impl(
      "custom_sdpa.out",
      WRAP_TO_ATEN(torch::executor::native::custom_sdpa_out_no_context, 8));
  m.impl(
      "update_quantized_cache",
      torch::executor::native::update_quantized_cache_aten);
  m.impl(
      "update_quantized_cache.out",
      WRAP_TO_ATEN(
          torch::executor::native::update_quantized_cache_out_no_context, 3));
}
