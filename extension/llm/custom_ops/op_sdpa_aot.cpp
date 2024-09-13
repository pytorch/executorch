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
    const c10::optional<at::Tensor> attn_mask,
    const double dropout_p,
    const bool is_causal,
    // @lint-ignore CLANGTIDY facebook-hte-ParameterMightThrowOnCopy
    const c10::optional<double> scale) {
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

} // namespace native
} // namespace executor
} // namespace torch

TORCH_LIBRARY(llama, m) {
  m.def(
      "sdpa_with_kv_cache(Tensor query, Tensor key, Tensor value, Tensor(a!) key_cache, "
      "Tensor(b!) value_cache, SymInt start_pos, SymInt seq_len, Tensor? attn_mask=None, "
      "float drpout_p=0.0, bool is_causal=False, float? scale=None) -> Tensor");
  m.def(
      "sdpa_with_kv_cache.out(Tensor query, Tensor key, Tensor value, Tensor(a!) key_cache, "
      "Tensor(b!) value_cache, SymInt start_pos, SymInt seq_len, Tensor? attn_mask=None, "
      "float drpout_p=0.0, bool is_causal=False, float? scale=None, *, Tensor(c!) out) -> Tensor(c!)");
}

TORCH_LIBRARY_IMPL(llama, CompositeExplicitAutograd, m) {
  m.impl(
      "sdpa_with_kv_cache", torch::executor::native::sdpa_with_kv_cache_aten);
  m.impl(
      "sdpa_with_kv_cache.out",
      WRAP_TO_ATEN(
          torch::executor::native::sdpa_with_kv_cache_out_no_context, 11));
}
