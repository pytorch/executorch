/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/aten_util/make_aten_functor_from_et_functor.h>
#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/extension/llm/custom_ops/op_moe.h>

#include <torch/library.h>

namespace torch {
namespace executor {
namespace native {

Tensor& quantized_moe_ffn_out_no_context(
    const Tensor& x,
    const Tensor& gate_weight,
    const Tensor& expert_bias,
    const Tensor& packed_w13,
    const Tensor& packed_w2,
    const int64_t num_activated_experts,
    const int64_t num_experts,
    const int64_t hidden_dim,
    const int64_t dim,
    const int64_t group_size,
    const int64_t weight_nbit,
    executorch::aten::string_view score_func,
    const double route_scale,
    Tensor& out) {
  executorch::runtime::KernelRuntimeContext context{};
  return torch::executor::native::quantized_moe_ffn_out(
      context,
      x,
      gate_weight,
      expert_bias,
      packed_w13,
      packed_w2,
      num_activated_experts,
      num_experts,
      hidden_dim,
      dim,
      group_size,
      weight_nbit,
      score_func,
      route_scale,
      out);
}

at::Tensor quantized_moe_ffn_aten(
    const at::Tensor& x,
    const at::Tensor& gate_weight,
    const at::Tensor& expert_bias,
    const at::Tensor& packed_w13,
    const at::Tensor& packed_w2,
    const int64_t num_activated_experts,
    const int64_t num_experts,
    const int64_t hidden_dim,
    const int64_t dim,
    const int64_t group_size,
    const int64_t weight_nbit,
    c10::string_view score_func,
    const double route_scale) {
  auto output = at::empty({x.size(0), dim}, x.options().dtype(at::kFloat));
  WRAP_TO_ATEN(quantized_moe_ffn_out_no_context, 13)
  (x,
   gate_weight,
   expert_bias,
   packed_w13,
   packed_w2,
   num_activated_experts,
   num_experts,
   hidden_dim,
   dim,
   group_size,
   weight_nbit,
   score_func,
   route_scale,
   output);
  return output;
}

} // namespace native
} // namespace executor
} // namespace torch

TORCH_LIBRARY_FRAGMENT(llama, m) {
  m.def(
      "quantized_moe_ffn(Tensor x, Tensor gate_weight, Tensor expert_bias, "
      "Tensor packed_w13, Tensor packed_w2, "
      "int num_activated_experts, int num_experts, int hidden_dim, int dim, "
      "int group_size, int weight_nbit, str score_func, float route_scale) "
      "-> Tensor");
  m.def(
      "quantized_moe_ffn.out(Tensor x, Tensor gate_weight, Tensor expert_bias, "
      "Tensor packed_w13, Tensor packed_w2, "
      "int num_activated_experts, int num_experts, int hidden_dim, int dim, "
      "int group_size, int weight_nbit, str score_func, float route_scale, "
      "*, Tensor(a!) out) -> Tensor(a!)");
  m.def("_quantized_moe_ffn_active() -> bool");
}

TORCH_LIBRARY_IMPL(llama, CompositeExplicitAutograd, m) {
  m.impl("quantized_moe_ffn", torch::executor::native::quantized_moe_ffn_aten);
  m.impl(
      "quantized_moe_ffn.out",
      WRAP_TO_ATEN(
          torch::executor::native::quantized_moe_ffn_out_no_context, 13));
  m.impl("_quantized_moe_ffn_active", []() { return true; });
}
