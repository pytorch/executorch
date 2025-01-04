/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <executorch/kernels/optimized/vec/functional.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

namespace {

template <typename T>
constexpr bool is_half_or_bf16_v = std::is_same_v<T, exec_aten::Half> ||
    std::is_same_v<T, exec_aten::BFloat16>;

template <
    typename CTYPE_IN,
    typename CTYPE_OUT,
    typename std::enable_if<
        std::is_same_v<CTYPE_IN, CTYPE_OUT> && !is_half_or_bf16_v<CTYPE_IN> &&
            !is_half_or_bf16_v<CTYPE_OUT>,
        int>::type = 0>
void sigmoid_data(
    const CTYPE_IN* in_data,
    const size_t numel,
    CTYPE_OUT* out_data) {
  using Vec = at::vec::Vectorized<CTYPE_IN>;
  at::vec::map<CTYPE_IN>(
      [](Vec x) {
        auto one_plus_exp = x.neg().exp() + Vec(static_cast<CTYPE_IN>(1.0));
        return one_plus_exp.reciprocal();
      },
      out_data,
      in_data,
      numel);
}

template <
    typename CTYPE_IN,
    typename CTYPE_OUT,
    typename std::enable_if<
        !std::is_same_v<CTYPE_IN, CTYPE_OUT> || is_half_or_bf16_v<CTYPE_IN> ||
            is_half_or_bf16_v<CTYPE_OUT>,
        int>::type = 0>
void sigmoid_data(
    const CTYPE_IN* in_data,
    const size_t numel,
    CTYPE_OUT* out_data) {
  for (size_t i = 0; i < numel; i++) {
    CTYPE_OUT xi = static_cast<CTYPE_OUT>(in_data[i]);
    out_data[i] = (1.0f / (1.0f + std::exp(-xi)));
  }
}

} // namespace

using Tensor = exec_aten::Tensor;

Tensor&
opt_sigmoid_out(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx, in.scalar_type() != ScalarType::Bool, InvalidArgument, out);
  ET_KERNEL_CHECK(ctx, tensor_is_floating_type(out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, in.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ScalarType in_type = in.scalar_type();
  ScalarType out_type = out.scalar_type();
  ET_SWITCH_REALHB_TYPES(in_type, ctx, "sigmoid.out", CTYPE_IN, [&]() {
    ET_SWITCH_FLOATH_TYPES(out_type, ctx, "sigmoid.out", CTYPE_OUT, [&]() {
      sigmoid_data<CTYPE_IN, CTYPE_OUT>(
          in.const_data_ptr<CTYPE_IN>(),
          in.numel(),
          out.mutable_data_ptr<CTYPE_OUT>());
    });
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
