/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/native/cpu/Elu.h>

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/kernel/thread_parallel_interface.h>
#include <executorch/runtime/platform/assert.h>

namespace torch::executor::native {

namespace {
template <typename CTYPE>
void elu(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale,
    Tensor& out) {
  const CTYPE* in_data = input.const_data_ptr<CTYPE>();
  CTYPE* out_data = out.mutable_data_ptr<CTYPE>();
  using MathT =
      std::conditional_t<c10::is_reduced_floating_point_v<CTYPE>, float, CTYPE>;
  const auto math_alpha = utils::scalar_to<MathT>(alpha);
  const auto math_scale = utils::scalar_to<MathT>(scale);
  const auto math_input_scale = utils::scalar_to<MathT>(input_scale);
  const auto scalar_func =
      at::native::get_scalar_elu_elementwise_func<CTYPE, MathT>(
          math_alpha, math_scale, math_input_scale);
  const auto vec_func = at::native::get_vectorized_elu_elementwise_func<CTYPE>(
      math_alpha, math_scale, math_input_scale);

  ::executorch::extension::parallel_for(
      0,
      out.numel(),
      ::executorch::extension::internal::GRAIN_SIZE,
      [&](const auto& begin, const auto& end) {
        using Vec = at::vec::Vectorized<CTYPE>;
        const auto vectorized_begin =
            begin + (Vec::size() - begin % Vec::size()) % Vec::size();
        const auto vectorized_end = end - (end % Vec::size());
        // Scalar prologue.
        for (const auto idx : c10::irange(begin, vectorized_begin)) {
          out_data[idx] = scalar_func(in_data[idx]);
        }

        // Main vectorized loop.
        for (auto idx = vectorized_begin; idx < vectorized_end;
             idx += Vec::size()) {
          auto result_vec = vec_func(Vec::loadu(&in_data[idx]));
          result_vec.store(&out_data[idx]);
        }

        // Scalar epilogue.
        for (const auto idx : c10::irange(vectorized_end, end)) {
          out_data[idx] = scalar_func(in_data[idx]);
        }
      });
}
} // namespace

Tensor& opt_elu_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale,
    Tensor& out) {
  ET_KERNEL_CHECK(ctx, tensors_have_same_dtype(in, out), InvalidArgument, out);
  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, in.sizes()) == Error::Ok, InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, tensor_is_floating_type(in), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, tensors_have_same_dtype(in, out), InvalidArgument, out);

  ET_SWITCH_FLOATHBF16_TYPES(in.scalar_type(), ctx, "elu.out", CTYPE, [&]() {
    elu<CTYPE>(ctx, in, alpha, scale, input_scale, out);
  });
  return out;
}

} // namespace torch::executor::native
