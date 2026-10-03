/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>

#include <executorch/kernels/optimized/utils/math_utils.h>
#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/kernels/portable/cpu/util/reduce_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

#include <type_traits>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;

// Forward decl of the portable kernel — used as a fallback for the shapes and
// dtype combinations the optimized path doesn't specialize. Both libraries
// live in the same binary, so a direct call is fine.
Tensor& mean_dim_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    optional<ArrayRef<int64_t>> dim_list,
    bool keepdim,
    optional<ScalarType> dtype,
    Tensor& out);

namespace {

// Contiguous innermost mean-reduction: one scalar out per row of
// `reduce_size` elements.
//
// Accumulates *and divides* in `executorch::utils::compute_dtype<CTYPE>` —
// fp32 for Half/BFloat16, the dtype itself for Float/Double. This is the
// accumulator convention already used by moments_utils.h /
// op_native_layer_norm in this directory, and it matches the portable mean
// kernel's `ACC`. CTYPE rounding happens exactly once, on store: rounding the
// running sum to a 16-bit type before the divide saturates (a 4096-element
// Half row summing past 65504 becomes inf), and reducing a Double row through
// an fp32 accumulator both loses ~9 digits and overflows above ~3.4e38.
template <typename CTYPE>
inline void mean_innermost(
    const CTYPE* in,
    CTYPE* out,
    int64_t outer_size,
    int64_t reduce_size) {
  using ACC = executorch::utils::compute_dtype<CTYPE>;
  using Vec = at::vec::Vectorized<ACC>;
  constexpr int64_t kVecSize = static_cast<int64_t>(Vec::size());
  const ACC denom = static_cast<ACC>(reduce_size);

  for (int64_t i = 0; i < outer_size; ++i) {
    const CTYPE* row = in + i * reduce_size;
    Vec acc_vec(static_cast<ACC>(0));
    int64_t j = 0;
    for (; j + kVecSize - 1 < reduce_size; j += kVecSize) {
      if constexpr (std::is_same_v<CTYPE, ACC>) {
        acc_vec = acc_vec + Vec::loadu(row + j);
      } else {
        // Half / BFloat16: widen to the accumulator type before adding.
        ACC tmp[kVecSize];
        for (int64_t k = 0; k < kVecSize; ++k) {
          tmp[k] = static_cast<ACC>(row[j + k]);
        }
        acc_vec = acc_vec + Vec::loadu(tmp);
      }
    }
    ACC acc =
        at::vec::vec_reduce_all<ACC>([](Vec a, Vec b) { return a + b; }, acc_vec);
    for (; j < reduce_size; ++j) {
      acc += static_cast<ACC>(row[j]);
    }
    out[i] = static_cast<CTYPE>(acc / denom);
  }
}

} // namespace

Tensor& opt_mean_dim_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    optional<ArrayRef<int64_t>> dim_list,
    bool keepdim,
    optional<ScalarType> dtype,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      check_mean_dim_args(in, dim_list, keepdim, dtype, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(in), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx,
      resize_reduction_out(in, dim_list, keepdim, out) == Error::Ok,
      InvalidArgument,
      out);

  // Vectorized fast path: contiguous tensor, single innermost-dim reduction,
  // same input/output dtype. Covers the RMSNorm reduction pattern. The
  // isFloatingType() guard is exactly the set ET_SWITCH_FLOATHBF16_TYPES
  // handles, so the switch below can never reach its ctx.fail() default.
  // Everything else falls through to the portable kernel.
  if (in.numel() > 0 && dim_list.has_value() && dim_list.value().size() == 1 &&
      in.scalar_type() == out.scalar_type() &&
      executorch::runtime::isFloatingType(in.scalar_type())) {
    const int64_t d = dim_list.value()[0] < 0 ? dim_list.value()[0] + in.dim()
                                              : dim_list.value()[0];
    if (d >= 0 && d < in.dim() && d == in.dim() - 1 &&
        tensor_is_contiguous(in)) {
      const int64_t reduce_size = in.size(d);
      const int64_t outer_size = in.numel() / reduce_size;

      // @lint-ignore CLANGTIDY facebook-hte-CArray
      static constexpr const char op_name[] = "mean.out";
      ET_SWITCH_FLOATHBF16_TYPES(in.scalar_type(), ctx, op_name, CTYPE, [&] {
        mean_innermost<CTYPE>(
            in.const_data_ptr<CTYPE>(),
            out.mutable_data_ptr<CTYPE>(),
            outer_size,
            reduce_size);
      });
      return out;
    }
  }

  // Fallback.
  return mean_dim_out(ctx, in, dim_list, keepdim, dtype, out);
}

Tensor& opt_mean_dtype_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    optional<ScalarType> dtype,
    Tensor& out) {
  return opt_mean_dim_out(ctx, in, ArrayRef<int64_t>(), false, dtype, out);
}

} // namespace native
} // namespace executor
} // namespace torch
