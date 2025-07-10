/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <executorch/kernels/optimized/cpu/binary_ops.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace kernels {
namespace impl {

namespace {
template <
    bool can_cast,
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct AddInner;

template <
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct AddInner<true, CTYPE_A, CTYPE_B, CTYPE_IN, CTYPE_OUT> {
  static void
  run(const Tensor& a, const Tensor& b, CTYPE_IN alpha_val, Tensor& out) {
    apply_binary_elementwise_fn<CTYPE_A, CTYPE_B, CTYPE_OUT>(
        // NOLINTNEXTLINE(facebook-hte-ConstantArgumentPassByValue)
        [alpha_val](const CTYPE_A val_a, const CTYPE_B val_b) {
          CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
          CTYPE_IN b_casted = static_cast<CTYPE_IN>(val_b);
          CTYPE_IN value = a_casted + alpha_val * b_casted;

          return static_cast<CTYPE_OUT>(value);
        },
        a,
        b,
        out);
  }
};

template <typename CTYPE_IN>
struct ReportCanCastBug {
  static void run(const Tensor&, const Tensor&, CTYPE_IN, Tensor&) {
    ET_DCHECK_MSG(false, "BUG: canCast should have been checked above");
  }
};

template <
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct AddInner<false, CTYPE_A, CTYPE_B, CTYPE_IN, CTYPE_OUT>
    : public ReportCanCastBug<CTYPE_IN> {};

} // namespace

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;

template <bool is_sub, const char* op_name>
Tensor& opt_add_sub_out_impl(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    const Scalar& alpha,
    Tensor& out) {
  (void)ctx;

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType out_type = out.scalar_type();

  auto selected_optimized_path = select_optimized_path(a, b, out);
  if (selected_optimized_path == ElementwiseOptimizedPath::kTreatAs1d) {
    // Resize for dynamic shape
    auto error = resize_tensor(out, a.sizes());
    ET_KERNEL_CHECK_MSG(
        ctx,
        error == Error::Ok,
        InvalidArgument,
        out,
        "Failed to resize output tensor.");

    ET_SWITCH_REALB_TYPES(a_type, ctx, op_name, CTYPE, [&]() {
      CTYPE alpha_val;
      ET_KERNEL_CHECK(
          ctx,
          torch::executor::native::utils::extract_scalar(alpha, &alpha_val),
          InvalidArgument, );
      if constexpr (is_sub) {
        alpha_val = -alpha_val;
      }
      using Vec = at::vec::Vectorized<CTYPE>;
      at::vec::map2<CTYPE>(
          [alpha_val](Vec x, Vec y) { return x + Vec(alpha_val) * y; },
          out.mutable_data_ptr<CTYPE>(),
          a.const_data_ptr<CTYPE>(),
          b.const_data_ptr<CTYPE>(),
          out.numel());
    });
  } else if (selected_optimized_path != ElementwiseOptimizedPath::kNone) {
    // Cannot apply the trick of -alpha here because alpha is Scalar without
    // support for - operator. At least not right now.
    ET_SWITCH_REALB_TYPES(out_type, ctx, op_name, CTYPE, [&]() {
      CTYPE alpha_val;
      ET_KERNEL_CHECK_MSG(
          ctx,
          torch::executor::native::utils::extract_scalar(alpha, &alpha_val),
          InvalidArgument,
          out,
          "Failed to extract scalar alpha.");
      using Vec = at::vec::Vectorized<CTYPE>;
      Vec alpha_val_vec(alpha_val);
      if constexpr (is_sub) {
        if (selected_optimized_path ==
                ElementwiseOptimizedPath::kBroadcast2dBy1dReverseArguments ||
            selected_optimized_path ==
                ElementwiseOptimizedPath::kBroadcastLastDimReverseArguments ||
            selected_optimized_path ==
                ElementwiseOptimizedPath::kBroadcastNdByNdReverseArguments) {
          auto add_lambda = [&alpha_val_vec](auto x, auto y) {
            return y - alpha_val_vec * x;
          };
          return torch::executor::handle_broadcast_elementwise<CTYPE>(
              ctx, add_lambda, a, b, out, selected_optimized_path, alpha);
        } else {
          auto add_lambda = [&alpha_val_vec](auto x, auto y) {
            return x - alpha_val_vec * y;
          };
          return torch::executor::handle_broadcast_elementwise<CTYPE>(
              ctx, add_lambda, a, b, out, selected_optimized_path, alpha);
        }
      } else {
        if (selected_optimized_path ==
                ElementwiseOptimizedPath::kBroadcast2dBy1dReverseArguments ||
            selected_optimized_path ==
                ElementwiseOptimizedPath::kBroadcastLastDimReverseArguments ||
            selected_optimized_path ==
                ElementwiseOptimizedPath::kBroadcastNdByNdReverseArguments) {
          // Reason we swap out args here is because
          // handle_broadcast_elementwise handles this selected_optimized_path
          // option a bit differently. This should really be resolved in
          // handle_broadcast_elementwise. However, the current blocker is that
          // handle_broadcast_elementwise tries to be agnostic of op. This
          // should be fixed, likely by moving lambda creation to
          // handle_broadcast_elementwise and it be aware of which op is being
          // executed.
          auto add_lambda = [&alpha_val_vec](auto x, auto y) {
            return y + alpha_val_vec * x;
          };
          return torch::executor::handle_broadcast_elementwise<CTYPE>(
              ctx, add_lambda, a, b, out, selected_optimized_path, alpha);
        } else {
          auto add_lambda = [&alpha_val_vec](auto x, auto y) {
            return x + alpha_val_vec * y;
          };
          return torch::executor::handle_broadcast_elementwise<CTYPE>(
              ctx, add_lambda, a, b, out, selected_optimized_path, alpha);
        }
      }
    });
  } else {
    ScalarType common_type =
        promoteTypes(a_type, b_type, /*half_to_float*/ true);
    ET_KERNEL_CHECK(ctx, canCast(common_type, out_type), InvalidArgument, out);

    ET_KERNEL_CHECK(
        ctx,
        resize_to_broadcast_target_size(a, b, out) == Error::Ok,
        InvalidArgument,
        out);

    ET_SWITCH_REALHBBF16_TYPES(a_type, ctx, op_name, CTYPE_A, [&]() {
      ET_SWITCH_REALHBBF16_TYPES(b_type, ctx, op_name, CTYPE_B, [&]() {
        using CTYPE_IN = typename torch::executor::
            promote_types<CTYPE_A, CTYPE_B, /*half_to_float*/ true>::type;
        ET_DCHECK(CppTypeToScalarType<CTYPE_IN>::value == common_type);
        ET_SWITCH_REALHBBF16_TYPES(out_type, ctx, op_name, CTYPE_OUT, [&]() {
          CTYPE_IN alpha_val;
          ET_KERNEL_CHECK(
              ctx,
              torch::executor::native::utils::extract_scalar(alpha, &alpha_val),
              InvalidArgument, );
          if constexpr (is_sub) {
            alpha_val = -alpha_val;
          }

          AddInner<
              can_cast<CTYPE_IN, CTYPE_OUT>::value,
              CTYPE_A,
              CTYPE_B,
              CTYPE_IN,
              CTYPE_OUT>::run(a, b, alpha_val, out);
        });
      });
    });
  }

  return out;
}
} // namespace impl
} // namespace kernels
} // namespace executor
} // namespace torch
