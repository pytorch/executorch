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
#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace kernels {
namespace impl {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;

template <bool is_sub, const char* op_name>
Tensor& opt_add_sub_out_impl(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    const Scalar& alpha,
    Tensor& out) {
  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType out_type = out.scalar_type();

  auto selected_optimized_path = select_optimized_path(a, b, out);

  if (executorch::runtime::isComplexType(a_type) ||
      executorch::runtime::isComplexType(b_type) ||
      executorch::runtime::isComplexType(out_type)) {
    // TODO: The current implementation for complex dtypes enforces that the
    // inputs and output tensors have same dtype and shape. Handle mixed dtypes
    // and broadcasting in the future.
    ET_KERNEL_CHECK(
        ctx,
        a_type == b_type && a_type == out_type &&
            selected_optimized_path == ElementwiseOptimizedPath::kTreatAs1d,
        InvalidArgument,
        out);
    ET_SWITCH_COMPLEXH_TYPES(out_type, ctx, op_name, CTYPE, [&]() {
      CTYPE alpha_val = torch::executor::native::utils::scalar_to<CTYPE>(alpha);
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
    return out;
  }

  if (selected_optimized_path == ElementwiseOptimizedPath::kTreatAs1d) {
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
    ET_SWITCH_REALB_TYPES(out_type, ctx, op_name, CTYPE, [&]() -> void {
      CTYPE alpha_val;
      ET_KERNEL_CHECK_MSG(
          ctx,
          torch::executor::native::utils::extract_scalar(alpha, &alpha_val),
          InvalidArgument,
          ,
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
          torch::executor::handle_broadcast_elementwise<CTYPE>(
              ctx, add_lambda, a, b, out, selected_optimized_path, alpha);
        } else {
          auto add_lambda = [&alpha_val_vec](auto x, auto y) {
            return x - alpha_val_vec * y;
          };
          torch::executor::handle_broadcast_elementwise<CTYPE>(
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
          torch::executor::handle_broadcast_elementwise<CTYPE>(
              ctx, add_lambda, a, b, out, selected_optimized_path, alpha);
        } else {
          auto add_lambda = [&alpha_val_vec](auto x, auto y) {
            return x + alpha_val_vec * y;
          };
          torch::executor::handle_broadcast_elementwise<CTYPE>(
              ctx, add_lambda, a, b, out, selected_optimized_path, alpha);
        }
      }
    });
  } else {
    ScalarType common_type = promoteTypes(a_type, b_type);
    ScalarType compute_type =
        native::utils::internal::get_compute_type(common_type);

    ET_SWITCH_REALB_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
      CTYPE_COMPUTE val_alpha;
      ET_KERNEL_CHECK(
          ctx,
          native::utils::extract_scalar(alpha, &val_alpha),
          InvalidArgument, );
      if constexpr (is_sub) {
        val_alpha = -val_alpha;
      }
      native::utils::apply_bitensor_elementwise_fn<
          CTYPE_COMPUTE,
          op_name,
          native::utils::SupportedTensorDtypes::REALHBBF16>(
          [val_alpha](const auto val_a, const auto val_b) {
            return val_a + val_alpha * val_b;
          },
          ctx,
          a,
          native::utils::SupportedTensorDtypes::REALHBBF16,
          b,
          native::utils::SupportedTensorDtypes::REALHBBF16,
          out);
    });
  }

  return out;
}
} // namespace impl
} // namespace kernels
} // namespace executor
} // namespace torch
