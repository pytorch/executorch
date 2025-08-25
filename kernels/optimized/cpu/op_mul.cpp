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
#include <executorch/kernels/portable/cpu/op_mul.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h> // IWYU pragma: export
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;

Tensor& opt_mul_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  (void)ctx;

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType out_type = out.scalar_type();

  if (b.numel() == 1) {
    if (a_type == b_type && a_type == out_type && a_type != ScalarType::Half &&
        a_type != ScalarType::BFloat16) {
      ET_KERNEL_CHECK(
          ctx,
          resize_to_broadcast_target_size(a, b, out) == Error::Ok,
          InvalidArgument,
          out);

      ET_SWITCH_REALB_TYPES(a_type, ctx, "mul.out", CTYPE, [&]() {
        ET_SWITCH_REALB_TYPES(b_type, ctx, "mul.out", CTYPE_B, [&]() {
          CTYPE_B b_val = *b.const_data_ptr<CTYPE_B>();
          CTYPE b_casted = static_cast<CTYPE>(b_val);

          using Vec = at::vec::Vectorized<CTYPE>;
          at::vec::map<CTYPE>(
              [b_casted](Vec x) { return x * Vec(b_casted); },
              out.mutable_data_ptr<CTYPE>(),
              a.const_data_ptr<CTYPE>(),
              out.numel());
        });
      });
      return out;
    }
  } else if (a.numel() == 1) {
    return opt_mul_out(ctx, b, a, out);
  }

  auto selected_optimized_path = select_optimized_path(a, b, out);
  if (selected_optimized_path == ElementwiseOptimizedPath::kTreatAs1d) {
    ET_KERNEL_CHECK(
        ctx,
        resize_to_broadcast_target_size(a, b, out) == Error::Ok,
        InvalidArgument,
        out);

    if (executorch::runtime::isComplexType(out_type)) {
      ET_KERNEL_CHECK(
          ctx, a_type == b_type && a_type == out_type, InvalidArgument, out);

      ET_SWITCH_COMPLEXH_TYPES(out_type, ctx, "mul.out", CTYPE, [&]() {
        using Vec = at::vec::Vectorized<CTYPE>;
        at::vec::map2<CTYPE>(
            [](Vec x, Vec y) { return x * y; },
            out.mutable_data_ptr<CTYPE>(),
            a.const_data_ptr<CTYPE>(),
            b.const_data_ptr<CTYPE>(),
            out.numel());
      });
    } else {
      ET_SWITCH_REALB_TYPES(out_type, ctx, "mul.out", CTYPE, [&]() {
        using Vec = at::vec::Vectorized<CTYPE>;
        at::vec::map2<CTYPE>(
            [](Vec x, Vec y) { return x * y; },
            out.mutable_data_ptr<CTYPE>(),
            a.const_data_ptr<CTYPE>(),
            b.const_data_ptr<CTYPE>(),
            out.numel());
      });
    }
  } else if (selected_optimized_path != ElementwiseOptimizedPath::kNone) {
    if (executorch::runtime::isComplexType(out_type)) {
      ET_KERNEL_CHECK(
          ctx, a_type == b_type && a_type == out_type, InvalidArgument, out);

      ET_SWITCH_COMPLEXH_TYPES(out_type, ctx, "mul.out", CTYPE, [&]() {
        auto mul_lambda = [](auto x, auto y) { return x * y; };
        torch::executor::handle_broadcast_elementwise<CTYPE>(
            ctx, mul_lambda, a, b, out, selected_optimized_path);
      });
    } else {
      ET_SWITCH_REALB_TYPES(out_type, ctx, "mul.out", CTYPE, [&]() {
        auto mul_lambda = [](auto x, auto y) { return x * y; };
        torch::executor::handle_broadcast_elementwise<CTYPE>(
            ctx, mul_lambda, a, b, out, selected_optimized_path);
      });
    }
  } else {
    utils::mul_out(ctx, a, b, out);
  }

  return out;
}

Tensor& opt_mul_scalar_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    Tensor& out) {
  (void)ctx;

  ScalarType a_type = a.scalar_type();
  ScalarType common_type =
      utils::promote_type_with_scalar(a_type, b, /*half_to_float*/ false);
  ScalarType out_type = out.scalar_type();

  ET_CHECK(common_type == out_type);

  if (common_type == ScalarType::Half || common_type == ScalarType::BFloat16) {
    common_type = ScalarType::Float;
  }

  // Resize for dynamic shape
  auto error = resize_tensor(out, a.sizes());
  ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");

  if (a_type == common_type && a_type == out_type &&
      a_type != ScalarType::Half && a_type != ScalarType::BFloat16) {
    ET_SWITCH_REALB_TYPES(a_type, ctx, "mul.Scalar_out", CTYPE, [&]() {
      CTYPE b_casted = utils::scalar_to<CTYPE>(b);

      using Vec = at::vec::Vectorized<CTYPE>;
      at::vec::map<CTYPE>(
          [b_casted](Vec x) { return x * Vec(b_casted); },
          out.mutable_data_ptr<CTYPE>(),
          a.const_data_ptr<CTYPE>(),
          out.numel());
    });
  } else {
    utils::mul_scalar_out(ctx, a, b, out);
  }

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
