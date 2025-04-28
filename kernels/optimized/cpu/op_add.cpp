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

#include <executorch/kernels/optimized/cpu/op_add_sub_impl.h>

namespace torch {
namespace executor {
namespace native {
using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;

Tensor& opt_add_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    const Scalar& alpha,
    Tensor& out) {
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

      ET_SWITCH_REALB_TYPES(a_type, ctx, "add.out", CTYPE, [&]() {
        ET_SWITCH_REALB_TYPES(b_type, ctx, "add.out", CTYPE_B, [&]() {
          CTYPE alpha_val;
          ET_KERNEL_CHECK(
              ctx,
              torch::executor::native::utils::extract_scalar(alpha, &alpha_val),
              InvalidArgument, );
          CTYPE_B b_val = *b.const_data_ptr<CTYPE_B>();
          CTYPE b_casted = static_cast<CTYPE>(b_val);

          using Vec = at::vec::Vectorized<CTYPE>;
          at::vec::map<CTYPE>(
              [alpha_val, b_casted](Vec x) {
                return x + Vec(alpha_val * b_casted);
              },
              out.mutable_data_ptr<CTYPE>(),
              a.const_data_ptr<CTYPE>(),
              out.numel());
        });
      });
      return out;
    }
  } else if (a.numel() == 1) {
    return opt_add_out(ctx, b, a, alpha, out);
  }

  static constexpr const char op_name[] = "add.out";
  return torch::executor::kernels::impl::opt_add_sub_out_impl<false, op_name>(
      ctx, a, b, alpha, out);
}

Tensor& opt_add_scalar_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    const Scalar& alpha,
    Tensor& out) {
  (void)ctx;

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = utils::get_scalar_dtype(b);
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
    ET_SWITCH_REALB_TYPES(a_type, ctx, "add.Scalar_out", CTYPE, [&]() {
      ET_SWITCH_SCALAR_OBJ_TYPES(b_type, ctx, "add.Scalar_out", CTYPE_B, [&]() {
        CTYPE_B b_val;
        ET_EXTRACT_SCALAR(b, b_val);
        CTYPE b_casted = static_cast<CTYPE>(b_val);
        CTYPE alpha_val;
        ET_EXTRACT_SCALAR(alpha, alpha_val);

        using Vec = at::vec::Vectorized<CTYPE>;
        at::vec::map<CTYPE>(
            [alpha_val, b_casted](Vec x) {
              return x + Vec(alpha_val * b_casted);
            },
            out.mutable_data_ptr<CTYPE>(),
            a.const_data_ptr<CTYPE>(),
            out.numel());
      });
    });
  } else {
    ET_SWITCH_REALHBBF16_TYPES(a_type, ctx, "add.Scalar_out", CTYPE_A, [&]() {
      ET_SWITCH_SCALAR_OBJ_TYPES(b_type, ctx, "add.Scalar_out", CTYPE_B, [&]() {
        ET_SWITCH_REALB_TYPES(
            common_type, ctx, "add.Scalar_out", CTYPE_IN, [&]() {
              ET_SWITCH_REALHBBF16_TYPES(
                  out_type, ctx, "add.Scalar_out", CTYPE_OUT, [&]() {
                    CTYPE_B b_val;
                    ET_EXTRACT_SCALAR(b, b_val);
                    CTYPE_IN b_casted = static_cast<CTYPE_IN>(b_val);
                    CTYPE_IN alpha_val;
                    ET_EXTRACT_SCALAR(alpha, alpha_val);

                    const size_t n = a.numel();
                    const CTYPE_A* a_data = a.const_data_ptr<CTYPE_A>();
                    CTYPE_OUT* out_data = out.mutable_data_ptr<CTYPE_OUT>();
                    for (auto i = 0; i < n; ++i) {
                      out_data[i] = static_cast<CTYPE_OUT>(
                          static_cast<CTYPE_IN>(a_data[i]) +
                          alpha_val * b_casted);
                    }
                  });
            });
      });
    });
  }

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
