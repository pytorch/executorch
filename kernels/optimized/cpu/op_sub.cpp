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
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

#include <executorch/kernels/optimized/cpu/op_add_sub_impl.h>

namespace torch {
namespace executor {
namespace native {
namespace {

template <
    bool can_cast,
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct SubInner;

template <
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct SubInner<true, CTYPE_A, CTYPE_B, CTYPE_IN, CTYPE_OUT> {
  static void
  run(const Tensor& a, const Tensor& b, CTYPE_IN alpha_val, Tensor& out) {
    apply_binary_elementwise_fn<CTYPE_A, CTYPE_B, CTYPE_OUT>(
        // NOLINTNEXTLINE(facebook-hte-ConstantArgumentPassByValue)
        [alpha_val](const CTYPE_A val_a, const CTYPE_B val_b) {
          CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
          CTYPE_IN b_casted = static_cast<CTYPE_IN>(val_b);
          CTYPE_IN value = a_casted - alpha_val * b_casted;

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
struct SubInner<false, CTYPE_A, CTYPE_B, CTYPE_IN, CTYPE_OUT>
    : public ReportCanCastBug<CTYPE_IN> {};

} // namespace

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;

Tensor& opt_sub_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    const Scalar& alpha,
    Tensor& out) {
  (void)ctx;

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, tensor_is_realh_type(out), InvalidArgument, out);
  if (a.numel() == 1 || b.numel() == 1) {
    if (a_type == b_type && a_type == out_type && a_type != ScalarType::Half) {
      const Tensor* tensor;
      const Tensor* scalar;
      ScalarType tensor_type;
      ScalarType scalar_type;
      if (a.numel() == 1) {
        tensor = &b;
        tensor_type = b_type;
        scalar = &a;
        scalar_type = a_type;
      } else {
        tensor = &a;
        tensor_type = a_type;
        scalar = &b;
        scalar_type = b_type;
      }
      ET_KERNEL_CHECK(
          ctx,
          resize_to_broadcast_target_size(a, b, out) == Error::Ok,
          InvalidArgument,
          out);
      ET_SWITCH_REAL_TYPES(tensor_type, ctx, "sub.out", CTYPE, [&]() {
        ET_SWITCH_REAL_TYPES(scalar_type, ctx, "sub.out", CTYPE_SCALAR, [&]() {
          CTYPE alpha_val;
          ET_KERNEL_CHECK(
              ctx, utils::extract_scalar(alpha, &alpha_val), InvalidArgument, );
          CTYPE_SCALAR scalar_val = *scalar->const_data_ptr<CTYPE_SCALAR>();
          CTYPE scalar_casted = static_cast<CTYPE>(scalar_val);

          using Vec = at::vec::Vectorized<CTYPE>;
          if (a.numel() == 1) {
            at::vec::map<CTYPE>(
                [alpha_val, scalar_casted](Vec x) {
                  return Vec(scalar_casted) - Vec(alpha_val) * x;
                },
                out.mutable_data_ptr<CTYPE>(),
                tensor->const_data_ptr<CTYPE>(),
                out.numel());
          } else {
            at::vec::map<CTYPE>(
                [alpha_val, scalar_casted](Vec x) {
                  return x - Vec(alpha_val * scalar_casted);
                },
                out.mutable_data_ptr<CTYPE>(),
                tensor->const_data_ptr<CTYPE>(),
                out.numel());
          }
        });
      });
      return out;
    }
  }

  static constexpr const char op_name[] = "sub.out";
  return torch::executor::kernels::impl::opt_add_sub_out_impl<true, op_name>(
      ctx, a, b, alpha, out);
}

Tensor& opt_sub_scalar_out(
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

  if (common_type == ScalarType::Half) {
    common_type = ScalarType::Float;
  }

  // Resize for dynamic shape
  auto error = resize_tensor(out, a.sizes());
  ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");

  if (a_type == common_type && a_type == out_type &&
      a_type != ScalarType::Half) {
    ET_SWITCH_REAL_TYPES(a_type, ctx, "sub.Scalar_out", CTYPE, [&]() {
      ET_SWITCH_SCALAR_OBJ_REAL_TYPES(
          b_type, ctx, "sub.Scalar_out", CTYPE_B, [&]() {
            CTYPE_B b_val;
            ET_EXTRACT_SCALAR(b, b_val);
            CTYPE b_casted = static_cast<CTYPE>(b_val);
            CTYPE alpha_val;
            ET_EXTRACT_SCALAR(alpha, alpha_val);

            using Vec = at::vec::Vectorized<CTYPE>;
            at::vec::map<CTYPE>(
                [alpha_val, b_casted](Vec x) {
                  return x - Vec(alpha_val * b_casted);
                },
                out.mutable_data_ptr<CTYPE>(),
                a.const_data_ptr<CTYPE>(),
                out.numel());
          });
    });
  } else {
    ET_SWITCH_REALH_TYPES(a_type, ctx, "sub.Scalar_out", CTYPE_A, [&]() {
      ET_SWITCH_SCALAR_OBJ_REAL_TYPES(
          b_type, ctx, "sub.Scalar_out", CTYPE_B, [&]() {
            ET_SWITCH_REAL_TYPES(
                common_type, ctx, "sub.Scalar_out", CTYPE_IN, [&]() {
                  ET_SWITCH_REALH_TYPES(
                      out_type, ctx, "sub.Scalar_out", CTYPE_OUT, [&]() {
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
                              static_cast<CTYPE_IN>(a_data[i]) -
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
