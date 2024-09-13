/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

namespace {
template <
    bool can_cast,
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct PowInner;

template <
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct PowInner<true, CTYPE_A, CTYPE_B, CTYPE_IN, CTYPE_OUT> {
  static void run(const Tensor& a, const Tensor& b, Tensor& out) {
    apply_binary_elementwise_fn<CTYPE_A, CTYPE_B, CTYPE_OUT>(
        // NOLINTNEXTLINE(facebook-hte-ConstantArgumentPassByValue)
        [](const CTYPE_A val_a, const CTYPE_B val_b) {
          CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
          CTYPE_IN b_casted = static_cast<CTYPE_IN>(val_b);
          CTYPE_IN value = std::pow(a_casted, b_casted);
          return static_cast<CTYPE_OUT>(value);
        },
        a,
        b,
        out);
  }
};

struct ReportCanCastBug {
  static void run(const Tensor&, const Tensor&, Tensor&) {
    ET_DCHECK_MSG(false, "BUG: canCast should have been checked above");
  }
};

template <
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct PowInner<false, CTYPE_A, CTYPE_B, CTYPE_IN, CTYPE_OUT>
    : public ReportCanCastBug {};

} // namespace

Tensor& pow_Tensor_Tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  // Determine output size and resize for dynamic shapes
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType common_type = promoteTypes(a_type, b_type, /*half_to_float*/ true);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(
      ctx, common_type != exec_aten::ScalarType::Bool, InvalidArgument, out);
  ET_KERNEL_CHECK(ctx, canCast(common_type, out_type), InvalidArgument, out);

  ET_SWITCH_REALHB_TYPES(a_type, ctx, "pow.Tensor_Tensor_out", CTYPE_A, [&]() {
    ET_SWITCH_REALHB_TYPES(
        b_type, ctx, "pow.Tensor_Tensor_out", CTYPE_B, [&]() {
          using CTYPE_IN = typename torch::executor::
              promote_types<CTYPE_A, CTYPE_B, /*half_to_float*/ true>::type;
          ET_DCHECK(CppTypeToScalarType<CTYPE_IN>::value == common_type);
          ET_SWITCH_REALH_TYPES(
              out_type, ctx, "pow.Tensor_Tensor_out", CTYPE_OUT, [&]() {
                PowInner<
                    !std::is_same<CTYPE_IN, bool>::value &&
                        can_cast<CTYPE_IN, CTYPE_OUT>::value,
                    CTYPE_A,
                    CTYPE_B,
                    CTYPE_IN,
                    CTYPE_OUT>::run(a, b, out);
              });
        });
  });

  return out;
}

Tensor& pow_Tensor_Scalar_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, a.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = utils::get_scalar_dtype(b);
  ScalarType common_type =
      utils::promote_type_with_scalar(a_type, b, /*half_to_float*/ false);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, common_type == out_type, InvalidArgument, out);

  if (common_type == ScalarType::Half) {
    common_type = ScalarType::Float;
  }

  ET_SWITCH_REALHB_TYPES(a_type, ctx, "pow.Tensor_Scalar_out", CTYPE_A, [&]() {
    ET_SWITCH_SCALAR_OBJ_TYPES(
        b_type, ctx, "pow.Tensor_Scalar_out", CTYPE_B, [&]() {
          ET_SWITCH_REAL_TYPES(
              common_type, ctx, "pow.Tensor_Scalar_out", CTYPE_IN, [&]() {
                ET_SWITCH_REALH_TYPES(
                    out_type, ctx, "pow.Tensor_Scalar_out", CTYPE_OUT, [&]() {
                      CTYPE_B val_b = 0;
                      utils::extract_scalar(b, &val_b);
                      apply_unary_map_fn(
                          [val_b](const CTYPE_A val_a) {
                            CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
                            CTYPE_IN b_casted = static_cast<CTYPE_IN>(val_b);
                            CTYPE_IN value = std::pow(a_casted, b_casted);

                            return static_cast<CTYPE_OUT>(value);
                          },
                          a.const_data_ptr<CTYPE_A>(),
                          out.mutable_data_ptr<CTYPE_OUT>(),
                          out.numel());
                    });
              });
        });
  });

  return out;
}

Tensor& pow_Scalar_out(
    KernelRuntimeContext& ctx,
    const Scalar& a,
    const Tensor& b,
    Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, b.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ScalarType a_type = utils::get_scalar_dtype(a);
  ScalarType b_type = b.scalar_type();
  ScalarType common_type =
      utils::promote_type_with_scalar(b_type, a, /*half_to_float*/ false);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, common_type == out_type, InvalidArgument, out);

  if (common_type == ScalarType::Half) {
    common_type = ScalarType::Float;
  }

  ET_SWITCH_SCALAR_OBJ_TYPES(a_type, ctx, "pow.Scalar_out", CTYPE_A, [&]() {
    ET_SWITCH_REALHB_TYPES(b_type, ctx, "pow.Scalar_out", CTYPE_B, [&]() {
      ET_SWITCH_REAL_TYPES(common_type, ctx, "pow.Scalar_out", CTYPE_IN, [&]() {
        ET_SWITCH_REALH_TYPES(
            out_type, ctx, "pow.Scalar_out", CTYPE_OUT, [&]() {
              CTYPE_A val_a = 0;
              utils::extract_scalar(a, &val_a);

              apply_unary_map_fn(
                  [val_a](const CTYPE_B val_b) {
                    CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
                    CTYPE_IN b_casted = static_cast<CTYPE_IN>(val_b);
                    CTYPE_IN value = std::pow(a_casted, b_casted);
                    return static_cast<CTYPE_OUT>(value);
                  },
                  b.const_data_ptr<CTYPE_B>(),
                  out.mutable_data_ptr<CTYPE_OUT>(),
                  out.numel());
            });
      });
    });
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
