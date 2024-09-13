/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/math_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>
#include <cmath>
#include <type_traits>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

namespace {
template <
    bool can_cast,
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct FloorDivideInner;

template <
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct FloorDivideInner<true, CTYPE_A, CTYPE_B, CTYPE_IN, CTYPE_OUT> {
  static void
  run(const Tensor& a, const Tensor& b, Tensor& out, bool& div_by_zero_error) {
    apply_binary_elementwise_fn<CTYPE_A, CTYPE_B, CTYPE_OUT>(
        // NOLINTNEXTLINE(facebook-hte-ConstantArgumentPassByValue)
        [&div_by_zero_error](const CTYPE_A val_a, const CTYPE_B val_b) {
          if (is_integral_type<CTYPE_IN, /*includeBool=*/true>::value) {
            if (val_b == 0) {
              div_by_zero_error = true;
              return static_cast<CTYPE_OUT>(0);
            }
          }
          CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
          CTYPE_IN b_casted = static_cast<CTYPE_IN>(val_b);
          CTYPE_IN value = utils::floor_divide<CTYPE_IN>(a_casted, b_casted);

          return static_cast<CTYPE_OUT>(value);
        },
        a,
        b,
        out);
  }
};

struct ReportCanCastBug {
  static void run(const Tensor&, const Tensor&, Tensor&, bool&) {
    ET_DCHECK_MSG(false, "BUG: canCast should have been checked above");
  }
};

template <
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct FloorDivideInner<false, CTYPE_A, CTYPE_B, CTYPE_IN, CTYPE_OUT>
    : public ReportCanCastBug {};

} // namespace

Tensor& floor_divide_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(ctx, tensor_is_real_type(out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(a, b, out), InvalidArgument, out);

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType common_type = promoteTypes(a_type, b_type);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, canCast(common_type, out_type), InvalidArgument, out);

  auto div_by_zero_error = false;

  ET_SWITCH_REAL_TYPES_AND(
      Bool, a_type, ctx, "floor_divide.out", CTYPE_A, [&]() {
        ET_SWITCH_REAL_TYPES_AND(
            Bool, b_type, ctx, "floor_divide.out", CTYPE_B, [&]() {
              using CTYPE_IN = typename torch::executor::
                  promote_types<CTYPE_A, CTYPE_B>::type;
              ET_DCHECK(CppTypeToScalarType<CTYPE_IN>::value == common_type);
              ET_SWITCH_REAL_TYPES(
                  out_type, ctx, "floor_divide.out", CTYPE_OUT, [&]() {
                    FloorDivideInner<
                        can_cast<CTYPE_IN, CTYPE_OUT>::value,
                        CTYPE_A,
                        CTYPE_B,
                        CTYPE_IN,
                        CTYPE_OUT>::run(a, b, out, div_by_zero_error);
                  });
            });
      });

  ET_KERNEL_CHECK_MSG(
      ctx,
      !div_by_zero_error,
      InvalidArgument,
      out,
      "Floor divide operation encountered integer division by zero");

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
