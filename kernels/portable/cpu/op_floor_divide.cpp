/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
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

/**
 * Python's __floordiv__ operator is more complicated than just floor(a / b).
 * It aims to maintain the property: a == (a // b) * b + remainder(a, b)
 * which can otherwise fail due to rounding errors in the remainder.
 * So, instead it is calculated as: a // b = (a - remainder(a, b)) / b
 * With some additional fix-ups added to the result.
 */
template <
    typename INT_T,
    typename std::enable_if<std::is_integral<INT_T>::value, bool>::type = true>
INT_T floor_divide(INT_T a, INT_T b) {
  const auto quot = a / b;
  if (std::signbit(a) == std::signbit(b)) {
    return quot;
  }
  const auto rem = a % b;
  return rem ? quot - 1 : quot;
}

template <
    typename FLOAT_T,
    typename std::enable_if<std::is_floating_point<FLOAT_T>::value, bool>::
        type = true>
FLOAT_T floor_divide(FLOAT_T a, FLOAT_T b) {
  if (b == 0) {
    return std::signbit(a) ? -INFINITY : INFINITY;
  }
  const auto mod = std::fmod(a, b);
  auto div = (a - mod) / b;
  if ((mod != 0) && std::signbit(b) != std::signbit(mod)) {
    return div - 1;
  }
  return div;
}

} // namespace

Tensor& floor_divide_out(
    RuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType common_type = promoteTypes(a_type, b_type);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, canCast(common_type, out_type), InvalidArgument, out);

  ET_SWITCH_REAL_TYPES_AND(
      Bool, a_type, ctx, "floor_divide.out", CTYPE_A, [&]() {
        ET_SWITCH_REAL_TYPES_AND(
            Bool, b_type, ctx, "floor_divide.out", CTYPE_B, [&]() {
              ET_SWITCH_REAL_TYPES(
                  common_type, ctx, "floor_divide.out", CTYPE_IN, [&]() {
                    ET_SWITCH_REAL_TYPES(
                        out_type, ctx, "floor_divide.out", CTYPE_OUT, [&]() {
                          apply_binary_elementwise_fn<
                              CTYPE_A,
                              CTYPE_B,
                              CTYPE_OUT>(
                              [common_type](
                                  const CTYPE_A val_a, const CTYPE_B val_b) {
                                if (isIntegralType(
                                        common_type, /*includeBool=*/true)) {
                                  ET_CHECK(val_b != 0);
                                }
                                CTYPE_IN a_casted =
                                    static_cast<CTYPE_IN>(val_a);
                                CTYPE_IN b_casted =
                                    static_cast<CTYPE_IN>(val_b);
                                CTYPE_IN value =
                                    floor_divide<CTYPE_IN>(a_casted, b_casted);

                                return static_cast<CTYPE_OUT>(value);
                              },
                              a,
                              b,
                              out);
                        });
                  });
            });
      });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
