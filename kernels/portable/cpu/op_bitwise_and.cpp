/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// patternlint-disable-next-line executorch-cpp-nostdinc
#include <functional>

#include <executorch/kernels/portable/cpu/pattern/bitwise_op.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

Tensor& bitwise_and_Tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(a, b, out), InvalidArgument, out);

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType common_type = promoteTypes(a_type, b_type);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, canCast(common_type, out_type), InvalidArgument, out);

  ET_SWITCH_INT_TYPES_AND(
      Bool, a_type, ctx, "bitwise_and.Tensor_out", CTYPE_A, [&]() {
        ET_SWITCH_INT_TYPES_AND(
            Bool, b_type, ctx, "bitwise_and.Tensor_out", CTYPE_B, [&]() {
              using CTYPE_IN = typename torch::executor::
                  promote_types<CTYPE_A, CTYPE_B>::type;
              ET_DCHECK(CppTypeToScalarType<CTYPE_IN>::value == common_type);
              ET_SWITCH_REAL_TYPES_AND(
                  Bool,
                  out_type,
                  ctx,
                  "bitwise_and.Tensor_out",
                  CTYPE_OUT,
                  [&]() {
                    internal::BitwiseOpInner<
                        can_cast<CTYPE_IN, CTYPE_OUT>::value,
                        std::bit_and,
                        CTYPE_A,
                        CTYPE_B,
                        CTYPE_IN,
                        CTYPE_OUT>::run(a, b, out);
                  });
            });
      });

  return out;
}

Tensor& bitwise_and_Scalar_out(
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

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(a, out), InvalidArgument, out);

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = utils::get_scalar_dtype(b);
  ScalarType common_type = utils::promote_type_with_scalar(a_type, b);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, canCast(common_type, out_type), InvalidArgument, out);

  ET_SWITCH_INT_TYPES_AND(
      Bool, a_type, ctx, "bitwise_and.Scalar_out", CTYPE_A, [&]() {
        ET_SWITCH_SCALAR_OBJ_INTB_TYPES(
            b_type, ctx, "bitwise_and.Scalar_out", CTYPE_B, [&]() {
              CTYPE_B val_b = 0;
              utils::extract_scalar(b, &val_b);
              ET_SWITCH_INT_TYPES_AND(
                  Bool,
                  common_type,
                  ctx,
                  "bitwise_and.Scalar_out",
                  CTYPE_IN,
                  [&]() {
                    ET_SWITCH_REAL_TYPES_AND(
                        Bool,
                        out_type,
                        ctx,
                        "bitwise_and.Scalar_out",
                        CTYPE_OUT,
                        [&]() {
                          apply_unary_map_fn(
                              [val_b](const CTYPE_A val_a) {
                                CTYPE_IN a_casted =
                                    static_cast<CTYPE_IN>(val_a);
                                CTYPE_IN b_casted =
                                    static_cast<CTYPE_IN>(val_b);
                                CTYPE_IN value = std::bit_and<CTYPE_IN>()(
                                    a_casted, b_casted);

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

} // namespace native
} // namespace executor
} // namespace torch
