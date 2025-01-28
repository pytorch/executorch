/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/optimized/vec/functional.h>
#include <executorch/kernels/optimized/vec/vec.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

Tensor& opt_le_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(ctx, tensors_have_same_shape(a, b), InvalidArgument, out);

  // Resize for dynamic shape
  auto error = resize_tensor(out, a.sizes());
  ET_KERNEL_CHECK_MSG(
      ctx,
      error == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType out_type = out.scalar_type();

  if (a_type == b_type && a_type == out_type) {
    ET_SWITCH_REAL_TYPES_AND(
        Bool, out_type, ctx, "le.Tensor_out", CTYPE, [&]() {
          using Vec = executorch::vec::Vectorized<CTYPE>;
          executorch::vec::map2<CTYPE>(
              [](Vec x, Vec y) { return x.le(y); },
              out.mutable_data_ptr<CTYPE>(),
              a.const_data_ptr<CTYPE>(),
              b.const_data_ptr<CTYPE>(),
              a.numel());
        });
  } else {
    ET_SWITCH_REAL_TYPES_AND(
        Bool, a_type, ctx, "le.Tensor_out", CTYPE_A, [&]() {
          ET_SWITCH_REAL_TYPES_AND(
              Bool, b_type, ctx, "le.Tensor_out", CTYPE_B, [&]() {
                using CTYPE_IN = typename torch::executor::
                    promote_types<CTYPE_A, CTYPE_B>::type;
                ET_DCHECK(
                    CppTypeToScalarType<CTYPE_IN>::value ==
                    promoteTypes(a_type, b_type));
                ET_SWITCH_REAL_TYPES_AND(
                    Bool, out_type, ctx, "le.Tensor_out", CTYPE_OUT, [&]() {
                      const size_t n = a.numel();
                      const CTYPE_A* a_data = a.const_data_ptr<CTYPE_A>();
                      const CTYPE_B* b_data = b.const_data_ptr<CTYPE_B>();
                      CTYPE_OUT* out_data = out.mutable_data_ptr<CTYPE_OUT>();
                      for (auto i = 0; i < n; ++i) {
                        out_data[i] = static_cast<CTYPE_OUT>(
                            static_cast<CTYPE_IN>(a_data[i]) <=
                            static_cast<CTYPE_IN>(b_data[i]));
                      }
                    });
              });
        });
  }

  return out;
}

Tensor& opt_le_scalar_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  auto error = resize_tensor(out, a.sizes());
  ET_KERNEL_CHECK_MSG(
      ctx,
      error == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = utils::get_scalar_dtype(b);
  ScalarType common_type = promoteTypes(a_type, b_type);
  ScalarType out_type = out.scalar_type();

  if (a_type == common_type && a_type == out_type) {
    ET_SWITCH_REAL_TYPES_AND(Bool, a_type, ctx, "le.Scalar_out", CTYPE, [&]() {
      ET_SWITCH_REAL_TYPES_AND(
          Bool, b_type, ctx, "le.Scalar_out", CTYPE_B, [&]() {
            CTYPE_B b_val = 0;
            ET_EXTRACT_SCALAR(b, b_val);
            CTYPE b_casted = static_cast<CTYPE>(b_val);
            using Vec = executorch::vec::Vectorized<CTYPE>;
            executorch::vec::map<CTYPE>(
                [b_casted](Vec x) { return x.le(Vec(b_casted)); },
                out.mutable_data_ptr<CTYPE>(),
                a.const_data_ptr<CTYPE>(),
                a.numel());
          });
    });
  } else {
    ET_SWITCH_REAL_TYPES_AND(
        Bool, a_type, ctx, "le.Scalar_out", CTYPE_A, [&]() {
          ET_SWITCH_REAL_TYPES_AND(
              Bool, b_type, ctx, "le.Scalar_out", CTYPE_B, [&]() {
                ET_SWITCH_REAL_TYPES_AND(
                    Bool, common_type, ctx, "le.Scalar_out", CTYPE_IN, [&]() {
                      ET_SWITCH_REAL_TYPES_AND(
                          Bool,
                          out_type,
                          ctx,
                          "le.Scalar_out",
                          CTYPE_OUT,
                          [&]() {
                            CTYPE_B b_val = 0;
                            ET_EXTRACT_SCALAR(b, b_val);
                            CTYPE_IN b_casted = static_cast<CTYPE_IN>(b_val);
                            const size_t n = a.numel();
                            const CTYPE_A* a_data = a.const_data_ptr<CTYPE_A>();
                            CTYPE_OUT* out_data =
                                out.mutable_data_ptr<CTYPE_OUT>();
                            for (auto i = 0; i < n; ++i) {
                              out_data[i] = static_cast<CTYPE_OUT>(
                                  static_cast<CTYPE_IN>(a_data[i]) <= b_casted);
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
