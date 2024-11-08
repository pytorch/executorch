/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/optimized/cpu/binary_ops.h>
#include <executorch/kernels/optimized/vec/functional.h>
#include <executorch/kernels/optimized/vec/vec.h>
#include <executorch/kernels/portable/cpu/op_div_impl.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

Tensor& opt_div_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  (void)ctx;

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType out_type = out.scalar_type();

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
      ET_SWITCH_REALB_TYPES(tensor_type, ctx, "div.out", CTYPE, [&]() {
        ET_SWITCH_REALB_TYPES(scalar_type, ctx, "div.out", CTYPE_SCALAR, [&]() {
          CTYPE_SCALAR scalar_val = *scalar->const_data_ptr<CTYPE_SCALAR>();
          CTYPE scalar_casted = static_cast<CTYPE>(scalar_val);

          using Vec = executorch::vec::Vectorized<CTYPE>;
          if (a.numel() == 1) {
            executorch::vec::map<CTYPE>(
                [scalar_casted](Vec x) { return Vec(scalar_casted) / x; },
                out.mutable_data_ptr<CTYPE>(),
                tensor->const_data_ptr<CTYPE>(),
                out.numel());
          } else {
            Vec inv_scalar_casted_vec(CTYPE(1) / scalar_casted);
            executorch::vec::map<CTYPE>(
                [inv_scalar_casted_vec](Vec x) {
                  return x * inv_scalar_casted_vec;
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

    ET_SWITCH_REAL_TYPES_AND(Bool, out_type, ctx, "div.out", CTYPE, [&]() {
      using Vec = executorch::vec::Vectorized<CTYPE>;
      executorch::vec::map2<CTYPE>(
          [](Vec x, Vec y) { return x / y; },
          out.mutable_data_ptr<CTYPE>(),
          a.const_data_ptr<CTYPE>(),
          b.const_data_ptr<CTYPE>(),
          out.numel());
    });
  } else if (selected_optimized_path != ElementwiseOptimizedPath::kNone) {
    const Tensor* lhs;
    const Tensor* rhs;
    if (selected_optimized_path ==
        ElementwiseOptimizedPath::kBroadcast2dBy1dReverseArguments) {
      lhs = &b;
      rhs = &a;
    } else {
      // Catch failure to update logic when subing new broadcasting possibility.
      ET_DCHECK(
          selected_optimized_path ==
          ElementwiseOptimizedPath::kBroadcast2dBy1d);
      lhs = &a;
      rhs = &b;
    }
    auto error = resize_tensor(out, lhs->sizes());
    ET_KERNEL_CHECK_MSG(
        ctx,
        error == Error::Ok,
        InvalidArgument,
        out,
        "Failed to resize output tensor.");
    ET_SWITCH_REALB_TYPES(out_type, ctx, "sub.out", CTYPE, [&]() {
      using Vec = executorch::vec::Vectorized<CTYPE>;
      if (selected_optimized_path ==
          ElementwiseOptimizedPath::kBroadcast2dBy1dReverseArguments) {
        executorch::vec::broadcasting_map_2d_by_1d<CTYPE>(
            [](Vec x, Vec y) { return y / x; },
            out.mutable_data_ptr<CTYPE>(),
            lhs->const_data_ptr<CTYPE>(),
            rhs->const_data_ptr<CTYPE>(),
            lhs->sizes()[lhs->dim() - 2],
            lhs->sizes()[lhs->dim() - 1]);
      } else {
        executorch::vec::broadcasting_map_2d_by_1d<CTYPE>(
            [](Vec x, Vec y) { return x / y; },
            out.mutable_data_ptr<CTYPE>(),
            lhs->const_data_ptr<CTYPE>(),
            rhs->const_data_ptr<CTYPE>(),
            lhs->sizes()[lhs->dim() - 2],
            lhs->sizes()[lhs->dim() - 1]);
      }
    });
  } else {
    div_out_impl(ctx, a, b, out);
  }

  return out;
}

Tensor& opt_div_scalar_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    Tensor& out) {
  (void)ctx;

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = utils::get_scalar_dtype(b);
  ScalarType common_type = isFloatingType(a_type) ? a_type : ScalarType::Float;
  ScalarType out_type = out.scalar_type();

  ET_CHECK(common_type == out_type);

  // Resize for dynamic shape
  auto error = resize_tensor(out, a.sizes());
  ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");

  if (a_type == common_type && a_type == out_type) {
    ET_SWITCH_REAL_TYPES(a_type, ctx, "div.Scalar_out", CTYPE, [&]() {
      ET_SWITCH_REAL_TYPES_AND(
          Bool, b_type, ctx, "div.Scalar_out", CTYPE_B, [&]() {
            CTYPE_B b_val;
            ET_EXTRACT_SCALAR(b, b_val);
            CTYPE b_casted = static_cast<CTYPE>(b_val);

            using Vec = executorch::vec::Vectorized<CTYPE>;
            Vec inv_b_casted_vec(CTYPE(1) / b_casted);
            executorch::vec::map<CTYPE>(
                [inv_b_casted_vec](Vec x) { return x * inv_b_casted_vec; },
                out.mutable_data_ptr<CTYPE>(),
                a.const_data_ptr<CTYPE>(),
                out.numel());
          });
    });
  } else {
    div_scalar_out_impl(ctx, a, b, out);
  }

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
