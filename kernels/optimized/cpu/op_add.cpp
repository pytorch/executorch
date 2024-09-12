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
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

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
struct AddInner;

template <
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct AddInner<true, CTYPE_A, CTYPE_B, CTYPE_IN, CTYPE_OUT> {
  static void
  run(const Tensor& a, const Tensor& b, CTYPE_IN alpha_val, Tensor& out) {
    apply_binary_elementwise_fn<CTYPE_A, CTYPE_B, CTYPE_OUT>(
        // NOLINTNEXTLINE(facebook-hte-ConstantArgumentPassByValue)
        [alpha_val](const CTYPE_A val_a, const CTYPE_B val_b) {
          CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
          CTYPE_IN b_casted = static_cast<CTYPE_IN>(val_b);
          CTYPE_IN value = a_casted + alpha_val * b_casted;

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
struct AddInner<false, CTYPE_A, CTYPE_B, CTYPE_IN, CTYPE_OUT>
    : public ReportCanCastBug<CTYPE_IN> {};

} // namespace

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

Tensor& opt_add_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    const Scalar& alpha,
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

      ET_SWITCH_REALB_TYPES(a_type, ctx, "add.out", CTYPE, [&]() {
        ET_SWITCH_REALB_TYPES(b_type, ctx, "add.out", CTYPE_B, [&]() {
          CTYPE alpha_val;
          ET_KERNEL_CHECK(
              ctx, utils::extract_scalar(alpha, &alpha_val), InvalidArgument, );
          CTYPE_B b_val = *b.const_data_ptr<CTYPE_B>();
          CTYPE b_casted = static_cast<CTYPE>(b_val);

          using Vec = executorch::vec::Vectorized<CTYPE>;
          executorch::vec::map<CTYPE>(
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

    ET_SWITCH_REALB_TYPES(a_type, ctx, "add.out", CTYPE, [&]() {
      CTYPE alpha_val;
      ET_KERNEL_CHECK(
          ctx, utils::extract_scalar(alpha, &alpha_val), InvalidArgument, );

      using Vec = executorch::vec::Vectorized<CTYPE>;
      executorch::vec::map2<CTYPE>(
          [alpha_val](Vec x, Vec y) { return x + Vec(alpha_val) * y; },
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
      // Catch failure to update logic when adding new broadcasting possibility.
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
    ET_SWITCH_REALB_TYPES(out_type, ctx, "add.out", CTYPE, [&]() {
      CTYPE alpha_val;
      ET_KERNEL_CHECK(
          ctx, utils::extract_scalar(alpha, &alpha_val), InvalidArgument, );

      using Vec = executorch::vec::Vectorized<CTYPE>;
      executorch::vec::broadcasting_map_2d_by_1d<CTYPE>(
          [alpha_val](Vec x, Vec y) { return x + Vec(alpha_val) * y; },
          out.mutable_data_ptr<CTYPE>(),
          lhs->const_data_ptr<CTYPE>(),
          rhs->const_data_ptr<CTYPE>(),
          lhs->sizes()[lhs->dim() - 2],
          lhs->sizes()[lhs->dim() - 1]);
    });
  } else {
    ScalarType common_type =
        promoteTypes(a_type, b_type, /*half_to_float*/ true);
    ET_KERNEL_CHECK(ctx, canCast(common_type, out_type), InvalidArgument, out);

    ET_KERNEL_CHECK(
        ctx,
        resize_to_broadcast_target_size(a, b, out) == Error::Ok,
        InvalidArgument,
        out);

    ET_SWITCH_REALHBBF16_TYPES(a_type, ctx, "add.out", CTYPE_A, [&]() {
      ET_SWITCH_REALHBBF16_TYPES(b_type, ctx, "add.out", CTYPE_B, [&]() {
        using CTYPE_IN = typename torch::executor::
            promote_types<CTYPE_A, CTYPE_B, /*half_to_float*/ true>::type;
        ET_DCHECK(CppTypeToScalarType<CTYPE_IN>::value == common_type);
        ET_SWITCH_REALHBBF16_TYPES(out_type, ctx, "add.out", CTYPE_OUT, [&]() {
          CTYPE_IN alpha_val;
          ET_KERNEL_CHECK(
              ctx, utils::extract_scalar(alpha, &alpha_val), InvalidArgument, );

          AddInner<
              can_cast<CTYPE_IN, CTYPE_OUT>::value,
              CTYPE_A,
              CTYPE_B,
              CTYPE_IN,
              CTYPE_OUT>::run(a, b, alpha_val, out);
        });
      });
    });
  }

  return out;
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

        using Vec = executorch::vec::Vectorized<CTYPE>;
        executorch::vec::map<CTYPE>(
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
