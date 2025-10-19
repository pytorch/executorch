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
#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
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
  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType out_type = out.scalar_type();
  ScalarType common_type = promoteTypes(a_type, b_type);

  ET_KERNEL_CHECK(ctx, canCast(common_type, out_type), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(a, b, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "mul.out";

  if (b.numel() == 1) {
    if (a_type == b_type && a_type == out_type && a_type != ScalarType::Half &&
        a_type != ScalarType::BFloat16) {
      ET_SWITCH_REALB_TYPES(a_type, ctx, op_name, CTYPE, [&]() {
        ET_SWITCH_REALB_TYPES(b_type, ctx, op_name, CTYPE_B, [&]() {
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
    if (executorch::runtime::isComplexType(out_type)) {
      ET_KERNEL_CHECK(
          ctx, a_type == b_type && a_type == out_type, InvalidArgument, out);

      ET_SWITCH_COMPLEXH_TYPES(out_type, ctx, op_name, CTYPE, [&]() {
        using Vec = at::vec::Vectorized<CTYPE>;
        at::vec::map2<CTYPE>(
            [](Vec x, Vec y) { return x * y; },
            out.mutable_data_ptr<CTYPE>(),
            a.const_data_ptr<CTYPE>(),
            b.const_data_ptr<CTYPE>(),
            out.numel());
      });
    } else {
      ET_SWITCH_REALB_TYPES(out_type, ctx, op_name, CTYPE, [&]() {
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

      ET_SWITCH_COMPLEXH_TYPES(out_type, ctx, op_name, CTYPE, [&]() {
        auto mul_lambda = [](auto x, auto y) { return x * y; };
        torch::executor::handle_broadcast_elementwise<CTYPE>(
            ctx, mul_lambda, a, b, out, selected_optimized_path);
      });
    } else {
      ET_SWITCH_REALB_TYPES(out_type, ctx, op_name, CTYPE, [&]() {
        auto mul_lambda = [](auto x, auto y) { return x * y; };
        torch::executor::handle_broadcast_elementwise<CTYPE>(
            ctx, mul_lambda, a, b, out, selected_optimized_path);
      });
    }
  } else {
    if (executorch::runtime::isComplexType(a_type) ||
        executorch::runtime::isComplexType(b_type) ||
        executorch::runtime::isComplexType(out_type)) {
      ET_KERNEL_CHECK(
          ctx, a_type == b_type && a_type == out_type, InvalidArgument, out);

      ET_SWITCH_COMPLEXH_TYPES(out_type, ctx, op_name, CTYPE, [&]() {
        apply_binary_elementwise_fn<CTYPE, CTYPE, CTYPE>(
            [](const CTYPE val_a, const CTYPE val_b) { return val_a * val_b; },
            a,
            b,
            out);
      });
    } else {
      ScalarType compute_type = utils::internal::get_compute_type(common_type);

      ET_SWITCH_REALB_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
        utils::apply_bitensor_elementwise_fn<
            CTYPE_COMPUTE,
            op_name,
            utils::SupportedTensorDtypes::REALHBBF16>(
            [](const auto val_a, const auto val_b) { return val_a * val_b; },
            ctx,
            a,
            utils::SupportedTensorDtypes::REALHBBF16,
            b,
            utils::SupportedTensorDtypes::REALHBBF16,
            out);
      });
    }
  }

  return out;
}

Tensor& opt_mul_scalar_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    Tensor& out) {
  ScalarType a_type = a.scalar_type();
  ScalarType common_type = utils::promote_type_with_scalar(a_type, b);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, common_type == out_type, InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(a, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, a.sizes()) == Error::Ok, InvalidArgument, out);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "mul.Scalar_out";

  if (a_type == common_type && a_type == out_type &&
      a_type != ScalarType::Half && a_type != ScalarType::BFloat16) {
    ET_SWITCH_REALB_TYPES(a_type, ctx, op_name, CTYPE, [&]() {
      CTYPE b_casted = utils::scalar_to<CTYPE>(b);

      using Vec = at::vec::Vectorized<CTYPE>;
      at::vec::map<CTYPE>(
          [b_casted](Vec x) { return x * Vec(b_casted); },
          out.mutable_data_ptr<CTYPE>(),
          a.const_data_ptr<CTYPE>(),
          out.numel());
    });
  } else {
    ScalarType compute_type = utils::internal::get_compute_type(common_type);

    ET_SWITCH_REALB_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
      const CTYPE_COMPUTE val_b = utils::scalar_to<CTYPE_COMPUTE>(b);
      utils::apply_unitensor_elementwise_fn<
          CTYPE_COMPUTE,
          op_name,
          utils::SupportedTensorDtypes::SAME_AS_COMMON>(
          [val_b](const auto val_a) { return val_a * val_b; },
          ctx,
          a,
          utils::SupportedTensorDtypes::REALHBBF16,
          out);
    });
  }

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
