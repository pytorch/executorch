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
#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
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
  ScalarType common_type = promoteTypes(a_type, b_type);

  ET_KERNEL_CHECK(
      ctx,
      (canCast(common_type, out_type) &&
       check_alpha_type(utils::get_scalar_dtype(alpha), common_type)),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(a, b, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "add.out";

  if (b.numel() == 1) {
    if (executorch::runtime::isComplexType(a_type) ||
        executorch::runtime::isComplexType(b_type) ||
        executorch::runtime::isComplexType(out_type)) {
      // TODO: The current support for complex dtype enforces that input and
      // output tensors have the same dtype. Support mixed dtypes in the future.
      ET_KERNEL_CHECK(
          ctx, a_type == b_type && a_type == out_type, InvalidArgument, out);

      ET_SWITCH_COMPLEXH_TYPES(out_type, ctx, op_name, CTYPE, [&]() {
        CTYPE alpha_val = utils::scalar_to<CTYPE>(alpha);
        CTYPE b_val = *b.const_data_ptr<CTYPE>();

        using Vec = at::vec::Vectorized<CTYPE>;
        at::vec::map<CTYPE>(
            [alpha_val, b_val](Vec x) { return x + Vec(alpha_val * b_val); },
            out.mutable_data_ptr<CTYPE>(),
            a.const_data_ptr<CTYPE>(),
            out.numel());
      });
      return out;
    } else if (
        a_type == b_type && a_type == out_type && a_type != ScalarType::Half &&
        a_type != ScalarType::BFloat16) {
      ET_SWITCH_REALB_TYPES(a_type, ctx, op_name, CTYPE, [&]() {
        ET_SWITCH_REALB_TYPES(b_type, ctx, op_name, CTYPE_B, [&]() {
          CTYPE alpha_val;
          ET_KERNEL_CHECK(
              ctx, utils::extract_scalar(alpha, &alpha_val), InvalidArgument, );
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

  return torch::executor::kernels::impl::opt_add_sub_out_impl<false, op_name>(
      ctx, a, b, alpha, out);
}

Tensor& opt_add_scalar_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    const Scalar& alpha,
    Tensor& out) {
  ScalarType a_type = a.scalar_type();
  ScalarType common_type = utils::promote_type_with_scalar(a_type, b);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(
      ctx,
      (common_type == a_type &&
       check_alpha_type(utils::get_scalar_dtype(alpha), common_type)),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(a, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, a.sizes()) == Error::Ok, InvalidArgument, out);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "add.Scalar_out";

  if (a_type == common_type && a_type == out_type &&
      a_type != ScalarType::Half && a_type != ScalarType::BFloat16) {
    ET_SWITCH_REALB_TYPES(a_type, ctx, op_name, CTYPE, [&]() {
      CTYPE b_casted = utils::scalar_to<CTYPE>(b);
      CTYPE alpha_val;
      ET_KERNEL_CHECK(
          ctx, utils::extract_scalar(alpha, &alpha_val), InvalidArgument, );

      using Vec = at::vec::Vectorized<CTYPE>;
      at::vec::map<CTYPE>(
          [alpha_val, b_casted](Vec x) {
            return x + Vec(alpha_val * b_casted);
          },
          out.mutable_data_ptr<CTYPE>(),
          a.const_data_ptr<CTYPE>(),
          out.numel());
    });
  } else {
    ScalarType compute_type = utils::internal::get_compute_type(common_type);

    ET_SWITCH_REALB_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
      CTYPE_COMPUTE val_b = utils::scalar_to<CTYPE_COMPUTE>(b);
      CTYPE_COMPUTE val_alpha;
      ET_KERNEL_CHECK(
          ctx, utils::extract_scalar(alpha, &val_alpha), InvalidArgument, );
      auto val_alpha_times_b = val_alpha * val_b;
      utils::apply_unitensor_elementwise_fn<
          CTYPE_COMPUTE,
          op_name,
          utils::SupportedTensorDtypes::SAME_AS_COMMON>(
          [val_alpha_times_b](const auto val_a) {
            // Cast here supports vectorization; either it does nothing
            // or it casts from CTYPE_COMPUTE to
            // Vectorized<CTYPE_COMPUTE>.
            return val_a + decltype(val_a)(val_alpha_times_b);
          },
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
