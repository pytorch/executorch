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
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

namespace {

ScalarType get_common_type(ScalarType a_type, ScalarType b_type) {
  ET_CHECK(
      !isComplexType(a_type) && !isQIntType(a_type) && !isBitsType(a_type));
  ET_CHECK(
      !isComplexType(b_type) && !isQIntType(b_type) && !isBitsType(b_type));

  if (isFloatingType(a_type) && isFloatingType(b_type)) {
    return promoteTypes(a_type, b_type);
  } else if (isFloatingType(a_type)) {
    return a_type;
  } else if (isFloatingType(b_type)) {
    return b_type;
  }
  return ScalarType::Float;
}

} // namespace

Tensor& opt_div_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  // Check Dim Order
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(a, b, out), InvalidArgument, out);

  // Resize
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "div.out";

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType out_type = out.scalar_type();

  if (a.numel() == 1 || b.numel() == 1) {
    if (a_type == b_type && a_type == out_type && a_type != ScalarType::Half &&
        a_type != ScalarType::BFloat16) {
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
      ET_SWITCH_REALB_TYPES(tensor_type, ctx, op_name, CTYPE, [&]() {
        ET_SWITCH_REALB_TYPES(scalar_type, ctx, op_name, CTYPE_SCALAR, [&]() {
          CTYPE_SCALAR scalar_val = *scalar->const_data_ptr<CTYPE_SCALAR>();
          CTYPE scalar_casted = static_cast<CTYPE>(scalar_val);

          using Vec = at::vec::Vectorized<CTYPE>;
          if (a.numel() == 1) {
            at::vec::map<CTYPE>(
                [scalar_casted](Vec x) { return Vec(scalar_casted) / x; },
                out.mutable_data_ptr<CTYPE>(),
                tensor->const_data_ptr<CTYPE>(),
                out.numel());
          } else {
            Vec inv_scalar_casted_vec(CTYPE(1) / scalar_casted);
            at::vec::map<CTYPE>(
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
    ET_SWITCH_REALB_TYPES(out_type, ctx, op_name, CTYPE, [&]() {
      using Vec = at::vec::Vectorized<CTYPE>;
      at::vec::map2<CTYPE>(
          [](Vec x, Vec y) { return x / y; },
          out.mutable_data_ptr<CTYPE>(),
          a.const_data_ptr<CTYPE>(),
          b.const_data_ptr<CTYPE>(),
          out.numel());
    });
  } else if (selected_optimized_path != ElementwiseOptimizedPath::kNone) {
    // Reason for using alpha is becasuse handle_broadcast_elementwise
    // is used for add and sub as well:
    ET_SWITCH_REALB_TYPES(out_type, ctx, op_name, CTYPE, [&]() {
      if (selected_optimized_path ==
              ElementwiseOptimizedPath::kBroadcast2dBy1dReverseArguments ||
          selected_optimized_path ==
              ElementwiseOptimizedPath::kBroadcastLastDimReverseArguments ||
          selected_optimized_path ==
              ElementwiseOptimizedPath::kBroadcastNdByNdReverseArguments) {
        auto div_lambda = [](auto x, auto y) { return y / x; };
        torch::executor::handle_broadcast_elementwise<CTYPE>(
            ctx, div_lambda, a, b, out, selected_optimized_path);
      } else {
        auto div_lambda = [](auto x, auto y) { return x / y; };
        torch::executor::handle_broadcast_elementwise<CTYPE>(
            ctx, div_lambda, a, b, out, selected_optimized_path);
      }
    });
  } else {
    ScalarType common_type = get_common_type(a.scalar_type(), b.scalar_type());
    ScalarType compute_type = utils::get_compute_type(common_type);

    ET_SWITCH_FLOAT_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
      utils::apply_bitensor_elementwise_fn<
          CTYPE_COMPUTE,
          op_name,
          utils::SupportedTensorDtypes::FLOATHBF16>(
          [](const auto val_a, const auto val_b) { return val_a / val_b; },
          ctx,
          a,
          utils::SupportedTensorDtypes::REALHBBF16,
          b,
          utils::SupportedTensorDtypes::REALHBBF16,
          out);
    });
  }

  return out;
}

Tensor& opt_div_scalar_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    Tensor& out) {
  ScalarType a_type = a.scalar_type();
  ScalarType b_type = utils::get_scalar_dtype(b);
  ScalarType common_type = isFloatingType(a_type) ? a_type : ScalarType::Float;
  ScalarType out_type = out.scalar_type();

  // Check Common Dtype
  ET_KERNEL_CHECK(ctx, common_type == out_type, InvalidArgument, out);

  // Check Dim Order
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(a, out), InvalidArgument, out);

  // Resize
  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, a.sizes()) == Error::Ok, InvalidArgument, out);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "div.Scalar_out";

  if (a_type == common_type && a_type == out_type &&
      a_type != ScalarType::Half && a_type != ScalarType::BFloat16) {
    ET_SWITCH_REAL_TYPES(a_type, ctx, op_name, CTYPE, [&]() {
      ET_SWITCH_REALB_TYPES(b_type, ctx, op_name, CTYPE_B, [&]() {
        CTYPE_B b_val;
        ET_EXTRACT_SCALAR(b, b_val);
        CTYPE b_casted = static_cast<CTYPE>(b_val);

        using Vec = at::vec::Vectorized<CTYPE>;
        Vec inv_b_casted_vec(CTYPE(1) / b_casted);
        at::vec::map<CTYPE>(
            [inv_b_casted_vec](Vec x) { return x * inv_b_casted_vec; },
            out.mutable_data_ptr<CTYPE>(),
            a.const_data_ptr<CTYPE>(),
            out.numel());
      });
    });
  } else {
    ScalarType compute_type = utils::get_compute_type(common_type);

    ET_SWITCH_FLOAT_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
      const CTYPE_COMPUTE val_b = utils::scalar_to<CTYPE_COMPUTE>(b);
      utils::apply_unitensor_elementwise_fn<
          CTYPE_COMPUTE,
          op_name,
          utils::SupportedTensorDtypes::SAME_AS_COMMON>(
          [val_b](const auto val_a) { return val_a / val_b; },
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
