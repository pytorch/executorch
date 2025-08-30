/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/kernels/portable/cpu/util/math_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Scalar = executorch::aten::Scalar;
using ScalarType = executorch::aten::ScalarType;
using Tensor = executorch::aten::Tensor;

namespace {

template <typename CTYPE_OUT, typename CTYPE_CAST>
/** Check if val, when cast to CTYPE_CAST, is not in the range of CTYPE_OUT */
bool is_out_of_bounds(CTYPE_CAST val_cast) {
  return val_cast < std::numeric_limits<CTYPE_OUT>::lowest() ||
      val_cast > std::numeric_limits<CTYPE_OUT>::max();
}

ET_NODISCARD bool check_bounds(
    KernelRuntimeContext& ctx,
    const Scalar& val_scalar,
    const torch::executor::native::ScalarType& val_type,
    const torch::executor::native::ScalarType& out_type,
    const char* val_name) {
  auto is_valid = true;

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "clamp.out";

  if (isIntegralType(out_type, /*includeBool=*/false)) {
    const long val_long = utils::scalar_to<long>(val_scalar);
    ET_SWITCH_INT_TYPES(out_type, ctx, op_name, CTYPE_OUT, [&]() {
      if (is_out_of_bounds<CTYPE_OUT, long>(val_long)) {
        ET_LOG(Error, "%s value out of bounds", val_name);
        is_valid = false;
      }
    });
  } else if (isFloatingType(out_type)) {
    ET_SWITCH_FLOATHBF16_TYPES(out_type, ctx, op_name, CTYPE_OUT, [&]() {
      const double val_double = utils::scalar_to<double>(val_scalar);
      if (std::isfinite(val_double) &&
          is_out_of_bounds<CTYPE_OUT, double>(val_double)) {
        ET_LOG(Error, "%s value out of bounds", val_name);
        is_valid = false;
      }
    });
  }

  return is_valid;
}

} // namespace

Tensor& clamp_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const std::optional<Scalar>& min_opt,
    const std::optional<Scalar>& max_opt,
    Tensor& out) {
  bool has_min = min_opt.has_value();
  bool has_max = max_opt.has_value();

  ET_KERNEL_CHECK_MSG(
      ctx,
      has_min || has_max,
      InvalidArgument,
      out,
      "At least one of 'min' or 'max' must not be None");

  // Input Dtypes
  ScalarType in_type = in.scalar_type();
  ScalarType min_type =
      has_min ? utils::get_scalar_dtype(min_opt.value()) : in_type;
  ScalarType max_type =
      has_max ? utils::get_scalar_dtype(max_opt.value()) : in_type;
  ScalarType out_type = out.scalar_type();

  // Common Dtype
  ScalarType common_type = in_type;
  if (has_min) {
    common_type = utils::promote_type_with_scalar(common_type, min_opt.value());
  }
  if (has_max) {
    common_type = utils::promote_type_with_scalar(common_type, max_opt.value());
  }

  // Check Common Dtype
  ET_KERNEL_CHECK(ctx, common_type == out_type, InvalidArgument, out);

  // Check Scalar Bounds
  if (has_min) {
    ET_KERNEL_CHECK(
        ctx,
        check_bounds(ctx, min_opt.value(), min_type, out_type, "minimum"),
        InvalidArgument,
        out);
  }
  if (has_max) {
    ET_KERNEL_CHECK(
        ctx,
        check_bounds(ctx, max_opt.value(), max_type, out_type, "maximum"),
        InvalidArgument,
        out);
  }

  // Check Dim Order
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  // Resize
  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, in.sizes()) == Error::Ok, InvalidArgument, out);

  // Compute Dtype
  ScalarType compute_type = utils::get_compute_type(common_type);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "clamp.out";

  ET_SWITCH_REALB_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
    utils::apply_unitensor_elementwise_fn<
        CTYPE_COMPUTE,
        op_name,
        utils::SupportedTensorDtypes::SAME_AS_COMMON>(
        [has_min, min_opt, has_max, max_opt](const auto val_in) {
          auto val_out = val_in;
          if (has_min) {
            val_out = utils::max_override(
                val_out, utils::scalar_to<CTYPE_COMPUTE>(min_opt.value()));
          }
          if (has_max) {
            val_out = utils::min_override(
                val_out, utils::scalar_to<CTYPE_COMPUTE>(max_opt.value()));
          }
          return val_out;
        },
        ctx,
        in,
        utils::SupportedTensorDtypes::REALHBBF16,
        out);
  });

  return out;
}

Tensor& clamp_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const std::optional<Tensor>& min_opt,
    const std::optional<Tensor>& max_opt,
    Tensor& out) {
  bool has_min = min_opt.has_value();
  bool has_max = max_opt.has_value();

  ET_KERNEL_CHECK_MSG(
      ctx,
      has_min || has_max,
      InvalidArgument,
      out,
      "At least one of 'min' or 'max' must not be None");

  const Tensor& min = has_min ? min_opt.value() : in;
  const Tensor& max = has_max ? max_opt.value() : in;

  // Common Dtype
  ScalarType common_type = in.scalar_type();
  if (has_min) {
    common_type = promoteTypes(common_type, min.scalar_type());
  }
  if (has_max) {
    common_type = promoteTypes(common_type, max.scalar_type());
  }

  // Check Common Dtype
  ET_KERNEL_CHECK(
      ctx, canCast(common_type, out.scalar_type()), InvalidArgument, out);

  // Check Dim Order
  ET_KERNEL_CHECK(
      ctx,
      tensors_have_same_dim_order(in, min, max, out),
      InvalidArgument,
      out);

  // Resize
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(in, min, max, out) == Error::Ok,
      InvalidArgument,
      out);

  // Compute Dtype
  ScalarType compute_type = utils::get_compute_type(common_type);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "clamp.Tensor_out";

  ET_SWITCH_REALB_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
    utils::apply_tritensor_elementwise_fn<
        CTYPE_COMPUTE,
        op_name,
        utils::SupportedTensorDtypes::REALHBBF16>(
        [has_min, has_max](
            const CTYPE_COMPUTE val_in,
            const CTYPE_COMPUTE val_min,
            const CTYPE_COMPUTE val_max) {
          // TODO: rewrite this to be vectorization-capable.
          CTYPE_COMPUTE val_out = val_in;
          if (has_min) {
            val_out = utils::max_override(val_out, val_min);
          }
          if (has_max) {
            val_out = utils::min_override(val_out, val_max);
          }
          return val_out;
        },
        ctx,
        in,
        utils::SupportedTensorDtypes::REALHBBF16,
        min,
        utils::SupportedTensorDtypes::REALHBBF16,
        max,
        utils::SupportedTensorDtypes::REALHBBF16,
        out);
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
