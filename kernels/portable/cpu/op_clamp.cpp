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
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/kernels/portable/cpu/util/math_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Scalar = exec_aten::Scalar;
using ScalarType = exec_aten::ScalarType;
using Tensor = exec_aten::Tensor;

namespace {

template <typename CTYPE_VAL, typename CTYPE_OUT, typename CTYPE_CAST>
/** Check if val, when cast to CTYPE_CAST, is not in the range of CTYPE_OUT */
bool is_out_of_bounds(CTYPE_VAL val) {
  const CTYPE_CAST val_cast = static_cast<CTYPE_CAST>(val);
  return val_cast < std::numeric_limits<CTYPE_OUT>::lowest() ||
      val_cast > std::numeric_limits<CTYPE_OUT>::max();
}

ET_NODISCARD bool check_bounds(
    const Scalar& val_scalar,
    const torch::executor::native::ScalarType& val_type,
    const torch::executor::native::ScalarType& out_type,
    const char* val_name) {
  auto is_valid = true;

  ET_SWITCH_SCALAR_OBJ_TYPES(val_type, ctx, "clamp.out", CTYPE_VAL, [&]() {
    CTYPE_VAL val = 0;
    utils::extract_scalar(val_scalar, &val);
    if (isIntegralType(out_type, /*includeBool=*/false)) {
      ET_SWITCH_INT_TYPES(out_type, ctx, "clamp.out", CTYPE_OUT, [&]() {
        if (is_out_of_bounds<CTYPE_VAL, CTYPE_OUT, long>(val)) {
          ET_LOG(Error, "%s value out of bounds", val_name);
          is_valid = false;
        }
      });
    } else if (isFloatingType(out_type)) {
      ET_SWITCH_FLOATH_TYPES(out_type, ctx, "clamp", CTYPE_OUT, [&]() {
        if (std::isfinite(val) &&
            is_out_of_bounds<CTYPE_VAL, CTYPE_OUT, double>(val)) {
          ET_LOG(Error, "%s value out of bounds", val_name);
          is_valid = false;
        }
      });
    }
  });

  return is_valid;
}

} // namespace

Tensor& clamp_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const exec_aten::optional<Scalar>& min_opt,
    const exec_aten::optional<Scalar>& max_opt,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, in.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ScalarType in_type = in.scalar_type();
  ScalarType min_type = in_type;
  ScalarType max_type = in_type;
  ScalarType common_type = in_type;
  ScalarType out_type = out.scalar_type();

  bool has_min = min_opt.has_value();
  if (has_min) {
    min_type = utils::get_scalar_dtype(min_opt.value());
    common_type = utils::promote_type_with_scalar(common_type, min_opt.value());
    ET_KERNEL_CHECK(
        ctx,
        check_bounds(min_opt.value(), min_type, out_type, "minimum"),
        InvalidArgument,
        out);
  }
  bool has_max = max_opt.has_value();
  if (has_max) {
    max_type = utils::get_scalar_dtype(max_opt.value());
    common_type = utils::promote_type_with_scalar(common_type, max_opt.value());
    ET_KERNEL_CHECK(
        ctx,
        check_bounds(max_opt.value(), max_type, out_type, "maximum"),
        InvalidArgument,
        out);
  }

  ET_KERNEL_CHECK_MSG(
      ctx,
      has_min || has_max,
      InvalidArgument,
      out,
      "At least one of 'min' or 'max' must not be None");

  ET_KERNEL_CHECK(ctx, common_type == out_type, InvalidArgument, out);

  ET_SWITCH_REALH_TYPES(out_type, ctx, "clamp", CTYPE_OUT, [&]() {
    // Extract optional min value
    CTYPE_OUT min = 0;
    if (has_min) {
      ET_SWITCH_SCALAR_OBJ_TYPES(min_type, ctx, "clamp", CTYPE_MIN, [&]() {
        CTYPE_MIN min_val = 0;
        utils::extract_scalar(min_opt.value(), &min_val);
        min = static_cast<CTYPE_OUT>(min_val);
      });
    }

    // Extract optional max value
    CTYPE_OUT max = 0;
    if (has_max) {
      ET_SWITCH_SCALAR_OBJ_TYPES(max_type, ctx, "clamp", CTYPE_MAX, [&]() {
        CTYPE_MAX max_val = 0;
        utils::extract_scalar(max_opt.value(), &max_val);
        max = static_cast<CTYPE_OUT>(max_val);
      });
    }

    ET_SWITCH_REALHB_TYPES(in_type, ctx, "clamp", CTYPE_IN, [&]() {
      apply_unary_map_fn(
          [has_min, min, has_max, max](const CTYPE_IN val_in) {
            CTYPE_OUT val_out = static_cast<CTYPE_OUT>(val_in);
            if (has_min) {
              val_out = utils::max_override(val_out, min);
            }
            if (has_max) {
              val_out = utils::min_override(val_out, max);
            }
            return val_out;
          },
          in.const_data_ptr<CTYPE_IN>(),
          out.mutable_data_ptr<CTYPE_OUT>(),
          in.numel());
    });
  });

  return out;
}

Tensor& clamp_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const exec_aten::optional<Tensor>& min_opt,
    const exec_aten::optional<Tensor>& max_opt,
    Tensor& out) {
  (void)ctx;

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

  ET_KERNEL_CHECK(
      ctx,
      tensors_have_same_dim_order(in, min, max, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(in, min, max, out) == Error::Ok,
      InvalidArgument,
      out);

  ScalarType in_type = in.scalar_type();
  ScalarType min_type = min.scalar_type();
  ScalarType max_type = max.scalar_type();
  ScalarType common_type = in_type;
  ScalarType out_type = out.scalar_type();

  if (has_min) {
    common_type = promoteTypes(common_type, min_type, /*half_to_float*/ true);
  }
  if (has_max) {
    common_type = promoteTypes(common_type, max_type, /*half_to_float*/ true);
  }

  ET_KERNEL_CHECK(ctx, canCast(common_type, out_type), InvalidArgument, out);

  constexpr auto name = "clamp.Tensor_out";

  ET_SWITCH_REALHB_TYPES(in_type, ctx, name, CTYPE_IN, [&]() {
    ET_SWITCH_REALHB_TYPES(min_type, ctx, name, CTYPE_MIN, [&]() {
      ET_SWITCH_REALHB_TYPES(max_type, ctx, name, CTYPE_MAX, [&]() {
        ET_SWITCH_REALHB_TYPES(out_type, ctx, name, CTYPE_OUT, [&]() {
          using CTYPE_MINMAX = typename torch::executor::
              promote_types<CTYPE_MIN, CTYPE_MAX>::type;
          using CTYPE = typename torch::executor::
              promote_types<CTYPE_IN, CTYPE_MINMAX>::type;
          apply_ternary_elementwise_fn<
              CTYPE_IN,
              CTYPE_MIN,
              CTYPE_MAX,
              CTYPE_OUT>(
              [has_min, has_max](
                  const CTYPE_IN val_in,
                  const CTYPE_MIN val_min,
                  const CTYPE_MAX val_max) {
                CTYPE val_out = static_cast<CTYPE>(val_in);
                if (has_min) {
                  val_out =
                      utils::max_override(val_out, static_cast<CTYPE>(val_min));
                }
                if (has_max) {
                  val_out =
                      utils::min_override(val_out, static_cast<CTYPE>(val_max));
                }
                return static_cast<CTYPE_OUT>(val_out);
              },
              in,
              min,
              max,
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
