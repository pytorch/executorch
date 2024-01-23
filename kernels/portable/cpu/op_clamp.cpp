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
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

namespace {

//
// Override min/max so we can emulate PyTorch's behavior with NaN entries.
//

template <
    typename FLOAT_T,
    typename std::enable_if<std::is_floating_point<FLOAT_T>::value, bool>::
        type = true>
FLOAT_T min_override(FLOAT_T a, FLOAT_T b) {
  if (std::isnan(a)) {
    return a;
  } else if (std::isnan(b)) {
    return b;
  } else {
    return std::min(a, b);
  }
}

template <
    typename FLOAT_T,
    typename std::enable_if<std::is_floating_point<FLOAT_T>::value, bool>::
        type = true>
FLOAT_T max_override(FLOAT_T a, FLOAT_T b) {
  if (std::isnan(a)) {
    return a;
  } else if (std::isnan(b)) {
    return b;
  } else {
    return std::max(a, b);
  }
}

template <
    typename INT_T,
    typename std::enable_if<std::is_integral<INT_T>::value, bool>::type = true>
INT_T min_override(INT_T a, INT_T b) {
  return std::min(a, b);
}

template <
    typename INT_T,
    typename std::enable_if<std::is_integral<INT_T>::value, bool>::type = true>
INT_T max_override(INT_T a, INT_T b) {
  return std::max(a, b);
}

} // namespace

using namespace utils;

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

[[nodiscard]] bool check_bounds(
    const Scalar& val_scalar,
    const torch::executor::native::ScalarType& val_type,
    const torch::executor::native::ScalarType& out_type,
    const char* val_name) {
  auto is_valid = true;

  ET_SWITCH_SCALAR_OBJ_TYPES(val_type, ctx, "clamp.out", CTYPE_VAL, [&]() {
    CTYPE_VAL val = 0;
    ET_EXTRACT_SCALAR(val_scalar, val);
    if (isIntegralType(out_type, /*includeBool=*/false)) {
      ET_SWITCH_INT_TYPES(out_type, ctx, "clamp.out", CTYPE_OUT, [&]() {
        if (is_out_of_bounds<CTYPE_VAL, CTYPE_OUT, long>(val)) {
          ET_LOG(Error, "%s value out of bounds", val_name);
          is_valid = false;
        }
      });
    } else if (isFloatingType(out_type)) {
      ET_SWITCH_FLOAT_TYPES(out_type, ctx, "clamp", CTYPE_OUT, [&]() {
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
    RuntimeContext& ctx,
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

  ET_SWITCH_REAL_TYPES(out_type, ctx, "clamp", CTYPE_OUT, [&]() {
    // Extract optional min value
    CTYPE_OUT min = 0;
    if (has_min) {
      ET_SWITCH_SCALAR_OBJ_TYPES(min_type, ctx, "clamp", CTYPE_MIN, [&]() {
        CTYPE_MIN min_val = 0;
        ET_EXTRACT_SCALAR(min_opt.value(), min_val);
        min = static_cast<CTYPE_OUT>(min_val);
      });
    }

    // Extract optional max value
    CTYPE_OUT max = 0;
    if (has_max) {
      ET_SWITCH_SCALAR_OBJ_TYPES(max_type, ctx, "clamp", CTYPE_MAX, [&]() {
        CTYPE_MAX max_val = 0;
        ET_EXTRACT_SCALAR(max_opt.value(), max_val);
        max = static_cast<CTYPE_OUT>(max_val);
      });
    }

    ET_SWITCH_REAL_TYPES_AND(Bool, in_type, ctx, "clamp", CTYPE_IN, [&]() {
      apply_unary_map_fn(
          [has_min, min, has_max, max](const CTYPE_IN val_in) {
            CTYPE_OUT val_out = static_cast<CTYPE_OUT>(val_in);
            if (has_min) {
              val_out = max_override(val_out, min);
            }
            if (has_max) {
              val_out = min_override(val_out, max);
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

} // namespace native
} // namespace executor
} // namespace torch
