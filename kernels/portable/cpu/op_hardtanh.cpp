/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

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
    std::enable_if_t<std::is_floating_point<FLOAT_T>::value, bool> = true>
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
    std::enable_if_t<std::is_floating_point<FLOAT_T>::value, bool> = true>
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
    std::enable_if_t<std::is_integral<INT_T>::value, bool> = true>
INT_T min_override(INT_T a, INT_T b) {
  return std::min(a, b);
}

template <
    typename INT_T,
    std::enable_if_t<std::is_integral<INT_T>::value, bool> = true>
INT_T max_override(INT_T a, INT_T b) {
  return std::max(a, b);
}

} // namespace

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

Tensor& hardtanh_out(
    RuntimeContext& ctx,
    const Tensor& in,
    const Scalar& min,
    const Scalar& max,
    Tensor& out) {
  (void)ctx;

  Error err = resize_tensor(out, in.sizes());
  ET_CHECK_MSG(err == Error::Ok, "Could not resize output");

  ScalarType in_type = in.scalar_type();
  ScalarType min_type = utils::get_scalar_dtype(min);
  ScalarType max_type = utils::get_scalar_dtype(max);
  ScalarType out_type = out.scalar_type();

  ET_CHECK(in_type == out_type);

  ET_SWITCH_REAL_TYPES(in_type, ctx, "hardtanh", CTYPE, [&]() {
    CTYPE min_casted;
    ET_SWITCH_SCALAR_OBJ_TYPES(min_type, ctx, "hardtanh", CTYPE_MIN, [&]() {
      CTYPE_MIN min_val;
      ET_EXTRACT_SCALAR(min, min_val);
      min_casted = static_cast<CTYPE>(min_val);
    });

    CTYPE max_casted;
    ET_SWITCH_SCALAR_OBJ_TYPES(max_type, ctx, "hardtanh", CTYPE_MAX, [&]() {
      CTYPE_MAX max_val;
      ET_EXTRACT_SCALAR(max, max_val);
      max_casted = static_cast<CTYPE>(max_val);
    });

    apply_unary_map_fn(
        [min_casted, max_casted](const CTYPE val_in) {
          return min_override(max_override(val_in, min_casted), max_casted);
        },
        in.const_data_ptr<CTYPE>(),
        out.mutable_data_ptr<CTYPE>(),
        in.numel());
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
