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

using namespace utils;

using Scalar = exec_aten::Scalar;
using ScalarType = exec_aten::ScalarType;
using Tensor = exec_aten::Tensor;

Tensor& clamp_out(
    RuntimeContext& ctx,
    const Tensor& in,
    const exec_aten::optional<Scalar>& min_opt,
    const exec_aten::optional<Scalar>& max_opt,
    Tensor& out) {
  (void)ctx;

  Error err = resize_tensor(out, in.sizes());
  ET_CHECK_MSG(err == Error::Ok, "Could not resize output");

  ET_CHECK_SAME_SHAPE_AND_DTYPE2(in, out);

  ET_SWITCH_REAL_TYPES(in.scalar_type(), ctx, "clamp", CTYPE, [&]() {
    // Extract optional min value
    CTYPE min = 0;
    bool has_min = min_opt.has_value();
    if (has_min) {
      bool ok = utils::extract_scalar<CTYPE>(min_opt.value(), &min);
      ET_CHECK_MSG(ok, "Invalid min value: wrong type or out of range");
    }
    // Extract optional max value
    CTYPE max = 0;
    bool has_max = max_opt.has_value();
    if (has_max) {
      bool ok = utils::extract_scalar<CTYPE>(max_opt.value(), &max);
      ET_CHECK_MSG(ok, "Invalid max value: wrong type or out of range");
    }

    apply_unary_map_fn(
        [has_min, min, has_max, max](const CTYPE val_in) {
          CTYPE val_out = val_in;
          if (has_min) {
            val_out = max_override(val_out, min);
          }
          if (has_max) {
            val_out = min_override(val_out, max);
          }
          return val_out;
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
