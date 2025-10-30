/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/dtype_util.h>

namespace torch {
namespace executor {
namespace native {
namespace utils {
namespace internal {

bool check_tensor_dtype(
    const Tensor t,
    SupportedTensorDtypes dtypes,
    const ScalarType compute_type) {
  switch (dtypes) {
    case SupportedTensorDtypes::REALHBBF16:
      return executorch::runtime::tensor_is_realhbbf16_type(t);
    case SupportedTensorDtypes::REALHBF16:
      return executorch::runtime::tensor_is_realhbf16_type(t);
    case SupportedTensorDtypes::FLOATHBF16:
      return executorch::runtime::tensor_is_floating_type(t);
    case SupportedTensorDtypes::INTB:
      return executorch::runtime::tensor_is_integral_type(t, true);
    case SupportedTensorDtypes::BOOL:
      return executorch::runtime::tensor_is_type(t, ScalarType::Bool);
    case SupportedTensorDtypes::BOOL_OR_BYTE:
      return (executorch::runtime::tensor_is_type(
          t, ScalarType::Bool, ScalarType::Byte));
    case SupportedTensorDtypes::SAME_AS_COMPUTE:
      return executorch::runtime::tensor_is_type(t, compute_type);
    case SupportedTensorDtypes::SAME_AS_COMMON: {
      if (compute_type == ScalarType::Float) {
        return (executorch::runtime::tensor_is_type(
            t, ScalarType::Float, ScalarType::Half, ScalarType::BFloat16));
      } else {
        return executorch::runtime::tensor_is_type(t, compute_type);
      }
    }
  }
  ET_CHECK(false);
  return false;
}

} // namespace internal
} // namespace utils
} // namespace native
} // namespace executor
} // namespace torch
