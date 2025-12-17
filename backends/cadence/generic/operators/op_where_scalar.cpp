/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_where_scalar.h>

namespace impl {
namespace generic {
namespace native {

::executorch::aten::Tensor& where_Scalar_out(
    ET_UNUSED ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& condition,
    const double val1,
    const double val2,
    ::executorch::aten::Tensor& out) {
  const float val1_f = static_cast<float>(val1);
  const float val2_f = static_cast<float>(val2);
  for (int i = 0; i < out.numel(); ++i) {
    out.mutable_data_ptr<float>()[i] =
        condition.const_data_ptr<bool>()[i] ? val1_f : val2_f;
  }

  return out;
}

} // namespace native
} // namespace generic
} // namespace impl
