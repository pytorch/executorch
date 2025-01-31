/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/core/tensor_layout.h>

namespace executorch {
namespace runtime {

namespace {
Result<size_t> calculate_nbytes(
    const Span<const int32_t>& sizes,
    const executorch::aten::ScalarType& scalar_type) {
  ssize_t n = 1;
  for (ssize_t i = 0; i < sizes.size(); i++) {
    if (sizes[i] < 0) {
      return Error::InvalidArgument;
    }
    n *= sizes[i];
  }
  // Use the full namespace to disambiguate from c10::elementSize.
  return n * executorch::runtime::elementSize(scalar_type);
}
} // namespace

Result<const TensorLayout> TensorLayout::create(
    Span<const int32_t> sizes,
    Span<const uint8_t> dim_order,
    executorch::aten::ScalarType scalar_type) {
  auto nbytes = calculate_nbytes(sizes, scalar_type);
  if (!nbytes.ok()) {
    return nbytes.error();
  }

  if (dim_order.size() != sizes.size()) {
    return Error::InvalidArgument;
  }

  for (size_t i = 0; i < dim_order.size(); i++) {
    if (dim_order[i] >= sizes.size()) {
      return Error::InvalidArgument;
    }
  }
  return TensorLayout(sizes, dim_order, scalar_type, nbytes.get());
}
} // namespace runtime
} // namespace executorch
