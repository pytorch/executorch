/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/core/tensor_layout.h>

#include <limits>

namespace executorch {
namespace ET_RUNTIME_NAMESPACE {

namespace {
Result<size_t> calculate_nbytes(
    const Span<const int32_t>& sizes,
    const executorch::aten::ScalarType& scalar_type) {
  ssize_t n = 1;
  for (const auto i : c10::irange(sizes.size())) {
    if (sizes[i] < 0) {
      return Error::InvalidArgument;
    }
    if (n > std::numeric_limits<ssize_t>::max() /
                std::max(sizes[i], (int32_t)1)) {
      return Error::InvalidArgument;
    }
    n *= sizes[i];
  }
  // Use the full namespace to disambiguate from c10::elementSize.
  auto elem_size =
      static_cast<ssize_t>(executorch::runtime::elementSize(scalar_type));
  if (n > 0 && elem_size > 0 &&
      static_cast<size_t>(n) >
          std::numeric_limits<size_t>::max() / static_cast<size_t>(elem_size)) {
    return Error::InvalidArgument;
  }
  return static_cast<size_t>(n) * static_cast<size_t>(elem_size);
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

  for (const auto i : c10::irange(dim_order.size())) {
    if (dim_order[i] >= sizes.size()) {
      return Error::InvalidArgument;
    }
  }
  return TensorLayout(sizes, dim_order, scalar_type, nbytes.get());
}
} // namespace ET_RUNTIME_NAMESPACE
} // namespace executorch
