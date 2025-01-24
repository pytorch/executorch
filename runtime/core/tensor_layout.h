/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/span.h>

namespace executorch {
namespace runtime {

namespace {
size_t calculate_nbytes(
    const Span<const int32_t>& sizes,
    const exec_aten::ScalarType& scalar_type) {
  ssize_t n = 1;
  for (ssize_t i = 0; i < sizes.size(); i++) {
    ET_CHECK(sizes[i] >= 0);
    n *= sizes[i];
  }
  // Use the full namespace to disambiguate from c10::elementSize.
  return n * executorch::runtime::elementSize(scalar_type);
}
} // namespace

/**
 * Metadata describing the layout of external tensors (tensors that are not
 stored in the PTE file).
 *
 * The NamedDataMap used to create the TensorLayout must outlive the
 TensorLayout.
 */
class TensorLayout {
 public:
  TensorLayout(
      executorch::aten::ScalarType scalar_type,
      Span<const int32_t> sizes,
      Span<const uint8_t> dim_order)
      : sizes_(sizes),
        dim_order_(dim_order),
        scalar_type_(scalar_type),
        nbytes_(calculate_nbytes(sizes_, scalar_type_)) {}

  TensorLayout(const TensorLayout&) = default;
  TensorLayout(TensorLayout&&) = default;
  TensorLayout& operator=(const TensorLayout&) = default;
  TensorLayout& operator=(TensorLayout&& other) = default;
  ~TensorLayout() = default;

  /// Returns the sizes of the tensor.
  Span<const int32_t> sizes() const {
    return sizes_;
  }

  /// Returns the dim order of the tensor.
  Span<const uint8_t> dim_order() const {
    return dim_order_;
  }

  /// Returns the scalar type of the tensor.
  executorch::aten::ScalarType scalar_type() const {
    return scalar_type_;
  }

  /// Returns the size of the tensor in bytes.
  size_t nbytes() const {
    return nbytes_;
  }

 private:
  /// The sizes of the tensor.
  Span<const int32_t> sizes_;

  /// The dim order of the tensor.
  Span<const uint8_t> dim_order_;

  /// The scalar type of the tensor.
  executorch::aten::ScalarType scalar_type_;

  /// The size in bytes of the tensor.
  size_t nbytes_;
};

} // namespace runtime
} // namespace executorch
