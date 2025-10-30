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
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>

namespace executorch {
namespace ET_RUNTIME_NAMESPACE {

/**
 * Describes the layout of a tensor.
 */
class TensorLayout final {
 public:
  TensorLayout() = delete;

  /**
   * Creates a TensorLayout from the given parameters.
   *
   * @param[in] sizes The sizes of the tensor. Note: the span passed here must
   * outlive the TensorLayout and all copies of it.
   * @param[in] dim_order The dim order of the tensor. Note: the span passed
   * here must outlive the TensorLayout and all copies of it.
   * @param[in] scalar_type The scalar type of the tensor.
   * @return A Result containing the TensorLayout on success, or an error.
   */
  static executorch::runtime::Result<const TensorLayout> create(
      Span<const int32_t> sizes,
      Span<const uint8_t> dim_order,
      executorch::aten::ScalarType scalar_type);

  /**
   * Returns the sizes of the tensor.
   *
   * NOTE: The TensorLayout must outlive the spans returned here.
   */
  Span<const int32_t> sizes() const {
    return sizes_;
  }

  /**
   * Returns the dim order of the tensor.
   *
   * NOTE: The TensorLayout must outlive the spans returned here.
   */
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
  TensorLayout(
      Span<const int32_t> sizes,
      Span<const uint8_t> dim_order,
      executorch::aten::ScalarType scalar_type,
      size_t nbytes)
      : sizes_(sizes),
        dim_order_(dim_order),
        scalar_type_(scalar_type),
        nbytes_(nbytes) {}
  /// The sizes of the tensor.
  const Span<const int32_t> sizes_;

  /// The dim order of the tensor.
  const Span<const uint8_t> dim_order_;

  /// The scalar type of the tensor.
  const executorch::aten::ScalarType scalar_type_;

  /// The size in bytes of the tensor.
  const size_t nbytes_;
};

} // namespace ET_RUNTIME_NAMESPACE
} // namespace executorch
