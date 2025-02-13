/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <executorch/runtime/core/exec_aten/util/tensor_dimension_limit.h>
#include <executorch/runtime/core/span.h>

namespace executorch::runtime {

/**
 * Maximum size of a string returned by tensor_shape_to_c_string, for
 * stack allocation.
 */
constexpr size_t kTensorShapeStringSizeLimit = 1 + /* opening parenthesis */
    10 * kTensorDimensionLimit + /* maximum digits we will print; update
                                  * kMaximumPrintableTensorShapeElement
                                  * if changing */
    2 * kTensorDimensionLimit + /* comma and space after each item,
                                 * overwritten with closing paren and
                                 * NUL terminator for last element */
    1; /* padding for temporary NUL terminator for simplicity of implementation
        */

namespace internal {
constexpr size_t kMaximumPrintableTensorShapeElement =
    std::numeric_limits<int32_t>::max();
} // namespace internal

/**
 * Convert a shape to a NUL-terminated C string with limited size. If
 * elements of the shape are larger than
 * kMaximumPrintableTensorShapeElement, those elements will be
 * rendered as ERR instead.
 *
 * NOTE: There are two overloads of this function to support both ATen
 * tensors and ExecuTorch Tensors, which have different SizesType,
 * while also avoiding a dependency on exec_aten.h from this header
 * because that would cause a circular dependency.
 */
std::array<char, kTensorShapeStringSizeLimit> tensor_shape_to_c_string(
    executorch::runtime::Span<const std::int32_t> shape);

/**
 * Convert a shape to a NUL-terminated C string with limited size. If
 * elements of the shape are larger than
 * kMaximumPrintableTensorShapeElement, those elements will be
 * rendered as ERR instead.
 *
 * NOTE: There are two overloads of this function to support both ATen
 * tensors and ExecuTorch Tensors, which have different SizesType,
 * while also avoiding a dependency on exec_aten.h from this header
 * because that would cause a circular dependency.
 */
std::array<char, kTensorShapeStringSizeLimit> tensor_shape_to_c_string(
    executorch::runtime::Span<const std::int64_t> shape);

} // namespace executorch::runtime
