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
#include <limits>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
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
    std::is_same_v<executorch::aten::SizesType, int32_t>
    ? std::numeric_limits<int32_t>::max()
    : std::numeric_limits<uint32_t>::max();
} // namespace internal

/**
 * Convert a shape to a NUL-terminated C string with limited size. If
 * elements of the shape are larger than
 * kMaximumPrintableTensorShapeElement, those elements will be
 * rendered as ERR instead.
 */
std::array<char, kTensorShapeStringSizeLimit> tensor_shape_to_c_string(
    executorch::runtime::Span<const executorch::aten::SizesType> shape);

} // namespace executorch::runtime
