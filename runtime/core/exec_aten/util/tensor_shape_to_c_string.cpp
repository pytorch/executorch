/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/exec_aten/util/tensor_shape_to_c_string.h>

#include <executorch/runtime/platform/assert.h>

#include <cinttypes>
#include <cstdio> // For snprintf.
#include <cstring>

namespace executorch::runtime {
namespace {
template <typename SizesType>
std::array<char, kTensorShapeStringSizeLimit> tensor_shape_to_c_string_impl(
    executorch::runtime::Span<SizesType> shape) {
  std::array<char, kTensorShapeStringSizeLimit> out;
  char* p = out.data();
  if ET_UNLIKELY (shape.size() > kTensorDimensionLimit) {
    static constexpr char kLimitExceededError[] =
        "(ERR: tensor ndim exceeds limit)";
    static_assert(sizeof(kLimitExceededError) <= kTensorShapeStringSizeLimit);
    std::strcpy(p, kLimitExceededError);
    return out;
  }
  *p++ = '(';
  for (const auto elem : shape) {
    if (elem < 0 || elem > internal::kMaximumPrintableTensorShapeElement) {
      static_assert(
          internal::kMaximumPrintableTensorShapeElement > 99999,
          "must have room for error string!");
      strcpy(p, "ERR, ");
      p += strlen("ERR, ");
    } else {
      // snprintf returns characters *except* the NUL terminator, which is what
      // we want.
      p += snprintf(
          p,
          kTensorShapeStringSizeLimit - (p - out.data()),
          "%" PRIu32 ", ",
          static_cast<uint32_t>(elem));
    }
  }
  *(p - 2) = ')';
  *(p - 1) = '\0';
  return out;
}
} // namespace

std::array<char, kTensorShapeStringSizeLimit> tensor_shape_to_c_string(
    executorch::runtime::Span<const std::int32_t> shape) {
  return tensor_shape_to_c_string_impl(shape);
}

std::array<char, kTensorShapeStringSizeLimit> tensor_shape_to_c_string(
    executorch::runtime::Span<const std::int64_t> shape) {
  return tensor_shape_to_c_string_impl(shape);
}

} // namespace executorch::runtime
