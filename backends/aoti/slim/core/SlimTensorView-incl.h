/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/aoti/slim/c10/core/WrapDimMinimal.h>
#include <executorch/backends/aoti/slim/util/ArrayRefUtil.h>

namespace executorch::backends::aoti::slim {

inline SlimTensor SlimTensor::as_strided(
    IntArrayRef sizes,
    IntArrayRef strides,
    int64_t storage_offset) const {
  SlimTensor result = *this;
  result.as_strided_(sizes, strides, storage_offset);
  return result;
}

inline SlimTensor& SlimTensor::as_strided_(
    IntArrayRef sizes,
    IntArrayRef strides,
    int64_t storage_offset) {
  ET_CHECK_MSG(
      sizes.size() == strides.size(),
      "as_strided: number of sizes (%zu) must equal number of strides (%zu)",
      sizes.size(),
      strides.size());

  for (size_t i = 0; i < sizes.size(); ++i) {
    ET_CHECK_MSG(
        sizes[i] >= 0,
        "as_strided: size at dimension %zu is negative: %ld",
        i,
        static_cast<long>(sizes[i]));
  }

  ET_CHECK_MSG(
      storage_offset >= 0,
      "as_strided: storage_offset must be non-negative, got: %ld",
      static_cast<long>(storage_offset));

  this->set_sizes_and_strides(sizes, strides, storage_offset);
  return *this;
}

} // namespace executorch::backends::aoti::slim
