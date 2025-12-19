/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

#include <executorch/backends/aoti/slim/core/SlimTensor.h>
#include <executorch/backends/aoti/slim/util/ArrayRefUtil.h>
#include <executorch/backends/aoti/slim/util/SizeUtil.h>

namespace executorch::backends::aoti::slim {
// The returned SlimTensor owns the underlying storage
inline SlimTensor empty_strided(
    IntArrayRef sizes,
    IntArrayRef strides,
    executorch::backends::aoti::slim::c10::ScalarType dtype,
    const executorch::backends::aoti::slim::c10::Device& device = CPU_DEVICE) {
  Storage storage = new_storage(sizes, strides, dtype, device);
  return SlimTensor(std::move(storage), sizes, strides, dtype, 0);
}

inline SlimTensor empty(
    IntArrayRef sizes,
    executorch::backends::aoti::slim::c10::ScalarType dtype,
    const executorch::backends::aoti::slim::c10::Device& device = CPU_DEVICE) {
  std::vector<int64_t> contig_strides =
      executorch::backends::aoti::slim::compute_contiguous_strides(sizes);
  Storage storage =
      new_storage(sizes, makeArrayRef(contig_strides), dtype, device);
  return SlimTensor(
      std::move(storage), sizes, makeArrayRef(contig_strides), dtype, 0);
}

inline SlimTensor empty(
    std::initializer_list<int64_t> sizes,
    executorch::backends::aoti::slim::c10::ScalarType dtype,
    const executorch::backends::aoti::slim::c10::Device& device = CPU_DEVICE) {
  return empty(makeArrayRef(sizes), dtype, device);
}

inline SlimTensor empty_like(const SlimTensor& other) {
  return empty_strided(
      other.sizes(), other.strides(), other.dtype(), other.device());
}
} // namespace executorch::backends::aoti::slim
