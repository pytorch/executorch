/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/backends/aoti/slim/util/array_ref_util.h>
#include <executorch/backends/aoti/slim/util/size_util.h>

namespace executorch::backends::aoti::slim {

/// Creates an empty tensor with the specified sizes and strides.
/// The tensor owns its underlying storage, which is allocated but
/// uninitialized.
///
/// @param sizes The sizes of each dimension.
/// @param strides The strides of each dimension.
/// @param dtype The scalar type of tensor elements.
/// @param device The target device.
/// @return A new SlimTensor with allocated but uninitialized storage.
inline SlimTensor empty_strided(
    IntArrayRef sizes,
    IntArrayRef strides,
    c10::ScalarType dtype,
    const c10::Device& device = CPU_DEVICE) {
  Storage storage = new_storage(sizes, strides, dtype, device);
  return SlimTensor(std::move(storage), sizes, strides, dtype, 0);
}

/// Creates an empty contiguous tensor with the specified sizes.
/// The tensor owns its underlying storage, which is allocated but
/// uninitialized. Strides are computed automatically to be contiguous
/// (row-major order).
///
/// @param sizes The sizes of each dimension.
/// @param dtype The scalar type of tensor elements.
/// @param device The target device.
/// @return A new SlimTensor with contiguous strides and uninitialized storage.
inline SlimTensor empty(
    IntArrayRef sizes,
    c10::ScalarType dtype,
    const c10::Device& device = CPU_DEVICE) {
  std::vector<int64_t> contig_strides = compute_contiguous_strides(sizes);
  Storage storage =
      new_storage(sizes, makeArrayRef(contig_strides), dtype, device);
  return SlimTensor(
      std::move(storage), sizes, makeArrayRef(contig_strides), dtype, 0);
}

/// Creates an empty contiguous tensor with sizes from an initializer list.
/// Convenience overload for creating tensors with inline size specification.
///
/// @param sizes The sizes of each dimension as an initializer list.
/// @param dtype The scalar type of tensor elements.
/// @param device The target device.
/// @return A new SlimTensor with contiguous strides and uninitialized storage.
inline SlimTensor empty(
    std::initializer_list<int64_t> sizes,
    c10::ScalarType dtype,
    const c10::Device& device = CPU_DEVICE) {
  return empty(makeArrayRef(sizes), dtype, device);
}

/// Creates an empty tensor with the same sizes, strides, dtype, and device as
/// another tensor.
///
/// @param other The tensor to copy metadata from.
/// @return A new SlimTensor with the same shape and properties, but
/// uninitialized storage.
inline SlimTensor empty_like(const SlimTensor& other) {
  return empty_strided(
      other.sizes(), other.strides(), other.dtype(), other.device());
}

} // namespace executorch::backends::aoti::slim
