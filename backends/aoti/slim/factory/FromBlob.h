/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/aoti/slim/core/SlimTensor.h>
#include <executorch/backends/aoti/slim/util/ArrayRefUtil.h>
#include <executorch/backends/aoti/slim/util/SizeUtil.h>

namespace executorch::backends::aoti::slim {

/// Creates a SlimTensor that wraps external memory without taking ownership.
/// The returned tensor does NOT own the underlying storage; the caller is
/// responsible for keeping the data alive for the lifetime of the tensor.
///
/// @param data Pointer to external memory (must not be null).
/// @param sizes The sizes of each dimension.
/// @param strides The strides of each dimension.
/// @param dtype The scalar type of tensor elements.
/// @param device The device where the data resides.
/// @param storage_offset Offset into storage in number of elements.
/// @return A new SlimTensor with non-owning storage.
inline SlimTensor from_blob(
    void* data,
    IntArrayRef sizes,
    IntArrayRef strides,
    c10::ScalarType dtype,
    const c10::Device& device = CPU_DEVICE,
    int64_t storage_offset = 0) {
  ET_CHECK_MSG(data != nullptr, "from_blob: data pointer cannot be nullptr");

  size_t nbytes = compute_storage_nbytes(
      sizes, strides, c10::elementSize(dtype), storage_offset);

  Storage storage(new MaybeOwningStorage(device, data, nbytes));
  return SlimTensor(std::move(storage), sizes, strides, dtype, storage_offset);
}

/// Creates a contiguous SlimTensor that wraps external memory.
/// Computes contiguous strides automatically.
///
/// @param data Pointer to external memory (must not be null).
/// @param sizes The sizes of each dimension.
/// @param dtype The scalar type of tensor elements.
/// @param device The device where the data resides.
/// @param storage_offset Offset into storage in number of elements.
/// @return A new SlimTensor with non-owning storage and contiguous strides.
inline SlimTensor from_blob(
    void* data,
    IntArrayRef sizes,
    c10::ScalarType dtype,
    const c10::Device& device = CPU_DEVICE,
    int64_t storage_offset = 0) {
  std::vector<int64_t> contig_strides = compute_contiguous_strides(sizes);
  return from_blob(
      data, sizes, makeArrayRef(contig_strides), dtype, device, storage_offset);
}

/// Creates a contiguous SlimTensor from an initializer list of sizes.
///
/// @param data Pointer to external memory (must not be null).
/// @param sizes The sizes as an initializer list.
/// @param dtype The scalar type of tensor elements.
/// @param device The device where the data resides.
/// @param storage_offset Offset into storage in number of elements.
/// @return A new SlimTensor with non-owning storage and contiguous strides.
inline SlimTensor from_blob(
    void* data,
    std::initializer_list<int64_t> sizes,
    c10::ScalarType dtype,
    const c10::Device& device = CPU_DEVICE,
    int64_t storage_offset = 0) {
  return from_blob(data, makeArrayRef(sizes), dtype, device, storage_offset);
}

/// Creates a SlimTensor from initializer lists for both sizes and strides.
///
/// @param data Pointer to external memory (must not be null).
/// @param sizes The sizes as an initializer list.
/// @param strides The strides as an initializer list.
/// @param dtype The scalar type of tensor elements.
/// @param device The device where the data resides.
/// @param storage_offset Offset into storage in number of elements.
/// @return A new SlimTensor with non-owning storage.
inline SlimTensor from_blob(
    void* data,
    std::initializer_list<int64_t> sizes,
    std::initializer_list<int64_t> strides,
    c10::ScalarType dtype,
    const c10::Device& device = CPU_DEVICE,
    int64_t storage_offset = 0) {
  return from_blob(
      data,
      makeArrayRef(sizes),
      makeArrayRef(strides),
      dtype,
      device,
      storage_offset);
}

} // namespace executorch::backends::aoti::slim
