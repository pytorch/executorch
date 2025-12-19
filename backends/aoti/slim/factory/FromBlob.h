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

// The returned SlimTensor does not own the underlying storage
inline SlimTensor from_blob(
    void* data,
    IntArrayRef sizes,
    IntArrayRef strides,
    executorch::backends::aoti::slim::c10::ScalarType dtype,
    const executorch::backends::aoti::slim::c10::Device& device = CPU_DEVICE,
    int64_t storage_offset = 0) {
  ET_CHECK_MSG(data != nullptr, "data pointer can not be nullptr");

  Storage storage(new MaybeOwningStorage(
      device,
      data,
      compute_storage_nbytes(
          sizes, strides, elementSize(dtype), storage_offset)));
  return SlimTensor(std::move(storage), sizes, strides, dtype, storage_offset);
}

inline SlimTensor from_blob(
    void* data,
    IntArrayRef sizes,
    executorch::backends::aoti::slim::c10::ScalarType dtype,
    const executorch::backends::aoti::slim::c10::Device& device = CPU_DEVICE,
    int64_t storage_offset = 0) {
  std::vector<int64_t> contig_strides =
      executorch::backends::aoti::slim::compute_contiguous_strides(sizes);
  return from_blob(
      data, sizes, makeArrayRef(contig_strides), dtype, device, storage_offset);
}

inline SlimTensor from_blob(
    void* data,
    std::initializer_list<int64_t> sizes,
    executorch::backends::aoti::slim::c10::ScalarType dtype,
    const executorch::backends::aoti::slim::c10::Device& device = CPU_DEVICE,
    int64_t storage_offset = 0) {
  return from_blob(data, makeArrayRef(sizes), dtype, device, storage_offset);
}

inline SlimTensor from_blob(
    void* data,
    std::initializer_list<int64_t> sizes,
    std::initializer_list<int64_t> strides,
    executorch::backends::aoti::slim::c10::ScalarType dtype,
    const executorch::backends::aoti::slim::c10::Device& device = CPU_DEVICE,
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
