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

inline SlimTensor SlimTensor::permute(IntArrayRef dims) const {
  const size_t ndim = this->dim();
  ET_CHECK_MSG(
      ndim == dims.size(),
      "permute: dims length (%zu) must equal tensor.dim() (%zu)",
      dims.size(),
      ndim);

  IntArrayRef old_sizes = this->sizes();
  IntArrayRef old_strides = this->strides();
  std::vector<int64_t> new_sizes(ndim);
  std::vector<int64_t> new_strides(ndim);
  std::vector<bool> seen_dims(ndim, false);

  for (size_t i = 0; i < ndim; i++) {
    int64_t d = c10::maybe_wrap_dim(dims[i], ndim);
    ET_CHECK_MSG(!seen_dims[d], "permute: duplicate dims are not allowed");
    seen_dims[d] = true;
    new_sizes[i] = old_sizes[d];
    new_strides[i] = old_strides[d];
  }

  SlimTensor result = *this;
  result.as_strided_(
      makeArrayRef(new_sizes),
      makeArrayRef(new_strides),
      this->storage_offset());
  return result;
}

inline SlimTensor SlimTensor::reshape(IntArrayRef proposed_shape) const {
  std::vector<int64_t> final_shape_vec =
      infer_size(proposed_shape, static_cast<int64_t>(this->numel()));

  // compute_stride returns the proper strides to use if this
  // reshape can be just a view.
  std::optional<std::vector<int64_t>> new_strides_opt = compute_stride(
      this->sizes(), this->strides(), makeArrayRef(final_shape_vec));

  // Create a view if possible
  if (new_strides_opt.has_value()) {
    SlimTensor result = *this;
    result.as_strided_(
        makeArrayRef(final_shape_vec),
        makeArrayRef(new_strides_opt.value()),
        this->storage_offset());
    return result;
  }

  // If a view is not possible, create a contiguous clone and reshape that
  SlimTensor contiguous_clone = this->clone_contiguous();
  // After cloning, the tensor is already contiguous. We just need to update
  // its metadata to reflect the new shape. This is effectively a view of
  // the new contiguous clone.
  contiguous_clone.set_sizes_contiguous(makeArrayRef(final_shape_vec));
  return contiguous_clone;
}

} // namespace executorch::backends::aoti::slim
