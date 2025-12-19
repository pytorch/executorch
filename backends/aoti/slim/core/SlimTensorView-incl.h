#pragma once

#include <vector>

#include <executorch/backends/aoti/slim/c10/core/WrapDimMinimal.h>
#include <executorch/backends/aoti/slim/c10/util/ArrayRef.h>
#include <executorch/backends/aoti/slim/util/SizeUtil.h>

namespace executorch::backends::aoti::slim {
inline SlimTensor SlimTensor::as_strided(
    executorch::backends::aoti::slim::c10::IntArrayRef sizes,
    executorch::backends::aoti::slim::c10::IntArrayRef strides,
    int64_t storage_offset) const {
  SlimTensor result = *this;
  result.as_strided_(sizes, strides, storage_offset);
  return result;
}

inline SlimTensor SlimTensor::as_strided_(
    executorch::backends::aoti::slim::c10::IntArrayRef sizes,
    executorch::backends::aoti::slim::c10::IntArrayRef strides,
    int64_t storage_offset) {
  STANDALONE_CHECK(
      sizes.size() == strides.size(),
      "as_strided: number of sizes (",
      sizes.size(),
      ") must equal number of strides (",
      strides.size(),
      ")");
  for (size_t i = 0; i < sizes.size(); ++i) {
    STANDALONE_CHECK(
        sizes[i] >= 0,
        "as_strided: size at dimension ",
        i,
        " is negative: ",
        sizes[i]);
  }
  STANDALONE_CHECK(
      storage_offset >= 0,
      "as_strided: storage_offset must be non-negative, got: ",
      storage_offset);

  this->set_sizes_and_strides(sizes, strides, storage_offset);
  return *this;
}

inline SlimTensor SlimTensor::permute(
    executorch::backends::aoti::slim::c10::IntArrayRef dims) const {
  const size_t ndim = this->dim();
  STANDALONE_CHECK(
      ndim == static_cast<size_t>(dims.size()),
      "permute: dims length must be equal to tensor.dim()")

  executorch::backends::aoti::slim::c10::ArrayRef old_sizes = this->sizes();
  executorch::backends::aoti::slim::c10::ArrayRef old_strides = this->strides();
  std::vector<int64_t> new_sizes = old_sizes.vec();
  std::vector<int64_t> new_strides = old_strides.vec();
  std::vector<bool> seen_dims(ndim, false);

  for (size_t i = 0; i < ndim; i++) {
    int64_t d =
        executorch::backends::aoti::slim::c10::maybe_wrap_dim(dims[i], ndim);
    STANDALONE_CHECK(!seen_dims[d], "permute: duplicate dims are not allowed");
    seen_dims[d] = true;
    new_sizes[i] = old_sizes[d];
    new_strides[i] = old_strides[d];
  }

  SlimTensor result = *this;
  result.as_strided_(new_sizes, new_strides, this->storage_offset());
  return result;
}

inline SlimTensor SlimTensor::transpose() const {
  STANDALONE_CHECK(dim() == 2, "transpose() can only be called on 2D tensors");
  return permute({1, 0});
}

inline SlimTensor SlimTensor::transpose(int64_t dim0, int64_t dim1) const {
  const size_t ndim = this->dim();
  std::vector<int64_t> dims;
  for (size_t i = 0; i < ndim; i++) {
    dims.push_back(static_cast<int64_t>(i));
  }

  // Wrap dimensions and swap them
  dim0 = executorch::backends::aoti::slim::c10::maybe_wrap_dim(dim0, ndim);
  dim1 = executorch::backends::aoti::slim::c10::maybe_wrap_dim(dim1, ndim);
  std::swap(dims[dim0], dims[dim1]);

  return permute(dims);
}

inline SlimTensor SlimTensor::t() const {
  return transpose();
}

inline SlimTensor SlimTensor::reshape(
    executorch::backends::aoti::slim::c10::IntArrayRef proposed_shape) const {
  std::vector<int64_t> final_shape_vec =
      infer_size(proposed_shape, this->numel());

  // `compute_stride` return the proper strides to use if this
  // `reshape` can be just a view.
  std::optional<std::vector<int64_t>> new_strides_opt =
      compute_stride(this->sizes(), this->strides(), final_shape_vec);

  // create a view if possible
  if (new_strides_opt.has_value()) {
    SlimTensor result = *this;
    result.as_strided_(
        final_shape_vec, new_strides_opt.value(), this->storage_offset());
    return result;
  }

  // if a view is not possible, create a contiguous clone and reshape that
  SlimTensor contiguous_clone = this->clone_contiguous();
  // after cloning, the tensor is already contiguous. We just need to update
  // its metadata to reflect the new shape. This is effectively a view of
  // the new contiguous clone
  contiguous_clone.set_sizes_contiguous(final_shape_vec);
  return contiguous_clone;
}

inline SlimTensor SlimTensor::narrow(int64_t dim, int64_t start, int64_t length)
    const {
  STANDALONE_CHECK(
      this->dim() > 0, "narrow() cannot be applied to a 0-dim tensor.");
  dim = executorch::backends::aoti::slim::c10::maybe_wrap_dim(
      dim, static_cast<int64_t>(this->dim()));
  start = executorch::backends::aoti::slim::c10::maybe_wrap_dim(
      start, static_cast<int64_t>(this->size(dim)));

  STANDALONE_CHECK(length >= 0, "narrow(): length must be non-negative.");
  int64_t end = start + length;
  STANDALONE_CHECK(
      end <= this->size(dim),
      "Invalid range to narrow. range(",
      start,
      ", ",
      start + length,
      ") must be a subset of range(0, ",
      this->size(dim),
      ").");

  SlimTensor result = *this;
  int64_t new_storage_offset =
      this->storage_offset() + start * this->stride(dim);
  std::vector<int64_t> new_sizes = this->sizes().vec();
  new_sizes[dim] = length;
  result.as_strided_(new_sizes, this->strides(), new_storage_offset);
  return result;
}

} // namespace executorch::backends::aoti::slim
