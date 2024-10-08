/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/portable_type/tensor_impl.h>

#include <algorithm>
#include <cstdint>

#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/portable_type/qint_types.h>
#include <executorch/runtime/core/portable_type/scalar_type.h>
#include <executorch/runtime/platform/assert.h>

namespace executorch {
namespace runtime {
namespace etensor {

/**
 * Compute the number of elements based on the sizes of a tensor.
 */
ssize_t compute_numel(const TensorImpl::SizesType* sizes, ssize_t dim) {
  ET_CHECK_MSG(
      dim == 0 || sizes != nullptr,
      "Sizes must be provided for non-scalar tensors");
  ssize_t numel = 1; // Zero-dimensional tensors (scalars) have numel == 1.
  for (ssize_t i = 0; i < dim; ++i) {
    ET_CHECK_MSG(
        sizes[i] >= 0,
        "Size must be non-negative, got %d at dimension %zd",
        sizes[i],
        i);
    numel *= sizes[i];
  }
  return numel;
}

TensorImpl::TensorImpl(
    ScalarType type,
    ssize_t dim,
    SizesType* sizes,
    void* data,
    DimOrderType* dim_order,
    StridesType* strides,
    TensorShapeDynamism dynamism)
    : sizes_(sizes),
      dim_order_(dim_order),
      strides_(strides),
      data_(data),
      dim_(dim),
      numel_(compute_numel(sizes, dim)),
      numel_bound_(numel_),
      type_(type),
      shape_dynamism_(dynamism) {
  ET_CHECK_MSG(
      isValid(type_), "Invalid type %" PRId8, static_cast<int8_t>(type_));
  ET_CHECK_MSG(dim_ >= 0, "Dimension must be non-negative, got %zd", dim_);
}

size_t TensorImpl::nbytes() const {
  return numel_ * elementSize(type_);
}

// Return the size of one element of the tensor
ssize_t TensorImpl::element_size() const {
  return elementSize(type_);
}

Error TensorImpl::internal_resize_contiguous(ArrayRef<SizesType> new_sizes) {
  ET_CHECK_OR_RETURN_ERROR(
      new_sizes.size() == dim_,
      NotSupported,
      "Attempted to change the tensor rank which is immutable: old=%zu, new=%zu",
      dim_,
      new_sizes.size());

  // Kernels don't check that the provided out tensors have the right size.
  // Instead they always attempt to resize the out tensor to the right size,
  // even when the out tensor already had the right size. Therefore, if we call
  // an op with inputs that will produce a zero-dimensional output, and the out
  // tensor that we pass has non-STATIC dynamism, then we will end up here.
  // Since we have already checked above that the out tensor has the right
  // number of dimensions, it must be that the provided out tensor has zero
  // rank, therefore it already has the right size and we should just return.
  if (dim_ == 0) {
    return Error::Ok;
  }
  switch (shape_dynamism_) {
    case TensorShapeDynamism::STATIC:
      ET_CHECK_OR_RETURN_ERROR(
          std::equal(sizes_, sizes_ + dim_, new_sizes.begin()),
          NotSupported,
          "Attempted to resize a static tensor");
      break;
    case TensorShapeDynamism::DYNAMIC_BOUND:
      // TODO(T175194371): Unbounded dynamic tensor resizing is not yet
      // supported: treat them as upper-bounded.
    case TensorShapeDynamism::DYNAMIC_UNBOUND: {
      const auto new_numel = compute_numel(new_sizes.data(), dim_);
      ET_CHECK_OR_RETURN_ERROR(
          new_numel <= numel_bound_,
          NotSupported,
          "Attempted to resize a bounded tensor with capacity of %zu elements to %zu elements.",
          new_numel,
          numel_bound_);

      if (strides_ && dim_order_) {
        ET_CHECK_OK_OR_RETURN_ERROR(
            dim_order_to_stride(new_sizes.data(), dim_order_, dim_, strides_));
      }
      numel_ = new_numel;
      std::copy(new_sizes.begin(), new_sizes.end(), sizes_);
    }
  }
  return Error::Ok;
}

} // namespace etensor
} // namespace runtime
} // namespace executorch
