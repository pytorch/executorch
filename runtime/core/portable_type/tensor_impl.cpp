/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/portable_type/tensor_impl.h>

#include <cstdint>
#include <cstring> // std::memcpy
#include <functional> // std::multiplies
#include <numeric> // std::accumulate

#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/portable_type/qint_types.h>
#include <executorch/runtime/core/portable_type/scalar_type.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {

namespace {
/**
 * Compute the number of elements based on the sizes of a tensor.
 */
constexpr ssize_t compute_numel(
    const TensorImpl::SizesType* sizes,
    ssize_t dim) {
  return dim == 0
      ? 0
      : std::accumulate(
            sizes, sizes + dim, static_cast<ssize_t>(1), std::multiplies<>());
}
} // namespace

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
      shape_dynamism_(dynamism) {}

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
      "ETensor rank is immutable old: %zu new: %zu",
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

  // Can only resize a StaticShape Tensor to the same size
  if (shape_dynamism_ == TensorShapeDynamism::STATIC) {
    for (int i = 0; i < new_sizes.size(); i++) {
      ET_CHECK_OR_RETURN_ERROR(
          new_sizes[i] == sizes_[i],
          NotSupported,
          "Attempted to resize a static tensor to a new shape at "
          "dimension %d old_size: %d new_size: %d",
          i,
          sizes_[i],
          new_sizes[i]);
    }
    // no work to do after checking for error
    return Error::Ok;
  }

  const auto new_numel = compute_numel(new_sizes.data(), dim_);

  // Bounded tensors can be reshaped, but not beyond the upper bound.
  if (shape_dynamism_ == TensorShapeDynamism::DYNAMIC_BOUND ||
      // TODO(T175194371): Unbounded dynamic tensor resizing is not yet
      // supported: treat them as upper-bounded.
      shape_dynamism_ == TensorShapeDynamism::DYNAMIC_UNBOUND) {
    ET_CHECK_OR_RETURN_ERROR(
        new_numel <= numel_bound_,
        NotSupported,
        "Attempted to resize a bounded tensor with capacity of %zu elements to %zu elements.",
        new_numel,
        numel_bound_);
  }

  // Copy sizes over
  std::memcpy(sizes_, new_sizes.data(), sizeof(SizesType) * dim_);

  // Compute new strides
  ET_CHECK_OR_RETURN_ERROR(
      strides_ != nullptr, Internal, "Strides cannot be nullptr for resize");
  ET_CHECK_OR_RETURN_ERROR(
      dim_order_ != nullptr,
      Internal,
      "Dim order cannot be nullptr for resize");
  auto status = dim_order_to_stride(sizes_, dim_order_, dim_, strides_);
  ET_CHECK_OR_RETURN_ERROR(
      status == Error::Ok,
      Internal,
      "dim_order_to_stride returned invalid status");
  numel_ = new_numel;

  return Error::Ok;
}

} // namespace executor
} // namespace torch
