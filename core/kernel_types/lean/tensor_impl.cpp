// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <executorch/core/kernel_types/lean/tensor_impl.h>

#include <cstdint>
#include <cstring> // std::memcpy

#include <executorch/core/ArrayRef.h>
#include <executorch/core/Assert.h>
#include <executorch/core/Error.h>
#include <executorch/core/kernel_types/lean/qint_types.h>
#include <executorch/core/kernel_types/lean/scalar_type.h>
#include <executorch/core/kernel_types/util/DimOrderUtils.h>
#include <executorch/core/kernel_types/util/ScalarTypeUtil.h>

namespace torch {
namespace executor {

namespace {

/**
 * Compute the number of elements based on the sizes of a tensor.
 */
constexpr ssize_t compute_numel(
    const TensorImpl::SizesType* sizes,
    ssize_t dim) {
  ssize_t n = 1;
  for (ssize_t i = 0; i < dim; i++) {
    n *= sizes[i];
  }
  return n;
}
} // namespace

TensorImpl::TensorImpl(
    ScalarType type,
    ssize_t dim,
    SizesType* sizes,
    void* data,
    DimOrderType* dim_order,
    StridesType* strides,
    ssize_t storage_offset,
    TensorShapeDynamism dynamism)
    : sizes_(sizes),
      dim_order_(dim_order),
      strides_(strides),
      data_(data),
      dim_(dim),
      storage_offset_(storage_offset),
      numel_(compute_numel(sizes, dim)),
      capacity_(numel_ * sizeof_scalar_type(type)),
      type_(type),
      shape_dynamism_(dynamism) {}

size_t TensorImpl::nbytes() const {
  return numel_ * sizeof_scalar_type(type_);
}

ssize_t TensorImpl::size(ssize_t dim) const {
  ET_CHECK_MSG(
      dim < dim_ && dim >= 0,
      "Dimension out of range (expected to be in range of [0, %zd], but got %zd",
      dim_ - 1,
      dim);
  return sizes_[dim];
}

ssize_t TensorImpl::dim() const {
  return dim_;
}

ssize_t TensorImpl::numel() const {
  return numel_;
}

ScalarType TensorImpl::scalar_type() const {
  return type_;
}

// Return the size of one element of the tensor
ssize_t TensorImpl::element_size() const {
  return sizeof_scalar_type(type_);
}

ssize_t TensorImpl::storage_offset() const {
  return storage_offset_;
}

const ArrayRef<TensorImpl::SizesType> TensorImpl::sizes() const {
  return ArrayRef<SizesType>{sizes_, static_cast<size_t>(dim_)};
}

const ArrayRef<TensorImpl::DimOrderType> TensorImpl::dim_order() const {
  return ArrayRef<DimOrderType>{dim_order_, static_cast<size_t>(dim_)};
}

const ArrayRef<TensorImpl::StridesType> TensorImpl::strides() const {
  return ArrayRef<StridesType>{strides_, static_cast<size_t>(dim_)};
}

const void* TensorImpl::data() const {
  return mutable_data();
}

void* TensorImpl::mutable_data() const {
  if (data_ == nullptr) {
    return nullptr; // NOLINT facebook-hte-NullableReturn
  }
  return static_cast<void*>(
      static_cast<char*>(data_) + element_size() * storage_offset_);
}

void TensorImpl::set_data(void* ptr) {
  data_ = ptr;
}

Error TensorImpl::internal_resize_contiguous(ArrayRef<SizesType> new_sizes) {
  ET_CHECK_OR_RETURN_ERROR(
      new_sizes.size() == dim_,
      NotSupported,
      "ETensor rank is immutable old: %zu new: %zu",
      dim_,
      new_sizes.size());

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

  auto new_numel = compute_numel(new_sizes.data(), dim_);

  // Upper bounded tensors can be reshaped but not beyond upper bound
  if (shape_dynamism_ == TensorShapeDynamism::DYNAMIC_BOUND) {
    auto new_nbytes = new_numel * sizeof_scalar_type(type_);
    ET_CHECK_OR_RETURN_ERROR(
        new_nbytes <= capacity_,
        NotSupported,
        "Attempted to resize an upper bounded tensor "
        "to %zu which is beyond its capacity %zu",
        new_nbytes,
        capacity_);
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

bool TensorImpl::initialized() const {
  return data_ != nullptr;
}

} // namespace executor
} // namespace torch
