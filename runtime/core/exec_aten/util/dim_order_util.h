/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/compiler.h>

namespace executorch {
namespace runtime {

namespace {
template <typename DimOrderType>
bool validate_dim_order(const DimOrderType* dim_order, const size_t dims) {
  for (int32_t i = 0; i < dims; ++i) {
    if (dim_order[i] >= dims) {
      return false;
    }
  }
  return true;
}
} // namespace

/**
 * Check if a given dim_order array is equivalent to the contiguous dim order of
 * {0, 1, 2, 3, ...}
 *
 * @param[in] dim_order pointer to dim_order array
 * @param[in] dims length of the dim_order array
 */
template <typename DimOrderType>
inline bool is_contiguous_dim_order(
    const DimOrderType* dim_order,
    const size_t dims) {
  for (int i = 0; i < dims; ++i) {
    if (dim_order[i] != i) {
      return false;
    }
  }
  return true;
}

/**
 * Check if a given dim_order array is equivalent to a channels last dim order.
 * Channels last dim order is only valid for 4-dim and 5-dim tensors.
 *
 * @param[in] dim_order pointer to dim_order array
 * @param[in] dims length of the dim_order array
 */
template <typename DimOrderType>
bool is_channels_last_dim_order(
    const DimOrderType* dim_order,
    const size_t dims) {
  if (dims != 4 && dims != 5) {
    return false;
  }
  // 4-dim tensor is interpreted as NCHW, 5-dim tensor is interpreted as NCHWD
  size_t channels_dim = 1;
  // Last value in the dim order should be the channels dim
  if (dim_order[dims - 1] != channels_dim) {
    return false;
  }

  if (dim_order[0] != 0) {
    return false;
  }
  int d = 1;
  while (d < dims - 1) {
    if (dim_order[d] != d + 1) {
      return false;
    }
    d++;
  }
  return true;
}

/*
 * This utility translated sizes to strides by using dimension order
 * information. Dimension order specifies how the dimensions are laid out in the
 * memory. For example for Size = [2, 3, 4, 5] dim_names = [N, C, H, W]
 * dim_order = [0, 2, 3, 1]
 * strides = [60, 1, 15, 3]
 * param[in]: sizes, pointer to sizes array
 * param[in]: dim_order, pointer to dimension order array
 * param[in]: dims, number of dims. Sizes and dim_order must be sizes to dims
 * param[out]: strides, pointer to strides array that is filled in
 *
 * NB: Reason for not using ArrayRef is the dependency on kernel_types.h
 * This header cannot be included, because of circular dep it causes.
 * kernel_types depends on executorch_kernel_types in lean mode, which compiles
 * TensorImpl.cpp. executorch_kernel_types needs to depend on dim_order_utils
 * in order to utilize dim_order_to_stride in its resize impl. If
 * dim_order_utils depends on kernel_type, we have circular deps. This is also
 * the reason for templatizing this function. Better ideas welcome!
 * TODO(T148342910)
 *
 * Note that this function does not check that the provided dim order is valid.
 * This function should only be used when the validity of the dim order has been
 * checked beforehand. A safer version of this function is provided below as
 * dim_order_to_stride which will check that the dim order is valid.
 */
template <typename SizesType, typename DimOrderType, typename StridesType>
inline void dim_order_to_stride_nocheck(
    const SizesType* sizes,
    const DimOrderType* dim_order,
    const size_t dims,
    StridesType* strides) {
  // For 0 dim tensors, just return ok.
  if (dims == 0) {
    return;
  }
  // Fastest moving dim has stride of 1.
  // For example:
  // Size = [2, 3, 4, 5] dim_names = [N, C, H, W]
  // dim_order = [0, 2, 3, 1]
  // strides = [60, 1, 15, 3]
  strides[dim_order[dims - 1]] = 1;
  for (int32_t i = dims - 2; i >= 0; --i) {
    if (sizes[dim_order[i + 1]] == 0) {
      strides[dim_order[i]] = strides[dim_order[i + 1]];
    } else {
      strides[dim_order[i]] =
          strides[dim_order[i + 1]] * sizes[dim_order[i + 1]];
    }
  }
}

template <typename SizesType, typename DimOrderType, typename StridesType>
ET_NODISCARD inline Error dim_order_to_stride(
    const SizesType* sizes,
    const DimOrderType* dim_order,
    const size_t dims,
    StridesType* strides) {
  // For 0 dim tensors, just return ok.
  if (dims == 0) {
    return Error::Ok;
  }
  ET_CHECK_OR_RETURN_ERROR(
      validate_dim_order(dim_order, dims),
      InvalidArgument,
      "Invalid dim order. One of the value is larger than the number of dims %zu",
      dims);

  dim_order_to_stride_nocheck(sizes, dim_order, dims, strides);
  return Error::Ok;
}

namespace internal {

template <typename StridesType, typename DimOrderType>
struct StrideDimOrder {
  StridesType stride;
  DimOrderType dim_order;

  StrideDimOrder(StridesType stride, DimOrderType dim_order)
      : stride(stride), dim_order(dim_order) {}
  StrideDimOrder() = default;
  bool operator>(const StrideDimOrder& other) const {
    // descending order
    return stride < other.stride;
  }
};

template <typename ValueType>
struct Sorter {
 public:
  void quick_sort(ValueType arr[], int32_t low, int32_t high) {
    if (low < high) {
      ValueType pivot = arr[high];
      int32_t pos = partition(arr, low, high, pivot);

      quick_sort(arr, low, pos - 1);
      quick_sort(arr, pos + 1, high);
    }
  }

 private:
  void swap(ValueType arr[], int32_t pos1, int32_t pos2) noexcept {
    ValueType temp = arr[pos1];
    arr[pos1] = arr[pos2];
    arr[pos2] = temp;
  }

  int32_t
  partition(ValueType arr[], int32_t low, int32_t high, ValueType pivot) {
    int32_t i = low;
    int32_t j = low;
    while (i <= high) {
      if (arr[i] > pivot) {
        i++;
      } else {
        swap(arr, i++, j++);
      }
    }
    return j - 1;
  }
};

} // namespace internal

/*
 * This utility translated strides to dimension order
 * information. Dimension order specifies how the dimensions are laid out in the
 * memory. For example for tensor with sizes [3, 5, 2] and strides [5, 1, 15],
 * dim order should be [2, 0, 1], which is obtained by sorting strides in
 * descending order. param[in]: sizes, pointer to sizes array param[in]:
 * dim_order, pointer to dimension order array param[in]: dims, number of dims.
 * Sizes and dim_order must be sizes to dims param[out]: strides, pointer to
 * strides array that is filled in
 *
 * NB: Reason for not using ArrayRef is the dependency on kernel_types.h
 * This header cannot be included, because of circular dep it causes.
 * kernel_types depends on executorch_kernel_types in lean mode, which compiles
 * TensorImpl.cpp. executorch_kernel_types needs to depend on dim_order_utils
 * in order to utilize dim_order_to_stride in its resize impl. If
 * dim_order_utils depends on kernel_type, we have circular deps. This is also
 * the reason for templatizing this function. Better ideas welcome!
 * TODO(T148342910)
 */
template <typename DimOrderType, typename StridesType>
ET_NODISCARD inline Error stride_to_dim_order(
    const StridesType* strides,
    const size_t dims,
    DimOrderType* dim_order) {
  const size_t kMaxNumOfDimensions = 16;
  ET_CHECK_OR_RETURN_ERROR(
      dim_order != nullptr,
      MemoryAllocationFailed,
      "Need memory to get dim_order.");
  ET_CHECK_OR_RETURN_ERROR(
      dims <= kMaxNumOfDimensions,
      NotSupported,
      "dims %zu exceeds maximum allowed %zu",
      dims,
      kMaxNumOfDimensions);
  internal::StrideDimOrder<StridesType, DimOrderType>
      array[kMaxNumOfDimensions];
  for (DimOrderType i = 0; i < dims; i++) {
    array[i].dim_order = i;
    array[i].stride = strides[i];
  }

  internal::Sorter<internal::StrideDimOrder<StridesType, DimOrderType>> sorter;

  sorter.quick_sort(array, 0, dims - 1);

  for (auto i = 0; i < dims; i++) {
    dim_order[i] = array[i].dim_order;
  }
  return Error::Ok;
}
} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::dim_order_to_stride;
using ::executorch::runtime::dim_order_to_stride_nocheck;
using ::executorch::runtime::is_channels_last_dim_order;
using ::executorch::runtime::is_contiguous_dim_order;
using ::executorch::runtime::stride_to_dim_order;
} // namespace executor
} // namespace torch
