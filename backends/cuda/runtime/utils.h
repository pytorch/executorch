/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda_runtime.h>
#include <executorch/backends/aoti/slim/c10/cuda/Exception.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

#include <executorch/backends/aoti/slim/c10/core/Device.h>
#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/backends/aoti/slim/core/storage.h>

namespace executorch::backends::cuda {

namespace {
inline executorch::runtime::Error _check_tensor_metadata(
    const executorch::backends::aoti::slim::SlimTensor* slim_tensor,
    executorch::runtime::etensor::Tensor* etensor) {
  ET_CHECK_OR_RETURN_ERROR(
      slim_tensor != nullptr,
      InvalidArgument,
      "slim_tensor pointer cannot be nullptr");

  ET_CHECK_OR_RETURN_ERROR(
      etensor != nullptr, InvalidArgument, "etensor pointer cannot be nullptr");

  // Check storage_offset is 0 (ETensor does not support storage offset)
  ET_CHECK_OR_RETURN_ERROR(
      slim_tensor->storage_offset() == 0,
      InvalidArgument,
      "SlimTensor storage_offset must be 0, got %ld",
      static_cast<long>(slim_tensor->storage_offset()));

  // Check that SlimTensor is contiguous
  ET_CHECK_OR_RETURN_ERROR(
      slim_tensor->is_contiguous(),
      InvalidArgument,
      "SlimTensor must be contiguous");

  // Check dtype matches
  executorch::backends::aoti::slim::c10::ScalarType slim_dtype =
      slim_tensor->dtype();
  executorch::runtime::etensor::ScalarType etensor_dtype =
      etensor->scalar_type();
  ET_CHECK_OR_RETURN_ERROR(
      static_cast<int>(slim_dtype) == static_cast<int>(etensor_dtype),
      InvalidArgument,
      "dtype mismatch, SlimTensor dtype %d != ETensor dtype %d",
      static_cast<int>(slim_dtype),
      static_cast<int>(etensor_dtype));

  // Check dimensions match
  ET_CHECK_OR_RETURN_ERROR(
      static_cast<ssize_t>(slim_tensor->dim()) == etensor->dim(),
      InvalidArgument,
      "dimension mismatch, SlimTensor dim %zu != ETensor dim %zd",
      slim_tensor->dim(),
      etensor->dim());

  // Convert sizes from int64_t to SizesType (int32_t) for resize
  const size_t ndim = slim_tensor->dim();
  std::vector<executorch::runtime::etensor::TensorImpl::SizesType> new_sizes(
      ndim);
  auto slim_sizes = slim_tensor->sizes();
  for (size_t i = 0; i < ndim; ++i) {
    new_sizes[i] =
        static_cast<executorch::runtime::etensor::TensorImpl::SizesType>(
            slim_sizes[i]);
  }

  // Resize ETensor to match SlimTensor sizes
  executorch::runtime::Error resize_err =
      executorch::ET_RUNTIME_NAMESPACE::resize_tensor(
          *etensor,
          executorch::runtime::ArrayRef<
              executorch::runtime::etensor::TensorImpl::SizesType>(
              new_sizes.data(), new_sizes.size()));
  ET_CHECK_OK_OR_RETURN_ERROR(resize_err, "failed to resize ETensor");

  return executorch::runtime::Error::Ok;
}
// Check if src and dst strides match (same layout, no rearrangement needed).
inline bool _strides_match(
    const executorch::backends::aoti::slim::SlimTensor* slim_tensor,
    const executorch::runtime::etensor::Tensor* etensor) {
  const size_t ndim = slim_tensor->dim();
  auto slim_strides = slim_tensor->strides();
  auto et_strides = etensor->strides();
  for (size_t i = 0; i < ndim; ++i) {
    if (slim_strides[i] != static_cast<int64_t>(et_strides[i])) {
      return false;
    }
  }
  return true;
}

// Element-by-element copy from a contiguous CPU buffer (src_strides layout)
// to an ETensor (dst_strides layout), rearranging data to match.
inline void _strided_copy(
    void* dst,
    const void* src,
    size_t elem_size,
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& src_strides,
    const std::vector<int32_t>& dst_strides) {
  const size_t ndim = sizes.size();
  const size_t numel = [&]() {
    size_t n = 1;
    for (auto s : sizes)
      n *= static_cast<size_t>(s);
    return n;
  }();

  // Iterate over all elements using N-dimensional index
  std::vector<int64_t> idx(ndim, 0);
  auto* dst_bytes = static_cast<char*>(dst);
  auto* src_bytes = static_cast<const char*>(src);

  for (size_t i = 0; i < numel; ++i) {
    // Compute source and destination byte offsets
    size_t src_offset = 0, dst_offset = 0;
    for (size_t d = 0; d < ndim; ++d) {
      src_offset += static_cast<size_t>(idx[d]) *
          static_cast<size_t>(src_strides[d]) * elem_size;
      dst_offset += static_cast<size_t>(idx[d]) *
          static_cast<size_t>(dst_strides[d]) * elem_size;
    }
    std::memcpy(dst_bytes + dst_offset, src_bytes + src_offset, elem_size);

    // Increment N-dimensional index (last dimension fastest)
    for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
      if (++idx[d] < sizes[d])
        break;
      idx[d] = 0;
    }
  }
}

// Copy data from SlimTensor to ETensor, rearranging if strides differ.
// When stream is non-null, GPU copies use that stream (async fast path).
// When stream is null, GPU copies are synchronous.
inline executorch::runtime::Error _copy_slimtensor_to_etensor_impl(
    const executorch::backends::aoti::slim::SlimTensor* slim_tensor,
    executorch::runtime::etensor::Tensor* etensor,
    cudaStream_t stream) {
  ET_CHECK_OK_OR_RETURN_ERROR(_check_tensor_metadata(slim_tensor, etensor));

  size_t nbytes = slim_tensor->nbytes();
  if (nbytes == 0) {
    return executorch::runtime::Error::Ok;
  }

  void* dst_data = etensor->mutable_data_ptr();
  const void* src_data = slim_tensor->data_ptr();

  if (_strides_match(slim_tensor, etensor)) {
    // Fast path: strides match, raw byte copy
    if (slim_tensor->is_cpu()) {
      std::memcpy(dst_data, src_data, nbytes);
    } else if (stream) {
      executorch::backends::aoti::slim::DeviceTraits<
          executorch::backends::aoti::slim::c10::DeviceType::CUDA>::
          memcpy_async(
              dst_data,
              src_data,
              nbytes,
              executorch::backends::aoti::slim::CPU_DEVICE,
              slim_tensor->device(),
              stream);
    } else {
      executorch::backends::aoti::slim::DeviceTraits<
          executorch::backends::aoti::slim::c10::DeviceType::CUDA>::
          memcpy(
              dst_data,
              src_data,
              nbytes,
              executorch::backends::aoti::slim::CPU_DEVICE,
              slim_tensor->device());
    }
  } else {
    // Slow path: strides differ (e.g., AOTI delegate output layout differs
    // from .pte's dim_order). Copy to a temp CPU buffer, then rearrange
    // element-by-element to match the ETensor's expected layout.
    std::vector<char> tmp(nbytes);
    if (slim_tensor->is_cpu()) {
      std::memcpy(tmp.data(), src_data, nbytes);
    } else {
      if (stream) {
        ET_CUDA_CHECK_OR_RETURN_ERROR(cudaStreamSynchronize(stream));
      }
      ET_CUDA_CHECK_OR_RETURN_ERROR(
          cudaMemcpy(tmp.data(), src_data, nbytes, cudaMemcpyDeviceToHost));
    }

    const size_t ndim = slim_tensor->dim();
    auto slim_sizes = slim_tensor->sizes();
    auto slim_strides = slim_tensor->strides();
    auto et_strides = etensor->strides();

    std::vector<int64_t> sizes_vec(ndim);
    std::vector<int64_t> src_strides_vec(ndim);
    std::vector<int32_t> dst_strides_vec(ndim);
    for (size_t i = 0; i < ndim; ++i) {
      sizes_vec[i] = slim_sizes[i];
      src_strides_vec[i] = slim_strides[i];
      dst_strides_vec[i] = et_strides[i];
    }

    size_t elem_size = executorch::backends::aoti::slim::c10::elementSize(
        slim_tensor->dtype());
    _strided_copy(
        dst_data,
        tmp.data(),
        elem_size,
        sizes_vec,
        src_strides_vec,
        dst_strides_vec);
  }

  return executorch::runtime::Error::Ok;
}
} // namespace

/**
 * Copies data from a SlimTensor to an ETensor asynchronously.
 *
 * When strides match (common case), performs a fast async GPU-to-CPU copy on
 * the provided stream. When strides differ (e.g., AOTI delegate output layout
 * differs from the .pte's dim_order), falls back to a synchronous copy with
 * element-by-element rearrangement.
 *
 * NOTE: In the fast path the copy is asynchronous. The caller must synchronize
 * the stream before reading the ETensor data on the CPU side.
 *
 * @param slim_tensor Pointer to the source SlimTensor (must not be null).
 * @param etensor Pointer to the destination ETensor (must not be null).
 * @param stream The CUDA stream to use for async copy.
 * @return Error::Ok on success, or an appropriate error code on failure.
 */
inline executorch::runtime::Error copy_slimtensor_to_etensor_async(
    const executorch::backends::aoti::slim::SlimTensor* slim_tensor,
    executorch::runtime::etensor::Tensor* etensor,
    cudaStream_t stream) {
  return _copy_slimtensor_to_etensor_impl(slim_tensor, etensor, stream);
}

/**
 * Copies data from a SlimTensor to an ETensor synchronously.
 *
 * Handles stride mismatches between the delegate output and the .pte's
 * expected layout by rearranging data element-by-element when needed.
 *
 * @param slim_tensor Pointer to the source SlimTensor (must not be null).
 * @param etensor Pointer to the destination ETensor (must not be null).
 * @return Error::Ok on success, or an appropriate error code on failure.
 */
inline executorch::runtime::Error copy_slimtensor_to_etensor(
    const executorch::backends::aoti::slim::SlimTensor* slim_tensor,
    executorch::runtime::etensor::Tensor* etensor) {
  return _copy_slimtensor_to_etensor_impl(slim_tensor, etensor, nullptr);
}

/**
 * Wraps a SlimTensor's data into an existing ETensor (zero-copy).
 *
 * This function resizes the ETensor to match the SlimTensor's shape and
 * sets its data pointer to point directly to the SlimTensor's data buffer.
 * No data is copied - the ETensor becomes a view of the SlimTensor's data.
 *
 * IMPORTANT: The caller must ensure the SlimTensor remains alive as long
 * as the ETensor is in use, since the ETensor will reference the SlimTensor's
 * data directly.
 *
 * @param slim_tensor Pointer to the source SlimTensor (must not be null).
 * @param etensor Pointer to the destination ETensor (must not be null).
 * @return Error::Ok on success, or an appropriate error code on failure.
 */
inline executorch::runtime::Error wrap_slimtensor_to_etensor(
    const executorch::backends::aoti::slim::SlimTensor* slim_tensor,
    executorch::runtime::etensor::Tensor* etensor) {
  ET_CHECK_OK_OR_RETURN_ERROR(_check_tensor_metadata(slim_tensor, etensor));

  // Set data pointer to point directly to SlimTensor's data (zero-copy)
  etensor->unsafeGetTensorImpl()->set_data(
      const_cast<void*>(slim_tensor->data_ptr()));

  return executorch::runtime::Error::Ok;
}

/**
 * Deletes all SlimTensor pointers in a vector and clears the vector.
 *
 * This utility function safely deletes each non-null SlimTensor pointer in the
 * vector and then clears the vector. This pattern is used in multiple places
 * in the CUDA backend to clean up GPU tensors.
 *
 * @param tensors Reference to a vector of SlimTensor pointers to delete.
 */
inline void delete_slimtensor_vector(
    std::vector<executorch::backends::aoti::slim::SlimTensor*>& tensors) {
  for (auto* tensor : tensors) {
    if (tensor != nullptr) {
      delete tensor;
    }
  }
  tensors.clear();
}

} // namespace executorch::backends::cuda
