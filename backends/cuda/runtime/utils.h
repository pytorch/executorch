/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda_runtime.h>
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
} // namespace

/**
 * Copies data from a SlimTensor to an ETensor.
 *
 * This function converts a SlimTensor back to an ETensor. The ETensor is
 * assumed to always reside on CPU, so this handles both CPU→CPU and GPU→CPU
 * copies. The function will resize the ETensor if needed and copy the data.
 *
 * @param slim_tensor Pointer to the source SlimTensor (must not be null).
 * @param etensor Pointer to the destination ETensor (must not be null).
 * @return Error::Ok on success, or an appropriate error code on failure.
 */
inline executorch::runtime::Error copy_slimtensor_to_etensor(
    const executorch::backends::aoti::slim::SlimTensor* slim_tensor,
    executorch::runtime::etensor::Tensor* etensor) {
  _check_tensor_metadata(slim_tensor, etensor);

  // Copy data from SlimTensor to ETensor
  // SlimTensor may be on GPU or CPU, ETensor is always on CPU
  size_t nbytes = slim_tensor->nbytes();
  if (nbytes > 0) {
    void* dst_data = etensor->mutable_data_ptr();
    const void* src_data = slim_tensor->data_ptr();

    if (slim_tensor->is_cpu()) {
      // CPU → CPU copy
      std::memcpy(dst_data, src_data, nbytes);
    } else {
      // GPU → CPU copy
      executorch::backends::aoti::slim::DeviceTraits<
          executorch::backends::aoti::slim::c10::DeviceType::CUDA>::
          memcpy(
              dst_data,
              src_data,
              nbytes,
              executorch::backends::aoti::slim::CPU_DEVICE,
              slim_tensor->device());
    }
  }

  return executorch::runtime::Error::Ok;
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
  _check_tensor_metadata(slim_tensor, etensor);

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
