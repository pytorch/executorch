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
  ET_CHECK_OR_RETURN_ERROR(
      slim_tensor != nullptr,
      InvalidArgument,
      "copy_slimtensor_to_etensor: slim_tensor pointer cannot be nullptr");

  ET_CHECK_OR_RETURN_ERROR(
      etensor != nullptr,
      InvalidArgument,
      "copy_slimtensor_to_etensor: etensor pointer cannot be nullptr");

  // Check storage_offset is 0 (ETensor does not support storage offset)
  ET_CHECK_OR_RETURN_ERROR(
      slim_tensor->storage_offset() == 0,
      InvalidArgument,
      "copy_slimtensor_to_etensor: SlimTensor storage_offset must be 0, got %ld",
      static_cast<long>(slim_tensor->storage_offset()));

  // Check that SlimTensor is contiguous
  ET_CHECK_OR_RETURN_ERROR(
      slim_tensor->is_contiguous(),
      InvalidArgument,
      "copy_slimtensor_to_etensor: SlimTensor must be contiguous");

  // Check dtype matches
  executorch::backends::aoti::slim::c10::ScalarType slim_dtype =
      slim_tensor->dtype();
  executorch::runtime::etensor::ScalarType etensor_dtype =
      etensor->scalar_type();
  ET_CHECK_OR_RETURN_ERROR(
      static_cast<int>(slim_dtype) == static_cast<int>(etensor_dtype),
      InvalidArgument,
      "copy_slimtensor_to_etensor: dtype mismatch, SlimTensor dtype %d != ETensor dtype %d",
      static_cast<int>(slim_dtype),
      static_cast<int>(etensor_dtype));

  // Check dimensions match
  ET_CHECK_OR_RETURN_ERROR(
      static_cast<ssize_t>(slim_tensor->dim()) == etensor->dim(),
      InvalidArgument,
      "copy_slimtensor_to_etensor: dimension mismatch, SlimTensor dim %zu != ETensor dim %zd",
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
  ET_CHECK_OK_OR_RETURN_ERROR(
      resize_err, "copy_slimtensor_to_etensor: failed to resize ETensor");

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

} // namespace executorch::backends::cuda
