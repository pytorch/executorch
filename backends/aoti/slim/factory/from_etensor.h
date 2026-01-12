/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/backends/aoti/slim/factory/empty.h>
#include <executorch/backends/aoti/slim/util/array_ref_util.h>
#include <executorch/runtime/core/portable_type/tensor.h>

namespace executorch::backends::aoti::slim {

/// Creates a SlimTensor from an ETensor (ExecuTorch portable tensor).
///
/// This factory function converts an ETensor to a SlimTensor, optionally
/// copying the data to a target device. The ETensor is assumed to always
/// reside on CPU.
///
/// @param etensor The source ETensor (always on CPU).
/// @param target_device The target device for the output SlimTensor.
/// @return A new SlimTensor with data copied to the target device.
///
/// @note ETensor uses int32_t (SizesType/StridesType) for sizes and strides,
///       while SlimTensor uses int64_t. This function handles the conversion.
///
/// Example usage:
/// @code
///   auto* cpu_tensor = &(args[i]->toTensor());  // ETensor from EValue
///   SlimTensor gpu_tensor = from_etensor(*cpu_tensor, DEFAULT_CUDA_DEVICE);
/// @endcode
inline SlimTensor from_etensor(
    const executorch::runtime::etensor::Tensor& etensor,
    const c10::Device& target_device = CPU_DEVICE) {
  // Step 1: Extract metadata from ETensor
  const auto ndim = static_cast<size_t>(etensor.dim());

  // Convert sizes from exec_aten::SizesType (int32_t) to int64_t
  std::vector<int64_t> sizes_vec(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    sizes_vec[i] = static_cast<int64_t>(etensor.size(static_cast<ssize_t>(i)));
  }

  // Convert strides from exec_aten::StridesType (int32_t) to int64_t
  std::vector<int64_t> strides_vec(ndim);
  auto etensor_strides = etensor.strides();
  for (size_t i = 0; i < ndim; ++i) {
    strides_vec[i] = static_cast<int64_t>(etensor_strides[i]);
  }

  // Map ETensor ScalarType to SlimTensor ScalarType
  c10::ScalarType dtype = static_cast<c10::ScalarType>(etensor.scalar_type());

  // Step 2: Create SlimTensor on target device
  SlimTensor result = empty_strided(
      makeArrayRef(sizes_vec), makeArrayRef(strides_vec), dtype, target_device);

  // Step 3: Copy data from ETensor (CPU) to SlimTensor (target device)
  // ETensor is always on CPU, so this handles CPU→CPU or CPU→CUDA copy
  const void* src_data = etensor.const_data_ptr();
  void* dst_data = result.data_ptr();
  size_t nbytes = etensor.nbytes();

  if (nbytes > 0) {
    // const_cast is safe here because copy_ only reads from src_data
    result.storage()->copy_(
        dst_data, const_cast<void*>(src_data), nbytes, CPU_DEVICE);
  }

  return result;
}

/// Creates a SlimTensor from an ETensor pointer.
///
/// Convenience overload that accepts a pointer instead of a reference.
///
/// @param etensor Pointer to the source ETensor (must not be null).
/// @param target_device The target device for the output SlimTensor.
/// @return A new SlimTensor with data copied to the target device.
inline SlimTensor from_etensor(
    const executorch::runtime::etensor::Tensor* etensor,
    const c10::Device& target_device = CPU_DEVICE) {
  ET_CHECK_MSG(
      etensor != nullptr, "from_etensor: etensor pointer cannot be nullptr");
  return from_etensor(*etensor, target_device);
}

} // namespace executorch::backends::aoti::slim
