/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/aoti/common_shims.h>
#include <executorch/backends/aoti/utils.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/backends/cuda/runtime/shims/tensor_attribute.h>
#include <executorch/backends/cuda/runtime/shims/utils.h>
#include <executorch/runtime/platform/log.h>
#include <cstdint>
#include <cstdlib> // For posix_memalign
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace executorch {
namespace backends {
namespace cuda {

using executorch::aten::SizesType;
using executorch::aten::StridesType;
using executorch::backends::aoti::aoti_torch_get_device_index;
using executorch::backends::aoti::aoti_torch_get_dtype;
using executorch::backends::aoti::dtype_to_element_size;
using executorch::backends::aoti::dtype_to_scalar_type;
using executorch::backends::aoti::validate_storage_offset;

// Global storage for tensors and their metadata
std::unordered_set<std::shared_ptr<Tensor>> tensors;

// Reference counting for memory addresses
// Maps memory address to number of tensors using it
// Special value: NOT_OWN (-1) means tensor never owns the memory
constexpr int32_t NOT_OWN = -1;
std::unordered_map<void*, int32_t> memory_to_n_tensor;

extern "C" {

AOTITorchError aoti_torch_create_tensor_from_blob_v2(
    void* data,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    Tensor** ret_new_tensor,
    int32_t layout,
    const uint8_t* opaque_metadata,
    int64_t opaque_metadata_size) {
  // TODO(gasoonjia): verify given data is on the target device
  (void)device_type;
  (void)opaque_metadata;
  (void)layout;
  (void)opaque_metadata_size;

  // Validate input parameters first
  if (data == nullptr) {
    ET_LOG(
        Error,
        "aoti_torch_create_tensor_from_blob_v2 failed: data pointer is null");
    return Error::InvalidArgument;
  }

  if (sizes_ptr == nullptr && ndim > 0) {
    ET_LOG(
        Error,
        "aoti_torch_create_tensor_from_blob_v2 failed: sizes_ptr is null");
    return Error::InvalidArgument;
  }

  if (ret_new_tensor == nullptr) {
    ET_LOG(
        Error,
        "aoti_torch_create_tensor_from_blob_v2 failed: ret_new_tensor is null");
    return Error::InvalidArgument;
  }

  // Check that device_index is always 0
  if (device_index != 0) {
    ET_LOG(Error, "device_index must be 0, got: %d", device_index);
    return Error::InvalidArgument;
  }

  // Validate dtype using SupportedDTypes from utils.h
  AOTITorchError dtype_error = validate_dtype(dtype);
  if (dtype_error != Error::Ok) {
    return dtype_error;
  }

  // Storage offset must be 0 since from_blob cannot handle different offsets
  AOTITorchError storage_offset_error = validate_storage_offset(storage_offset);
  if (storage_offset_error != Error::Ok) {
    return storage_offset_error;
  }

  // Convert sizes to the format expected by ExecutorTorch using SizesType
  std::vector<executorch::aten::SizesType> sizes =
      convert_sizes_to_vector(ndim, sizes_ptr);

  // Convert strides using the common helper function with StridesType
  std::vector<executorch::aten::StridesType> strides =
      convert_strides_to_vector(ndim, sizes_ptr, strides_ptr);

  // Create ExecutorTorch tensor that wraps the existing memory
  // Note: We're NOT copying the data, just wrapping it
  auto tensor = executorch::extension::from_blob(
      data, // existing memory (don't copy!)
      sizes, // tensor dimensions
      strides, // tensor strides (allows different strides)
      dtype_to_scalar_type(dtype) // map int32_t dtype to ScalarType
  );

  if (!tensor) {
    ET_LOG(Error, "Failed to create tensor from blob");
    return Error::InvalidArgument;
  }

  // Store the tensor so it doesn't get destroyed
  tensors.insert(tensor);

  *ret_new_tensor = tensor.get();

  // Check if this memory address is already being tracked
  auto memory_it = memory_to_n_tensor.find(data);
  if (memory_it != memory_to_n_tensor.end()) {
    ET_LOG(
        Error,
        "Memory address %p is already being tracked by another tensor",
        data);
    return Error::InvalidArgument;
  }

  // Mark this memory as NOT_OWN since tensor created from blob never owns
  // memory
  memory_to_n_tensor[data] = NOT_OWN;

  return Error::Ok;
}

AOTITorchError aoti_torch_empty_strided(
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    Tensor** ret_new_tensor) {
  // Check that device_index is always 0
  if (device_index != 0) {
    ET_LOG(Error, "device_index must be 0, got: %d", device_index);
    return Error::InvalidArgument;
  }

  // This requires us to reserve CUDA memory and put it into a ETensor
  void* ptr;
  int64_t numel = 1;
  for (int64_t i = 0; i < ndim; i++) {
    numel *= sizes_ptr[i];
  }

  AOTITorchError dtype_error = validate_dtype(dtype);
  if (dtype_error != Error::Ok) {
    return dtype_error;
  }

  size_t element_size = dtype_to_element_size(dtype);
  if (element_size == 0) {
    ET_LOG(Error, "Invalid element size for dtype: %d", dtype);
    return Error::InvalidArgument;
  }
  int64_t nbytes = numel * element_size;

  if (device_type == 1) { // cuda
    ET_CUDA_CHECK_OR_RETURN_ERROR(cudaMallocManaged(&ptr, nbytes));
  } else if (device_type == 0) { // cpu
    // Ensure 16-byte alignment for CPU memory to match CUDA requirements
    int result = posix_memalign(&ptr, 16, nbytes);
    if (result != 0) {
      ET_LOG(Error, "Failed to allocate aligned CPU memory");
      return Error::MemoryAllocationFailed;
    }
    if (ptr == nullptr) {
      ET_LOG(Error, "Failed to call posix_memalign");
      return Error::MemoryAllocationFailed;
    }
  } else {
    ET_LOG(
        Error,
        "Need to implement empty_strided for non-CUDA non-CPU device type %d",
        device_type);
    return Error::NotImplemented;
  }

  // ETensor sizes
  auto sizes = convert_sizes_to_vector(ndim, sizes_ptr);

  // ETensor strides
  auto strides = convert_strides_to_vector(ndim, sizes_ptr, strides_ptr);

  // ETensor creation with dynamic shape support for edge cases
  auto tensor = executorch::extension::from_blob(
      ptr, sizes, strides, dtype_to_scalar_type(dtype));

  // Store the tensor so it doesn't get destroyed
  tensors.insert(tensor);
  *ret_new_tensor = tensor.get();

  // This tensor owns the memory it allocated, set reference count to 1
  memory_to_n_tensor[ptr] = 1;

  return Error::Ok;
}

void clear_all_tensors() {
  // Use aoti_torch_delete_tensor_object to properly delete each tensor
  // Note: We need to collect tensor pointers first since deletion modifies the
  // set
  auto old_tensors =
      std::move(tensors); // tensors is now empty and no need to copy
  for (const auto& tensor_shared : old_tensors) {
    aoti_torch_delete_tensor_object(tensor_shared.get());
  }

  // tensors set should now be empty, but ensure it's cleared
  tensors.clear();
}

AOTITorchError aoti_torch_delete_tensor_object(Tensor* tensor) {
  // Handle null tensor pointer
  if (tensor == nullptr) {
    ET_LOG(Error, "Cannot delete null tensor");
    return Error::InvalidArgument;
  }

  // Check if tensor exists in our tracking
  bool found_in_tensors = false;
  for (auto it = tensors.begin(); it != tensors.end(); ++it) {
    if (it->get() == tensor) {
      found_in_tensors = true;
      break;
    }
  }

  // If tensor not found in our tracking, it's invalid
  if (!found_in_tensors) {
    ET_LOG(Error, "Didn't find tensor %p", tensor);
    return Error::InvalidArgument;
  }

  // Find and delete the tensor
  for (auto it = tensors.begin(); it != tensors.end(); ++it) {
    if (it->get() == tensor) {
      // Get the tensor before erasing
      auto tensor_ptr = *it;
      void* data_ptr = tensor_ptr->mutable_data_ptr();

      // Find the reference count for this memory address
      auto memory_it = memory_to_n_tensor.find(data_ptr);
      if (memory_it != memory_to_n_tensor.end()) {
        int32_t ref_count = memory_it->second;

        if (ref_count == NOT_OWN) {
          // Tensor never owned the memory, skip freeing
          // Just remove tensor from tracking
          tensors.erase(it);
          return Error::Ok;
        } else if (ref_count == 1) {
          // Only current tensor using this memory, free it
          // Determine if it's GPU memory
          cudaPointerAttributes attributes{};
          ET_CUDA_CHECK_OR_RETURN_ERROR(
              cudaPointerGetAttributes(&attributes, data_ptr));

          if (attributes.type == cudaMemoryTypeManaged) {
            // This is CUDA managed memory - free with proper synchronization
            ET_CUDA_CHECK_OR_RETURN_ERROR(cudaDeviceSynchronize());
            ET_CUDA_CHECK_OR_RETURN_ERROR(cudaFree(data_ptr));
          } else {
            // This is CPU memory - free immediately
            free(data_ptr);
            data_ptr = nullptr;
          }

          // Remove from memory tracking
          memory_to_n_tensor.erase(memory_it);
        } else if (ref_count > 1) {
          // Other tensors still using this memory, just decrement count
          memory_to_n_tensor[data_ptr] = ref_count - 1;
        }
      } else {
        ET_LOG(Error, "Internal error: memory not found during deletion");
        return Error::Internal;
      }

      // Remove tensor from set (this will call the destructor if it's the last
      // reference)
      tensors.erase(it);
      return Error::Ok;
    }
  }

  // This should never be reached since we found it above
  ET_LOG(Error, "Internal error: tensor not found after validation");
  return Error::Internal;
}

AOTITorchError aoti_torch__reinterpret_tensor(
    Tensor* self,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    Tensor** ret_new_tensor) {
  // Validate input parameters first
  if (self == nullptr) {
    ET_LOG(Error, "aoti_torch__reinterpret_tensor failed: self tensor is null");
    return Error::InvalidArgument;
  }

  if (sizes_ptr == nullptr && ndim > 0) {
    ET_LOG(Error, "aoti_torch__reinterpret_tensor failed: sizes_ptr is null");
    return Error::InvalidArgument;
  }

  if (ret_new_tensor == nullptr) {
    ET_LOG(
        Error, "aoti_torch__reinterpret_tensor failed: ret_new_tensor is null");
    return Error::InvalidArgument;
  }

  // Check if storage_offset is not 0 - return error if not
  AOTITorchError storage_offset_error = validate_storage_offset(storage_offset);
  if (storage_offset_error != Error::Ok) {
    return storage_offset_error;
  }

  // Get the device info from the source tensor to perform device_index
  // validation
  int32_t device_type = 0;
  int32_t device_index = 0;
  AOTITorchError device_error = aoti_torch_get_device_type(self, &device_type);
  if (device_error != Error::Ok) {
    return device_error;
  }

  device_error = aoti_torch_get_device_index(self, &device_index);
  if (device_error != Error::Ok) {
    return device_error;
  }

  // Ensure device_index is always 0
  if (device_index != 0) {
    ET_LOG(Error, "device_index must be 0, got: %d", device_index);
    return Error::InvalidArgument;
  }

  // Get the dtype from the source tensor
  int32_t dtype = 0;
  AOTITorchError dtype_error = aoti_torch_get_dtype(self, &dtype);
  if (dtype_error != Error::Ok) {
    return dtype_error;
  }

  // Validate dtype using SupportedDTypes
  dtype_error = validate_dtype(dtype);
  if (dtype_error != Error::Ok) {
    return dtype_error;
  }

  // Get the original data pointer from the source tensor
  void* data_ptr = self->mutable_data_ptr();
  if (data_ptr == nullptr) {
    ET_LOG(Error, "Source tensor has null data pointer");
    return Error::InvalidArgument;
  }

  // Check if the given memory is in the map, if not return error
  auto memory_it = memory_to_n_tensor.find(data_ptr);
  if (memory_it == memory_to_n_tensor.end()) {
    ET_LOG(
        Error,
        "Memory address %p is not being tracked by reference counting system",
        data_ptr);
    return Error::InvalidArgument;
  }

  // Convert sizes using utility function from utils.h
  std::vector<SizesType> sizes = convert_sizes_to_vector(ndim, sizes_ptr);

  // Convert strides using utility function from utils.h
  std::vector<StridesType> strides =
      convert_strides_to_vector(ndim, sizes_ptr, strides_ptr);

  // Create new tensor view that reinterprets the same memory with different
  // shape/strides This creates a view, not a copy - the data pointer is shared
  std::shared_ptr<Tensor> tensor = executorch::extension::from_blob(
      data_ptr, // Reuse the same memory from source tensor
      sizes, // New sizes with explicit SizesType
      strides, // New strides with explicit StridesType
      dtype_to_scalar_type(dtype) // Convert dtype with explicit type casting
  );

  if (!tensor) {
    ET_LOG(Error, "Failed to create reinterpreted tensor view");
    return Error::InvalidArgument;
  }

  // Store the tensor so it doesn't get destroyed
  tensors.insert(tensor);

  *ret_new_tensor = tensor.get();

  // Increment the reference count for this memory address only if it is owned
  // by tensor
  memory_to_n_tensor[data_ptr] = memory_to_n_tensor[data_ptr] == NOT_OWN
      ? NOT_OWN
      : memory_to_n_tensor[data_ptr] + 1;

  return Error::Ok;
}
  
} // extern "C"

} // namespace cuda
} // namespace backends
} // namespace executorch
