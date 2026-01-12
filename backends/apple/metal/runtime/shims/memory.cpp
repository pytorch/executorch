/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/aoti/utils.h>
#include <executorch/backends/apple/metal/runtime/shims/et_metal.h>
#include <executorch/backends/apple/metal/runtime/shims/memory.h>
#include <executorch/backends/apple/metal/runtime/shims/tensor_attribute.h>
#include <executorch/backends/apple/metal/runtime/shims/utils.h>
#include <executorch/runtime/platform/log.h>
#include <cstdint> // Ensure we have int64_t, int32_t definitions
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace executorch {
namespace backends {
namespace metal {

// Import all from aoti namespace
using namespace executorch::backends::aoti;

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
    AOTITensorHandle* ret_new_tensor,
    int32_t layout,
    const uint8_t* opaque_metadata,
    int64_t opaque_metadata_size) {
  ET_LOG(Debug, "aoti_torch_create_tensor_from_blob_v2: entered");

  (void)device_type;
  (void)opaque_metadata;
  (void)layout;
  (void)opaque_metadata_size;

  // Validate input parameters first
  ET_CHECK_OR_RETURN_ERROR(
      data != nullptr,
      InvalidArgument,
      "aoti_torch_create_tensor_from_blob_v2 failed: data pointer is null");

  ET_CHECK_OR_RETURN_ERROR(
      !(sizes_ptr == nullptr && ndim > 0),
      InvalidArgument,
      "aoti_torch_create_tensor_from_blob_v2 failed: sizes_ptr is null");

  ET_CHECK_OR_RETURN_ERROR(
      ret_new_tensor != nullptr,
      InvalidArgument,
      "aoti_torch_create_tensor_from_blob_v2 failed: ret_new_tensor is null");

  ET_CHECK_OK_OR_RETURN_ERROR(validate_dtype(dtype));

  // Handle storage offset by adjusting the data pointer
  void* adjusted_data = static_cast<char*>(data) +
      (storage_offset * dtype_to_element_size(dtype));

  ET_LOG(
      Debug,
      "aoti_torch_create_tensor_from_blob_v2: original_data=%p, storage_offset=%lld, element_size=%zu, adjusted_data=%p",
      data,
      storage_offset,
      dtype_to_element_size(dtype),
      adjusted_data);

  // ETensor sizes
  auto sizes = convert_sizes_to_vector(ndim, sizes_ptr);

  // ETensor strides
  auto strides = convert_strides_to_vector(ndim, sizes_ptr, strides_ptr);

  // Log if the tensor is contiguous
  if (is_contiguous_tensor(sizes, strides)) {
    ET_LOG(Debug, "aoti_torch_create_tensor_from_blob_v2: contiguous tensor");
  } else {
    ET_LOG(
        Debug, "aoti_torch_create_tensor_from_blob_v2: non-contiguous tensor");
  }

  // ETensor creation
  // Note: We're NOT copying the data, just wrapping it
  auto tensor = executorch::extension::from_blob(
      adjusted_data, sizes, strides, dtype_to_scalar_type(dtype));

  ET_CHECK_OR_RETURN_ERROR(
      tensor != nullptr, InvalidArgument, "Failed to create tensor from blob");

  // Store the tensor so it doesn't get destroyed
  tensors.insert(tensor);
  *ret_new_tensor = tensor.get();

  // Check if this memory address is already being tracked
  auto memory_it = memory_to_n_tensor.find(adjusted_data);
  ET_CHECK_OR_RETURN_ERROR(
      memory_it == memory_to_n_tensor.end(),
      InvalidArgument,
      "Memory address %p is already being tracked by another tensor",
      adjusted_data);

  // Mark this memory as NOT_OWN since tensor created from blob never owns
  // memory
  memory_to_n_tensor[adjusted_data] = NOT_OWN;

  ET_LOG(Debug, "aoti_torch_create_tensor_from_blob_v2: successfull");
  return Error::Ok;
}

AOTITorchError aoti_torch_empty_strided(
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AOTITensorHandle* ret_new_tensor) {
  ET_LOG(Debug, "aoti_torch_empty_strided: entered");

  // This requires us to reserve device memory and put it into a ETensor
  void* ptr;
  int64_t numel = 1;
  for (int i = 0; i < ndim; i++) {
    numel *= sizes_ptr[i];
  }

  ET_CHECK_OK_OR_RETURN_ERROR(validate_dtype(dtype));

  size_t element_size = dtype_to_element_size(dtype);
  ET_CHECK_OR_RETURN_ERROR(
      element_size != 0,
      InvalidArgument,
      "Invalid element size for dtype: %d",
      dtype);
  int64_t nbytes = numel * element_size;

  int32_t mps_device_type = aoti_torch_device_type_mps(); // Returns 13
  if (device_type == mps_device_type) {
    ptr = metal_allocate_buffer(nbytes);
    if (!ptr) {
      ET_LOG(Error, "Failed to allocate %lld bytes on Metal device", nbytes);
      return Error::MemoryAllocationFailed;
    }
  } else if (device_type == 0) { // cpu
    // Ensure 16-byte alignment for CPU memory to match device requirements
    int result = posix_memalign(&ptr, 16, nbytes);
    ET_CHECK_OR_RETURN_ERROR(
        result == 0,
        MemoryAllocationFailed,
        "Failed to allocate aligned CPU memory");
    ET_CHECK_OR_RETURN_ERROR(
        ptr != nullptr,
        MemoryAllocationFailed,
        "Failed to call posix_memalign");
    ET_LOG(Debug, "Allocated %lld bytes on CPU", nbytes);
  } else {
    ET_CHECK_OR_RETURN_ERROR(
        false,
        NotImplemented,
        "Need to implement empty_strided for non-CUDA non-CPU device type %d",
        device_type);
  }

  // ETensor sizes
  auto sizes = convert_sizes_to_vector(ndim, sizes_ptr);

  // ETensor strides
  auto strides = convert_strides_to_vector(ndim, sizes_ptr, strides_ptr);

  // Log if the tensor is contiguous
  if (is_contiguous_tensor(sizes, strides)) {
    ET_LOG(Debug, "aoti_torch_empty_strided: contiguous tensor");
  } else {
    ET_LOG(Debug, "aoti_torch_empty_strided: non-contiguous tensor");
  }

  // ETensor creation
  // Note: We're NOT copying the data, just wrapping it
  executorch::aten::ScalarType scalar_type = dtype_to_scalar_type(dtype);
  auto tensor =
      executorch::extension::from_blob(ptr, sizes, strides, scalar_type);

  // Store the tensor so it doesn't get destroyed
  tensors.insert(tensor);
  *ret_new_tensor = tensor.get();

  // This tensor owns the memory it allocated, set reference count to 1
  memory_to_n_tensor[ptr] = 1;

  ET_LOG(Debug, "aoti_torch_empty_strided: successfull");
  return Error::Ok;
}

AOTITorchError aoti_torch_delete_tensor_object(AOTITensorHandle tensor) {
  ET_LOG(Debug, "aoti_torch_delete_tensor_object: entered");

  // Handle null tensor pointer
  if (tensor == nullptr) {
    ET_LOG(Debug, "aoti_torch_delete_tensor_object: null tensor");
    return Error::Ok;
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
  ET_CHECK_OR_RETURN_ERROR(
      found_in_tensors, InvalidArgument, "Didn't find tensor %p", tensor);

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
          ET_LOG(
              Debug,
              "aoti_torch_delete_tensor_object: tensor doesn't own memory, skipping free");
          return Error::Ok;
        } else if (ref_count == 1) {
          // Only current tensor using this memory, free it
          // Check if it's Metal GPU memory
          if (metal_is_device_pointer(data_ptr)) {
            metal_deallocate_buffer(data_ptr);
          } else {
            // This is CPU memory - free immediately
            free(data_ptr);
            data_ptr = nullptr;
            ET_LOG(
                Debug, "aoti_torch_delete_tensor_object: freeing CPU memory");
          }

          // Remove from memory tracking
          memory_to_n_tensor.erase(memory_it);
        } else if (ref_count > 1) {
          // Other tensors still using this memory, just decrement count
          memory_to_n_tensor[data_ptr] = ref_count - 1;
          ET_LOG(
              Debug,
              "aoti_torch_delete_tensor_object: decremented ref count from %d to %d",
              ref_count,
              ref_count - 1);
        }
      } else {
        ET_CHECK_OR_RETURN_ERROR(
            false,
            Internal,
            "Internal error: memory not found during deletion");
      }

      // Remove tensor from set (this will call the destructor if it's the last
      // reference)
      tensors.erase(it);
      ET_LOG(Debug, "aoti_torch_delete_tensor_object: successfull");
      return Error::Ok;
    }
  }

  // This should never be reached since we found it above
  ET_CHECK_OR_RETURN_ERROR(
      false, Internal, "Internal error: tensor not found after validation");
}

AOTITorchError aoti_torch_copy_(
    AOTITensorHandle self,
    AOTITensorHandle src,
    int32_t non_blocking) {
  ET_LOG(Debug, "aoti_torch_copy_: entered");

  (void)non_blocking;

  // Check for null pointers first
  ET_CHECK_OR_RETURN_ERROR(
      self != nullptr,
      InvalidArgument,
      "aoti_torch_copy_ failed: self tensor is null");

  ET_CHECK_OR_RETURN_ERROR(
      src != nullptr,
      InvalidArgument,
      "aoti_torch_copy_ failed: src tensor is null");

  // Get dtype information and validate compatibility
  int32_t self_dtype, src_dtype;
  aoti_torch_get_dtype(self, &self_dtype);
  aoti_torch_get_dtype(src, &src_dtype);

  ET_CHECK_OK_OR_RETURN_ERROR(validate_dtype(self_dtype));

  ET_CHECK_OK_OR_RETURN_ERROR(validate_dtype(src_dtype));

  // Check dtype compatibility - both tensors must have the same dtype
  ET_CHECK_OR_RETURN_ERROR(
      self_dtype == src_dtype,
      InvalidArgument,
      "dtype mismatch. self.dtype=%d, src.dtype=%d. aoti_torch_copy_ requires same dtypes",
      self_dtype,
      src_dtype);

  // Check total number of elements compatibility (PyTorch copy_ behavior)
  int64_t self_numel = self->numel();
  int64_t src_numel = src->numel();

  ET_CHECK_OR_RETURN_ERROR(
      self_numel == src_numel,
      InvalidArgument,
      "numel mismatch. self.numel()=%ld, src.numel()=%ld",
      self_numel,
      src_numel);

  // Get tensor metadata
  int64_t* self_strides;
  int64_t* src_strides;
  aoti_torch_get_strides(self, &self_strides);
  aoti_torch_get_strides(src, &src_strides);

  int64_t* self_sizes;
  int64_t* src_sizes;
  aoti_torch_get_sizes(self, &self_sizes);
  aoti_torch_get_sizes(src, &src_sizes);

  // Determine device locations
  bool srcIsDevice = false;
  bool dstIsDevice = false;

  // Check if pointers are Metal device pointers
  if (!srcIsDevice) {
    srcIsDevice = metal_is_device_pointer(const_cast<void*>(src->data_ptr()));
  }
  if (!dstIsDevice) {
    dstIsDevice = metal_is_device_pointer(self->mutable_data_ptr());
  }

  // Check if tensors have the same schema (sizes, strides, dtype) for fast path
  // TODO: This should be improved to catch cases like (4, 1, 5) -> (4, 5)
  bool same_schema = true;
  for (int i = 0; i < self->dim(); i++) {
    if (self_strides[i] != src_strides[i]) {
      same_schema = false;
      break;
    }
  }

  size_t total_bytes = src->nbytes();
  int64_t total_elements = self->numel();

  if (same_schema) {
    int result = metal_copy_memory(
        self->mutable_data_ptr(),
        src->data_ptr(),
        total_bytes,
        srcIsDevice,
        dstIsDevice);
    if (result != 0) {
      ET_LOG(Error, "metal_copy_memory failed with status %d", result);
      return Error::Internal;
    }
  } else {
    ET_LOG(Error, "Layout conversion not supported");
    return Error::NotImplemented;
  }

  ET_LOG(Debug, "aoti_torch_copy_: successfull");
  return Error::Ok;
}

AOTITorchError aoti_torch__reinterpret_tensor(
    AOTITensorHandle self,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    AOTITensorHandle* ret_new_tensor) {
  ET_LOG(Debug, "aoti_torch__reinterpret_tensor: entered");

  // Validate input parameters first
  ET_CHECK_OR_RETURN_ERROR(
      self != nullptr,
      InvalidArgument,
      "aoti_torch__reinterpret_tensor failed: self tensor is null");

  ET_CHECK_OR_RETURN_ERROR(
      !(sizes_ptr == nullptr && ndim > 0),
      InvalidArgument,
      "aoti_torch__reinterpret_tensor failed: sizes_ptr is null");

  ET_CHECK_OR_RETURN_ERROR(
      ret_new_tensor != nullptr,
      InvalidArgument,
      "aoti_torch__reinterpret_tensor failed: ret_new_tensor is null");

  // Get the device info from the source tensor to perform device_index
  // validation
  int32_t device_type = 0;
  int32_t device_index = 0;
  ET_CHECK_OK_OR_RETURN_ERROR(aoti_torch_get_device_type(self, &device_type));

  ET_CHECK_OK_OR_RETURN_ERROR(aoti_torch_get_device_index(self, &device_index));

  // Ensure device_index is always 0
  ET_CHECK_OR_RETURN_ERROR(
      device_index == 0,
      InvalidArgument,
      "device_index must be 0, got: %d",
      device_index);

  // Get the dtype from the source tensor
  int32_t dtype = 0;
  ET_CHECK_OK_OR_RETURN_ERROR(aoti_torch_get_dtype(self, &dtype));

  // Validate dtype using SupportedDTypes
  ET_CHECK_OK_OR_RETURN_ERROR(validate_dtype(dtype));

  // Get the original data pointer from the source tensor
  void* data_ptr = self->mutable_data_ptr();
  ET_CHECK_OR_RETURN_ERROR(
      data_ptr != nullptr,
      InvalidArgument,
      "Source tensor has null data pointer");

  // Check if the given memory is in the map, if not return error
  auto memory_it = memory_to_n_tensor.find(data_ptr);
  ET_CHECK_OR_RETURN_ERROR(
      memory_it != memory_to_n_tensor.end(),
      InvalidArgument,
      "Memory address %p is not being tracked by reference counting system",
      data_ptr);

  // Handle storage offset by adjusting the data pointer
  void* adjusted_data = static_cast<char*>(data_ptr) +
      (storage_offset * dtype_to_element_size(dtype));

  // Convert sizes using utility function from utils.h
  std::vector<aten::SizesType> sizes = convert_sizes_to_vector(ndim, sizes_ptr);

  // Convert strides using utility function from utils.h
  std::vector<aten::StridesType> strides =
      convert_strides_to_vector(ndim, sizes_ptr, strides_ptr);

  // Create new tensor view that reinterprets the same memory with different
  // shape/strides This creates a view, not a copy - the data pointer is shared
  std::shared_ptr<Tensor> tensor = executorch::extension::from_blob(
      adjusted_data, // Use adjusted data pointer with storage offset applied
      sizes, // New sizes with explicit SizesType
      strides, // New strides with explicit StridesType
      dtype_to_scalar_type(dtype) // Convert dtype with explicit type casting
  );

  ET_CHECK_OR_RETURN_ERROR(
      tensor != nullptr,
      InvalidArgument,
      "Failed to create reinterpreted tensor view");

  // Store the tensor so it doesn't get destroyed
  tensors.insert(tensor);

  *ret_new_tensor = tensor.get();

  if (adjusted_data != data_ptr) {
    ET_LOG(
        Debug,
        "aoti_torch__reinterpret_tensor: Adjusted original_data=%p, storage_offset=%lld, element_size=%zu, adjusted_data=%p",
        data_ptr,
        storage_offset,
        dtype_to_element_size(dtype),
        adjusted_data);

    metal_buffer_nocopy(adjusted_data, tensor->nbytes(), true);
  }

  // Increment the reference count for this memory address only if it is owned
  // by tensor
  memory_to_n_tensor[adjusted_data] = memory_to_n_tensor[adjusted_data] == NOT_OWN
      ? NOT_OWN
      : memory_to_n_tensor[adjusted_data] + 1;

  ET_LOG(Debug, "aoti_torch__reinterpret_tensor: successfull");
  return Error::Ok;
}

AOTITorchError aoti_torch_new_tensor_handle(
    Tensor* orig_handle,
    Tensor** new_handle) {
  (void)orig_handle;
  (void)new_handle;
  throw std::runtime_error("Not implemented");
  return Error::Internal;
}

// Cleanup function for clearing global state
void cleanup_memory() {
  // Use aoti_torch_delete_tensor_object to properly delete each tensor
  // Note: We need to collect tensor pointers first since deletion modifies the
  // set
  std::vector<Tensor*> tensor_ptrs;
  tensor_ptrs.reserve(tensors.size());
  for (const auto& tensor_shared : tensors) {
    tensor_ptrs.push_back(tensor_shared.get());
  }

  // Now delete each tensor - this will modify the global tensors set
  for (Tensor* tensor_ptr : tensor_ptrs) {
    aoti_torch_delete_tensor_object(tensor_ptr);
  }

  // tensors set should now be empty, but ensure it's cleared
  tensors.clear();

  // Clean up Metal resources
  metal_cleanup_resources();

  ET_LOG(Info, "Cleared all tensors and Metal resources");
}

} // extern "C"

} // namespace metal
} // namespace backends
} // namespace executorch
