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
std::unordered_map<Tensor*, bool> is_tensor_own_memory;

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
  is_tensor_own_memory[tensor.get()] = false;

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
  is_tensor_own_memory[tensor.get()] = true;

  ET_LOG(Debug, "aoti_torch_empty_strided: successfull");
  return Error::Ok;
}

AOTITorchError aoti_torch_delete_tensor_object(AOTITensorHandle tensor) {
  ET_LOG(Debug, "aoti_torch_delete_tensor_object: entered");
  // Find tensor in the set
  for (auto it = tensors.begin(); it != tensors.end(); ++it) {
    if (it->get() == tensor) {
      auto tensor_ptr = *it;

      // Check ownership before cleaning up
      auto ownership_it = is_tensor_own_memory.find(tensor);
      bool owns_memory = (ownership_it != is_tensor_own_memory.end())
          ? ownership_it->second
          : false;

      // Clean up ownership metadata
      is_tensor_own_memory.erase(tensor);

      if (owns_memory) {
        // et tensor owns the memory; need to free it manually
        void* data_ptr = tensor_ptr->mutable_data_ptr();

        // Check if it's Metal GPU memory
        if (metal_is_device_pointer(data_ptr)) {
          // This is Metal GPU memory - the Metal helper will handle cleanup
          // Metal buffers are automatically managed by ARC when the buffer is
          // released
          tensors.erase(it);
          ET_LOG(
              Debug,
              "aoti_torch_delete_tensor_object: successfull (Metal GPU memory)");
          return Error::Ok;
        }

        // This is CPU memory - free immediately
        free(data_ptr);
      }
      // else: Don't free memory since the tensor doesn't own it

      // Remove from set (this will call the destructor if it's the last
      // reference)
      tensors.erase(it);
      ET_LOG(
          Debug, "aoti_torch_delete_tensor_object: successfull (CPU memory)");
      return Error::Ok;
    }
  }
  ET_LOG(Error, "Didn't find tensor %p", tensor);
  return Error::InvalidArgument;
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

  // Get the dtype from the source tensor
  int32_t dtype = 0;
  ET_CHECK_OK_OR_RETURN_ERROR(aoti_torch_get_dtype(self, &dtype));

  // Validate dtype using SupportedDTypes
  ET_CHECK_OK_OR_RETURN_ERROR(validate_dtype(dtype));

  int32_t device_type = 0;
  ET_CHECK_OK_OR_RETURN_ERROR(aoti_torch_get_device_type(self, &device_type));

  int32_t device_index = 0;
  ET_CHECK_OK_OR_RETURN_ERROR(aoti_torch_get_device_index(self, &device_index));

  // Get the base data pointer from the source tensor
  void* base_data_ptr = self->mutable_data_ptr();
  ET_CHECK_OR_RETURN_ERROR(
      base_data_ptr != nullptr,
      InvalidArgument,
      "Source tensor has null data pointer");

  // Calculate new tensor size in elements for logging
  int64_t new_numel = 1;
  for (int64_t i = 0; i < ndim; i++) {
    new_numel *= sizes_ptr[i];
  }

  ET_LOG(
      Debug,
      "aoti_torch__reinterpret_tensor: base_data_ptr=%p, new_numel=%lld, storage_offset=%lld",
      base_data_ptr,
      new_numel,
      storage_offset);

  // Create a new tensor view that shares the same underlying storage
  // This is the correct way to implement reinterpret_tensor - as a view, not a
  // copy
  AOTITorchError create_err = aoti_torch_create_tensor_from_blob_v2(
      base_data_ptr, // Same underlying data pointer
      ndim, // New dimensions
      sizes_ptr, // New sizes
      strides_ptr, // New strides
      storage_offset, // Storage offset (will be handled properly now)
      dtype,
      device_type,
      device_index,
      ret_new_tensor,
      0, // layout (default)
      nullptr, // opaque_metadata
      0 // opaque_metadata_size
  );

  if (create_err != Error::Ok) {
    ET_LOG(Error, "failed to create reinterpreted tensor view");
    return create_err;
  }

  ET_LOG(Debug, "aoti_torch__reinterpret_tensor: successfull");
  return Error::Ok;
}

// Cleanup function for clearing global state
void cleanup_memory() {
  is_tensor_own_memory.clear();
  if (!tensors.empty()) {
    ET_LOG(Error, "Warning: tensors not empty during cleanup");
  }

  // Clean up Metal resources
  metal_cleanup_resources();
}

} // extern "C"

} // namespace metal
} // namespace backends
} // namespace executorch
