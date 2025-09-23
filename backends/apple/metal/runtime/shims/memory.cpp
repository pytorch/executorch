/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "memory.h"
#include <executorch/runtime/platform/log.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "tensor_attribute.h"
#include "utils.h"

#include <cstdint> // Ensure we have int64_t, int32_t definitions

#include "metal_helper.h"

namespace executorch {
namespace backends {
namespace aoti {

namespace { // Internal namespace for utility functions

// Utility function to log array values as error msg in format [val1, val2, ...]
// For use with pointer-based arrays (e.g., int64_t* strides, int64_t* sizes)
void et_error_log_array_values(
    const int64_t* values,
    int64_t count,
    const std::string& name = "values") {
  if (count <= 0) {
    ET_LOG(Error, "%s: empty array", name.c_str());
    return;
  }

  // Build array string representation
  std::string array_str = "[";
  for (int64_t i = 0; i < count; i++) {
    array_str += std::to_string(values[i]);
    if (i < count - 1) {
      array_str += ", ";
    }
  }
  array_str += "]";

  ET_LOG(Error, "%s: %s", name.c_str(), array_str.c_str());
}

// Check if tensor is in contiguous memory format (NCHW for 4D tensors)
// Contiguous format means strides decrease from left to right:
// For NCHW: strides = [C*H*W, H*W, W, 1]
bool is_tensor_contiguous(
    int64_t ndim,
    const int64_t* sizes,
    const int64_t* strides) {
  int64_t expected_stride = 1;
  for (int i = ndim - 1; i >= 0; i--) {
    if (strides[i] != expected_stride) {
      return false;
    }
    expected_stride *= sizes[i];
  }
  return true;
}

// Check if tensor is in channels-last format (NHWC for 4D tensors)
// Channels-last format for 4D: strides = [H*W*C, 1, W*C, C]
bool is_tensor_channels_last(
    int64_t ndim,
    const int64_t* sizes,
    const int64_t* strides) {
  if (ndim != 4) {
    return false; // Channels-last only defined for 4D tensors
  }

  int64_t N = sizes[0], C = sizes[1], H = sizes[2], W = sizes[3];

  // Check NHWC format: strides = [H*W*C, 1, W*C, C]
  // Handle edge cases where dimensions might be 1
  return (strides[0] == H * W * C || N <= 1) && (strides[1] == 1 || C <= 1) &&
      (strides[2] == W * C || H <= 1) && (strides[3] == C || W <= 1);
}

} // anonymous namespace

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
  // Only float32 tensors are supported
  AOTITorchError dtype_error = validate_dtype(dtype);
  if (dtype_error != Error::Ok) {
    return dtype_error;
  }

  // Handle storage offset by adjusting the data pointer
  void* adjusted_data = static_cast<char*>(data) + (storage_offset * dtype_to_element_size(dtype));

  ET_LOG(Debug, "aoti_torch_create_tensor_from_blob_v2: original_data=%p, storage_offset=%ld, element_size=%zu, adjusted_data=%p",
         data, storage_offset, dtype_to_element_size(dtype), adjusted_data);

  // Convert sizes to the format expected by ExecutorTorch
  std::vector<int32_t> sizes(ndim);
  for (int i = 0; i < ndim; i++) {
    sizes[i] = static_cast<int32_t>(sizes_ptr[i]);
  }

  // Check tensor format - support both contiguous and custom strides
  bool is_contiguous = is_tensor_contiguous(ndim, sizes_ptr, strides_ptr);
  bool is_chlast = is_tensor_channels_last(ndim, sizes_ptr, strides_ptr);

  if (is_contiguous) {
    ET_LOG(Debug, "aoti_torch_create_tensor_from_blob_v2: Creating contiguous tensor");
    // Use simple make_tensor_ptr for contiguous tensors
    auto tensor = executorch::extension::make_tensor_ptr(
        sizes, // tensor dimensions
        adjusted_data, // adjusted memory pointer (with storage offset applied)
        dtype_to_scalar_type(dtype) // map int32_t dtype to ScalarType
    );

    if (!tensor) {
      ET_LOG(Error, "Failed to create contiguous tensor from blob");
      return Error::InvalidArgument;
    }

    // Store the tensor so it doesn't get destroyed
    tensors.insert(tensor);
    *ret_new_tensor = tensor.get();
    is_tensor_own_memory[tensor.get()] = false;

  } else {
    ET_LOG(Debug, "aoti_torch_create_tensor_from_blob_v2: Creating tensor with custom strides");

    // Convert strides to the format expected by ExecutorTorch, handling broadcast strides
    std::vector<int32_t> strides(ndim);
    std::vector<int32_t> adjusted_sizes(ndim);
    bool has_broadcast_dims = false;

    for (int i = 0; i < ndim; i++) {
      int64_t original_stride = strides_ptr[i];
      int64_t size = sizes_ptr[i];

      if (original_stride == 0) {
        // Stride 0 means broadcasting - convert to size 1 with stride 1
        // This tells ExecutorTorch this dimension doesn't actually consume memory
        strides[i] = 1;
        adjusted_sizes[i] = 1;  // Broadcast dimensions have effective size 1
        has_broadcast_dims = true;
        ET_LOG(Debug, "aoti_torch_create_tensor_from_blob_v2: Converted broadcast dim %d: stride 0->1, size %ld->1", i, size);
      } else {
        strides[i] = static_cast<int32_t>(original_stride);
        adjusted_sizes[i] = static_cast<int32_t>(size);
      }
    }

    // For tensors with broadcast dimensions, we need to create a smaller underlying tensor
    // and then expand it to the desired size
    if (has_broadcast_dims) {
      ET_LOG(Debug, "aoti_torch_create_tensor_from_blob_v2: Creating base tensor for broadcasting");

      // Create base tensor with adjusted (smaller) sizes
      auto base_tensor = executorch::extension::from_blob(
          adjusted_data, // adjusted memory pointer
          adjusted_sizes, // adjusted (smaller) sizes
          strides,       // adjusted strides
          dtype_to_scalar_type(dtype) // scalar type
      );

      if (!base_tensor) {
        ET_LOG(Error, "Failed to create base tensor for broadcasting");
        return Error::InvalidArgument;
      }

      // Now expand the base tensor to the desired broadcast size
      // For ExecutorTorch, we need to create a view that represents the broadcast
      // Since ExecutorTorch may not support expand() directly, we'll need to handle this differently

      // For now, let's try to create a simple view using the original sizes
      // but with adjusted memory layout understanding

      auto tensor = executorch::extension::make_tensor_ptr(
          sizes, // original requested sizes (with broadcast dimensions)
          adjusted_data, // same data pointer
          dtype_to_scalar_type(dtype) // scalar type
      );

      if (!tensor) {
        ET_LOG(Error, "Failed to create broadcast tensor view");
        return Error::InvalidArgument;
      }

      // Store the tensor so it doesn't get destroyed
      tensors.insert(tensor);
      *ret_new_tensor = tensor.get();
      is_tensor_own_memory[tensor.get()] = false;

    } else {
      // No broadcast dimensions - use normal strided tensor creation
      auto tensor = executorch::extension::from_blob(
          adjusted_data, // adjusted memory pointer
          sizes,         // tensor dimensions
          strides,       // custom strides
          dtype_to_scalar_type(dtype) // scalar type
      );

      if (!tensor) {
        ET_LOG(Error, "Failed to create strided tensor from blob");
        return Error::InvalidArgument;
      }

      // Store the tensor so it doesn't get destroyed
      tensors.insert(tensor);
      *ret_new_tensor = tensor.get();
      is_tensor_own_memory[tensor.get()] = false;
    }
  }

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
  // This requires us to reserve device memory and put it into a ETensor
  void* ptr;
  int64_t numel = 1;
  for (int i = 0; i < ndim; i++) {
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

  if (device_type == 2) { // Metal/MPS
    ptr = metal_allocate_buffer(nbytes);
    if (!ptr) {
      ET_LOG(Error, "Failed to allocate %ld bytes on Metal device", nbytes);
      return Error::MemoryAllocationFailed;
    }
  } else if (device_type == 0) { // cpu
    // Ensure 16-byte alignment for CPU memory to match device requirements
    int result = posix_memalign(&ptr, 16, nbytes);
    if (result != 0) {
      ET_LOG(Error, "Failed to allocate aligned CPU memory");
      return Error::MemoryAllocationFailed;
    }
    if (ptr == nullptr) {
      ET_LOG(Error, "Failed to call posix_memalign");
      return Error::MemoryAllocationFailed;
    }
    ET_LOG(Debug, "Allocated %ld bytes on CPU", nbytes);
  } else {
    ET_LOG(
        Error,
        "Unsupported device type %d",
        device_type);
    return Error::NotImplemented;
  }

  // ETensor sizes
  std::vector<int32_t> sizes(ndim);
  for (int i = 0; i < ndim; i++) {
    sizes[i] = sizes_ptr[i];
  }

  // ETensor strides
  std::vector<int32_t> strides(ndim);
  if (strides_ptr != nullptr) {
    // Use provided strides
    for (int i = 0; i < ndim; i++) {
      strides[i] = strides_ptr[i];
    }
  } else {
    // Calculate strides from sizes, assume it is in contiguous memory format
    strides[ndim - 1] = 1; // Last dimension has stride 1
    for (int i = ndim - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * sizes_ptr[i + 1];
    }
  }

  // ETensor creation
  auto tensor = executorch::extension::from_blob(ptr, sizes, strides);

  // Store the tensor so it doesn't get destroyed
  tensors.insert(tensor);
  *ret_new_tensor = tensor.get();
  is_tensor_own_memory[tensor.get()] = true;

  return Error::Ok;
}

AOTITorchError aoti_torch_delete_tensor_object(AOTITensorHandle tensor) {
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
          // Metal buffers are automatically managed by ARC when the buffer is released
          tensors.erase(it);
          return Error::Ok;
        }

        // This is CPU memory - free immediately
        free(data_ptr);
      }
      // else: Don't free memory since the tensor doesn't own it

      // Remove from set (this will call the destructor if it's the last
      // reference)
      tensors.erase(it);
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
  // assert same dim for now
  if (self->dim() != src->dim()) {
    ET_LOG(
        Error,
        "dimension mismatch. self.dim()=%d, src.dim()=%d",
        self->dim(),
        src->dim());
    return Error::InvalidArgument;
  }

  // only support float32 for now
  int32_t self_dtype, src_dtype;
  aoti_torch_get_dtype(self, &self_dtype);
  aoti_torch_get_dtype(src, &src_dtype);

  AOTITorchError self_dtype_error = validate_dtype(self_dtype);
  if (self_dtype_error != Error::Ok) {
    return self_dtype_error;
  }

  AOTITorchError src_dtype_error = validate_dtype(src_dtype);
  if (src_dtype_error != Error::Ok) {
    return src_dtype_error;
  }

  // Get stride information for layout validation
  int64_t* self_strides;
  int64_t* src_strides;
  aoti_torch_get_strides(self, &self_strides);
  aoti_torch_get_strides(src, &src_strides);

  int64_t* self_sizes;
  int64_t* src_sizes;
  aoti_torch_get_sizes(self, &self_sizes);
  aoti_torch_get_sizes(src, &src_sizes);

  // Check if tensors have the same tensor schema (sizes, strides, dtype)
  bool same_schema = true;

  // Check schema match
  for (int i = 0; i < self->dim(); i++) {
    if (self_sizes[i] != src_sizes[i] || self_strides[i] != src_strides[i]) {
      same_schema = false;
      break;
    }
  }

  // Declare layout variables for both cases
  bool self_is_contiguous = true;
  bool src_is_contiguous = true;
  bool self_is_channels_last = false;
  bool src_is_channels_last = false;

  // For same schema, we don't need to check memory formats - just use direct
  // copy
  if (!same_schema) {
    // Different strides: check memory format and only support contiguous <->
    // channels-last conversion

    // Check if contiguous (strides decrease from left to right)
    self_is_contiguous =
        is_tensor_contiguous(self->dim(), self_sizes, self_strides);

    src_is_contiguous =
        is_tensor_contiguous(src->dim(), src_sizes, src_strides);

    // Check if channels-last (4D: NHWC format)
    if (!self_is_contiguous) {
      self_is_channels_last =
          is_tensor_channels_last(self->dim(), self_sizes, self_strides);
    }

    if (!src_is_contiguous) {
      src_is_channels_last =
          is_tensor_channels_last(src->dim(), src_sizes, src_strides);
    }

    // Validate layout assumptions only when schemas differ
    if (!self_is_contiguous && !self_is_channels_last) {
      ET_LOG(
          Error,
          "self tensor must be contiguous or channels-last for stride conversion");
      et_error_log_array_values(self_strides, self->dim(), "self strides");
      et_error_log_array_values(self_sizes, self->dim(), "self_sizes");
      return Error::InvalidArgument;
    }

    if (!src_is_contiguous && !src_is_channels_last) {
      ET_LOG(
          Error,
          "src tensor must be contiguous or channels-last for stride conversion");
      et_error_log_array_values(src_strides, src->dim(), "self strides");
      et_error_log_array_values(src_sizes, src->dim(), "src_sizes");
      return Error::InvalidArgument;
    }
  }

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

  size_t total_bytes = src->nbytes();

  if (same_schema) {
    // Simple copy since layouts match
    if (srcIsDevice && dstIsDevice) {
      // For Metal, use the helper function for device-to-device copy
      int result = metal_copy_memory(
          self->mutable_data_ptr(),
          src->data_ptr(),
          total_bytes,
          true,  // src is device
          true); // dst is device
      if (result != 0) {
        ET_LOG(Error, "Failed to copy from Metal device to device");
        return Error::Internal;
      }
    } else if (srcIsDevice && !dstIsDevice) {
      // For Metal, use the helper function for device-to-host copy
      int result = metal_copy_memory(
          self->mutable_data_ptr(),
          src->data_ptr(),
          total_bytes,
          true,   // src is device
          false); // dst is host
      if (result != 0) {
        ET_LOG(Error, "Failed to copy from Metal device to host");
        return Error::Internal;
      }
    } else if (!srcIsDevice && dstIsDevice) {
      // For Metal, use the helper function for host-to-device copy
      int result = metal_copy_memory(
          self->mutable_data_ptr(),
          src->data_ptr(),
          total_bytes,
          false, // src is host
          true); // dst is device
      if (result != 0) {
        ET_LOG(Error, "Failed to copy from host to Metal device");
        return Error::Internal;
      }
    } else {
      std::memcpy(self->mutable_data_ptr(), src->data_ptr(), total_bytes);
    }
  } else {
    // Layout conversion needed (contiguous <-> channels-last)

    if (self->dim() != 4) {
      ET_LOG(Error, "Layout conversion only supported for 4D tensors");
      return Error::NotImplemented;
    }

    // Get data to host for processing
    size_t total_elements = total_bytes / sizeof(float);
    float* src_host_data = nullptr;
    float* dst_host_data = nullptr;
    bool need_free_src = false;
    bool need_free_dst = false;

    if (srcIsDevice) {
      src_host_data = new float[total_elements];
      int result = metal_copy_memory(
          src_host_data,
          src->data_ptr(),
          total_bytes,
          true,   // src is device
          false); // dst is host
      if (result != 0) {
        ET_LOG(Error, "Failed to copy src from Metal device to host");
        delete[] src_host_data;
        return Error::Internal;
      }
      need_free_src = true;
    } else {
      src_host_data = static_cast<float*>(const_cast<void*>(src->data_ptr()));
    }

    if (dstIsDevice) {
      dst_host_data = new float[total_elements];
      need_free_dst = true;
    } else {
      dst_host_data = static_cast<float*>(self->mutable_data_ptr());
    }

    // Perform layout conversion (4D NCHW <-> NHWC)
    int64_t N = self_sizes[0], C = self_sizes[1], H = self_sizes[2],
            W = self_sizes[3];

    for (int64_t n = 0; n < N; n++) {
      for (int64_t c = 0; c < C; c++) {
        for (int64_t h = 0; h < H; h++) {
          for (int64_t w = 0; w < W; w++) {
            size_t src_offset, dst_offset;

            if (src_is_contiguous) {
              // Source is NCHW
              src_offset = n * C * H * W + c * H * W + h * W + w;
            } else {
              // Source is NHWC
              src_offset = n * H * W * C + h * W * C + w * C + c;
            }

            if (self_is_contiguous) {
              // Destination is NCHW
              dst_offset = n * C * H * W + c * H * W + h * W + w;
            } else {
              // Destination is NHWC
              dst_offset = n * H * W * C + h * W * C + w * C + c;
            }

            dst_host_data[dst_offset] = src_host_data[src_offset];
          }
        }
      }
    }

    // Copy result back to device if needed
    if (dstIsDevice) {
      int result = metal_copy_memory(
          self->mutable_data_ptr(),
          dst_host_data,
          total_bytes,
          false, // src is host
          true); // dst is device
      if (result != 0) {
        ET_LOG(Error, "Failed to copy result to Metal device");
        // Clean up temporary buffers before returning
        if (need_free_src)
          delete[] src_host_data;
        if (need_free_dst)
          delete[] dst_host_data;
        return Error::Internal;
      }
    }

    // Clean up temporary buffers
    if (need_free_src)
      delete[] src_host_data;
    if (need_free_dst)
      delete[] dst_host_data;
  }

  // Verify the copy by checking first element
  float src_first, dst_first;
  if (srcIsDevice) {
    // For Metal, we can directly access since it uses shared memory
    src_first = static_cast<const float*>(src->data_ptr())[0];
  } else {
    src_first = static_cast<const float*>(src->data_ptr())[0];
  }

  if (dstIsDevice) {
    // For Metal, we can directly access since it uses shared memory
    dst_first = static_cast<const float*>(self->data_ptr())[0];
  } else {
    dst_first = static_cast<const float*>(self->data_ptr())[0];
  }

  return Error::Ok;
}

AOTITorchError aoti_torch__reinterpret_tensor(
    AOTITensorHandle self,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    AOTITensorHandle* ret_new_tensor) {

  ET_LOG(Debug, "aoti_torch__reinterpret_tensor: self->dim()=%d, ndim=%ld, storage_offset=%ld",
         self->dim(), ndim, storage_offset);

  // Get tensor properties from the input tensor
  int32_t dtype;
  AOTITorchError dtype_err = aoti_torch_get_dtype(self, &dtype);
  if (dtype_err != Error::Ok) {
    ET_LOG(Error, "failed to get dtype from input tensor");
    return dtype_err;
  }

  int32_t device_type;
  AOTITorchError device_type_err = aoti_torch_get_device_type(self, &device_type);
  if (device_type_err != Error::Ok) {
    ET_LOG(Error, "failed to get device_type from input tensor");
    return device_type_err;
  }

  int32_t device_index;
  AOTITorchError device_index_err = aoti_torch_get_device_index(self, &device_index);
  if (device_index_err != Error::Ok) {
    ET_LOG(Error, "failed to get device_index from input tensor");
    return device_index_err;
  }

  // Get the base data pointer from the source tensor
  void* base_data_ptr = self->mutable_data_ptr();

  // Calculate new tensor size in elements for logging
  int64_t new_numel = 1;
  for (int64_t i = 0; i < ndim; i++) {
    new_numel *= sizes_ptr[i];
  }

  ET_LOG(Debug, "aoti_torch__reinterpret_tensor: base_data_ptr=%p, new_numel=%ld, storage_offset=%ld",
         base_data_ptr, new_numel, storage_offset);

  // Create a new tensor view that shares the same underlying storage
  // This is the correct way to implement reinterpret_tensor - as a view, not a copy
  AOTITorchError create_err = aoti_torch_create_tensor_from_blob_v2(
      base_data_ptr,       // Same underlying data pointer
      ndim,                // New dimensions
      sizes_ptr,           // New sizes
      strides_ptr,         // New strides
      storage_offset,      // Storage offset (will be handled properly now)
      dtype,
      device_type,
      device_index,
      ret_new_tensor,
      0,                   // layout (default)
      nullptr,             // opaque_metadata
      0                    // opaque_metadata_size
  );

  if (create_err != Error::Ok) {
    ET_LOG(Error, "failed to create reinterpreted tensor view");
    return create_err;
  }

  ET_LOG(Debug, "aoti_torch__reinterpret_tensor: Successfully created tensor view");
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

} // namespace aoti
} // namespace backends
} // namespace executorch
