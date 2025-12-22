/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/aoti/common_shims.h>
#include <executorch/backends/aoti/utils.h>
#include <executorch/backends/cuda/runtime/platform/platform.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/backends/cuda/runtime/shims/tensor_attribute.h>
#include <executorch/backends/cuda/runtime/tensor/tensor_maker.h>
#include <executorch/backends/cuda/runtime/utils.h>
#include <executorch/runtime/platform/log.h>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace executorch::backends::cuda {

using executorch::aten::SizesType;
using executorch::aten::StridesType;
using executorch::backends::aoti::aoti_torch_dtype_bool;
using executorch::backends::aoti::aoti_torch_get_device_index;
using executorch::backends::aoti::aoti_torch_get_dtype;
using executorch::backends::aoti::aoti_torch_get_sizes;
using executorch::backends::aoti::aoti_torch_get_strides;
using executorch::backends::aoti::convert_sizes_to_vector;
using executorch::backends::aoti::convert_strides_to_vector;
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

namespace {

// Calculate linear offset from strides and indices
int64_t calculate_linear_offset(
    const int64_t* indices,
    const int64_t* strides,
    int64_t ndim) {
  int64_t offset = 0;
  for (int64_t i = 0; i < ndim; ++i) {
    offset += indices[i] * strides[i];
  }
  return offset;
}

// Convert linear index to multi-dimensional indices based on sizes
void linear_to_indices(
    int64_t linear_idx,
    const int64_t* sizes,
    int64_t ndim,
    int64_t* indices) {
  for (int64_t i = ndim - 1; i >= 0; --i) {
    indices[i] = linear_idx % sizes[i];
    linear_idx /= sizes[i];
  }
}

// Generic pointwise copy function that handles arbitrary strides
template <typename T>
AOTITorchError pointwise_copy_generic(
    T* dst_data,
    const T* src_data,
    const int64_t* dst_sizes,
    const int64_t* dst_strides,
    const int64_t* src_sizes,
    const int64_t* src_strides,
    int64_t dst_ndim,
    int64_t src_ndim,
    int64_t total_elements) {
  std::vector<int64_t> dst_indices(dst_ndim);
  std::vector<int64_t> src_indices(src_ndim);

  for (int64_t linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
    // Convert linear index to multi-dimensional indices for both tensors
    linear_to_indices(linear_idx, dst_sizes, dst_ndim, dst_indices.data());
    linear_to_indices(linear_idx, src_sizes, src_ndim, src_indices.data());

    // Calculate offsets for both source and destination
    int64_t src_offset =
        calculate_linear_offset(src_indices.data(), src_strides, src_ndim);
    int64_t dst_offset =
        calculate_linear_offset(dst_indices.data(), dst_strides, dst_ndim);

    // Copy element
    dst_data[dst_offset] = src_data[src_offset];
  }

  return Error::Ok;
}

} // anonymous namespace

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

  // Check that device_index is always 0
  ET_CHECK_OR_RETURN_ERROR(
      device_index == 0,
      InvalidArgument,
      "device_index must be 0, got: %d",
      device_index);

  // Validate dtype using SupportedDTypes from utils.h
  ET_CHECK_OK_OR_RETURN_ERROR(validate_dtype(dtype));

  // Storage offset must be 0 since from_blob cannot handle different offsets
  ET_CHECK_OK_OR_RETURN_ERROR(validate_storage_offset(storage_offset));

  // Verify that data pointer location matches the requested device_type
  cudaPointerAttributes data_attributes{};
  ET_CUDA_CHECK_OR_RETURN_ERROR(
      cudaPointerGetAttributes(&data_attributes, data));

  bool data_is_on_device = data_attributes.type == cudaMemoryTypeDevice;
  bool data_is_on_host = data_attributes.type == cudaMemoryTypeHost ||
      data_attributes.type == cudaMemoryTypeUnregistered;
  bool requested_device =
      device_type == static_cast<int32_t>(SupportedDevices::CUDA);
  bool requested_cpu =
      device_type == static_cast<int32_t>(SupportedDevices::CPU);

  // Error if data location doesn't match requested device type
  ET_CHECK_OR_RETURN_ERROR(
      !(data_is_on_device && requested_cpu),
      InvalidArgument,
      "aoti_torch_create_tensor_from_blob_v2 failed: data pointer %p is on CUDA "
      "but device_type is CPU. Data must be on CPU for CPU tensors.",
      data);

  ET_CHECK_OR_RETURN_ERROR(
      !(data_is_on_host && requested_device),
      InvalidArgument,
      "aoti_torch_create_tensor_from_blob_v2 failed: data pointer %p is on CPU "
      "but device_type is CUDA. Data must be on GPU for CUDA tensors.",
      data);

  // Convert sizes to the format expected by ExecutorTorch using SizesType
  std::vector<executorch::aten::SizesType> sizes =
      convert_sizes_to_vector(ndim, sizes_ptr);

  // Convert strides using the common helper function with StridesType
  std::vector<executorch::aten::StridesType> strides =
      convert_strides_to_vector(ndim, sizes_ptr, strides_ptr);

  // Create ExecutorTorch tensor that wraps the existing memory
  // Note: We're NOT copying the data, just wrapping it
  // Using CUDA-specific tensor maker that supports incontiguous tensors
  auto tensor = make_tensor(
      sizes, // tensor dimensions
      data, // existing memory (don't copy!)
      {}, // dim_order (empty, will be auto-generated)
      strides, // tensor strides (allows different strides)
      dtype_to_scalar_type(dtype) // map int32_t dtype to ScalarType
  );

  ET_CHECK_OR_RETURN_ERROR(
      tensor != nullptr, InvalidArgument, "Failed to create tensor from blob");

  // Store the tensor so it doesn't get destroyed
  tensors.insert(tensor);

  *ret_new_tensor = tensor.get();

  // Check if this memory address is already being tracked
  auto memory_it = memory_to_n_tensor.find(data);
  ET_CHECK_OR_RETURN_ERROR(
      memory_it == memory_to_n_tensor.end(),
      InvalidArgument,
      "Memory address %p is already being tracked by another tensor",
      data);

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
  ET_CHECK_OR_RETURN_ERROR(
      device_index == 0,
      InvalidArgument,
      "device_index must be 0, got: %d",
      device_index);

  // This requires us to reserve CUDA memory and put it into a ETensor
  void* ptr;

  ET_CHECK_OK_OR_RETURN_ERROR(validate_dtype(dtype));

  size_t element_size = dtype_to_element_size(dtype);
  ET_CHECK_OR_RETURN_ERROR(
      element_size != 0,
      InvalidArgument,
      "Invalid element size for dtype: %d",
      dtype);

  // Calculate storage size based on strides, matching PyTorch's behavior
  // This is critical when sizes and strides don't match the expected contiguous
  // layout Reference: PyTorch's computeStorageNbytes in EmptyTensor.cpp
  int64_t storage_size = 1; // storage offset (0) + 1
  for (int64_t i = 0; i < ndim; i++) {
    if (sizes_ptr[i] == 0) {
      storage_size = 0;
      break;
    }
    // For each dimension, add stride[i] * (size[i] - 1)
    // This gives us the maximum offset in that dimension
    int64_t stride_i = (strides_ptr != nullptr) ? strides_ptr[i] : 1;
    if (strides_ptr == nullptr) {
      // Calculate contiguous stride if not provided
      for (int64_t j = i + 1; j < ndim; j++) {
        stride_i *= sizes_ptr[j];
      }
    }
    storage_size += stride_i * (sizes_ptr[i] - 1);
  }
  int64_t nbytes = storage_size * element_size;

  if (device_type == static_cast<int32_t>(SupportedDevices::CUDA)) {
    ET_CUDA_CHECK_OR_RETURN_ERROR(
        cudaMallocAsync(&ptr, static_cast<size_t>(nbytes), cudaStreamDefault));
  } else if (device_type == static_cast<int32_t>(SupportedDevices::CPU)) {
    // Ensure 16-byte alignment for CPU memory to match CUDA requirements
    ptr = aligned_alloc(16, nbytes);
    ET_CHECK_OR_RETURN_ERROR(
        ptr != nullptr,
        MemoryAllocationFailed,
        "Failed to allocate aligned CPU memory");
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

  // ETensor creation with dynamic shape support for edge cases
  // Using CUDA-specific tensor maker that supports incontiguous tensors
  auto tensor = make_tensor(
      sizes,
      ptr,
      {}, // dim_order (empty, will be auto-generated)
      strides,
      dtype_to_scalar_type(dtype));

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

  ET_LOG(Info, "Cleared all tensors");
}

void clear_memory_tracking() {
  memory_to_n_tensor.clear();
  ET_LOG(Info, "Cleared memory tracking map");
}

AOTITorchError aoti_torch_delete_tensor_object(Tensor* tensor) {
  // Handle null tensor pointer
  ET_CHECK_OR_RETURN_ERROR(
      tensor != nullptr, InvalidArgument, "Cannot delete null tensor");

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
          return Error::Ok;
        } else if (ref_count == 1) {
          // Only current tensor using this memory, free it
          // Determine if it's GPU memory
          cudaPointerAttributes attributes{};
          ET_CUDA_CHECK_OR_RETURN_ERROR(
              cudaPointerGetAttributes(&attributes, data_ptr));

          if (attributes.type == cudaMemoryTypeDevice) {
            ET_CUDA_CHECK_OR_RETURN_ERROR(
                cudaFreeAsync(data_ptr, cudaStreamDefault));
          } else {
            ET_CHECK_OR_RETURN_ERROR(
                attributes.type != cudaMemoryTypeManaged,
                Internal,
                "Expected host memory but got managed!")
            // This is CPU memory - free immediately
            aligned_free(data_ptr);
            data_ptr = nullptr;
          }

          // Remove from memory tracking
          memory_to_n_tensor.erase(memory_it);
        } else if (ref_count > 1) {
          // Other tensors still using this memory, just decrement count
          memory_to_n_tensor[data_ptr] = ref_count - 1;
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
      return Error::Ok;
    }
  }

  // This should never be reached since we found it above
  ET_CHECK_OR_RETURN_ERROR(
      false, Internal, "Internal error: tensor not found after validation");
}

AOTITorchError
aoti_torch_copy_(Tensor* self, Tensor* src, int32_t non_blocking) {
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
  cudaPointerAttributes srcAttributes{};
  cudaPointerAttributes dstAttributes{};

  ET_CUDA_CHECK_OR_RETURN_ERROR(
      cudaPointerGetAttributes(&srcAttributes, src->data_ptr()));

  ET_CUDA_CHECK_OR_RETURN_ERROR(
      cudaPointerGetAttributes(&dstAttributes, self->data_ptr()));

  bool srcIsDevice = srcAttributes.type == cudaMemoryTypeDevice;
  bool dstIsDevice = dstAttributes.type == cudaMemoryTypeDevice;

  // Check if tensors have the same schema (sizes, strides, dtype) for fast path
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
    // Fast path: Direct memory copy since layouts match exactly
    if (srcIsDevice && dstIsDevice) {
      ET_CUDA_CHECK_OR_RETURN_ERROR(cudaMemcpy(
          self->mutable_data_ptr(),
          src->data_ptr(),
          total_bytes,
          cudaMemcpyDeviceToDevice));
    } else if (srcIsDevice && !dstIsDevice) {
      ET_CUDA_CHECK_OR_RETURN_ERROR(cudaMemcpy(
          self->mutable_data_ptr(),
          src->data_ptr(),
          total_bytes,
          cudaMemcpyDeviceToHost));
    } else if (!srcIsDevice && dstIsDevice) {
      ET_CUDA_CHECK_OR_RETURN_ERROR(cudaMemcpy(
          self->mutable_data_ptr(),
          src->data_ptr(),
          total_bytes,
          cudaMemcpyHostToDevice));
    } else {
      std::memcpy(self->mutable_data_ptr(), src->data_ptr(), total_bytes);
    }
  } else {
    // Fallback path: Pointwise copy with stride-aware indexing
    // This handles arbitrary tensor layouts and strides

    size_t element_size = dtype_to_element_size(self_dtype);
    ET_CHECK_OR_RETURN_ERROR(
        element_size != 0,
        InvalidArgument,
        "Invalid element size for dtype: %d",
        self_dtype);

    // Allocate temporary host memory for GPU tensors
    float* src_host_data = nullptr;
    float* dst_host_data = nullptr;
    bool need_free_src = false;
    bool need_free_dst = false;

    if (srcIsDevice) {
      src_host_data =
          static_cast<float*>(malloc(total_elements * sizeof(float)));
      ET_CHECK_OR_RETURN_ERROR(
          src_host_data != nullptr,
          MemoryAllocationFailed,
          "Failed to allocate memory for src_host_data");
      ET_CUDA_CHECK_OR_RETURN_ERROR(cudaMemcpy(
          src_host_data, src->data_ptr(), total_bytes, cudaMemcpyDeviceToHost));
      need_free_src = true;
    } else {
      src_host_data = static_cast<float*>(src->data_ptr());
    }

    if (dstIsDevice) {
      dst_host_data =
          static_cast<float*>(malloc(total_elements * sizeof(float)));
      if (dst_host_data == nullptr) {
        if (need_free_src) {
          free(src_host_data);
        }
        ET_CHECK_OR_RETURN_ERROR(
            false,
            MemoryAllocationFailed,
            "Failed to allocate memory for dst_host_data");
      }
      need_free_dst = true;
    } else {
      dst_host_data = static_cast<float*>(self->mutable_data_ptr());
    }

    // Perform pointwise copy with stride calculation
    AOTITorchError copy_err = pointwise_copy_generic(
        dst_host_data,
        src_host_data,
        self_sizes,
        self_strides,
        src_sizes,
        src_strides,
        self->dim(),
        src->dim(),
        total_elements);

    if (copy_err != Error::Ok) {
      // Clean up temporary buffers before returning
      if (need_free_src) {
        free(src_host_data);
      }
      if (need_free_dst) {
        free(dst_host_data);
      }
      return copy_err;
    }

    // Copy result back to device if needed
    if (dstIsDevice) {
      ET_CUDA_CHECK_OR_RETURN_ERROR(cudaMemcpy(
          self->mutable_data_ptr(),
          dst_host_data,
          total_bytes,
          cudaMemcpyHostToDevice));
    }

    // Clean up temporary buffers
    if (need_free_src) {
      free(src_host_data);
    }
    if (need_free_dst) {
      free(dst_host_data);
    }
  }

  return Error::Ok;
}

AOTITorchError aoti_torch__reinterpret_tensor(
    Tensor* self,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    Tensor** ret_new_tensor) {
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

  // Check if storage_offset is not 0 - return error if not
  ET_CHECK_OK_OR_RETURN_ERROR(validate_storage_offset(storage_offset));

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

  // Convert sizes using utility function from utils.h
  std::vector<SizesType> sizes = convert_sizes_to_vector(ndim, sizes_ptr);

  // Convert strides using utility function from utils.h
  std::vector<StridesType> strides =
      convert_strides_to_vector(ndim, sizes_ptr, strides_ptr);

  // Create new tensor view that reinterprets the same memory with different
  // shape/strides This creates a view, not a copy - the data pointer is shared
  // Using CUDA-specific tensor maker that supports incontiguous tensors
  std::shared_ptr<Tensor> tensor = make_tensor(
      sizes, // New sizes with explicit SizesType
      data_ptr, // Reuse the same memory from source tensor
      {}, // dim_order (empty, will be auto-generated)
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

  // Increment the reference count for this memory address only if it is owned
  // by tensor
  memory_to_n_tensor[data_ptr] = memory_to_n_tensor[data_ptr] == NOT_OWN
      ? NOT_OWN
      : memory_to_n_tensor[data_ptr] + 1;

  return Error::Ok;
}

AOTITorchError aoti_torch_new_tensor_handle(
    Tensor* orig_handle,
    Tensor** new_handle) {
  // Validate input parameters
  ET_CHECK_OR_RETURN_ERROR(
      orig_handle != nullptr,
      InvalidArgument,
      "aoti_torch_new_tensor_handle failed: orig_handle is null");

  ET_CHECK_OR_RETURN_ERROR(
      new_handle != nullptr,
      InvalidArgument,
      "aoti_torch_new_tensor_handle failed: new_handle is null");

  // Get metadata from the original tensor
  int64_t* sizes_ptr;
  int64_t* strides_ptr;
  int32_t dtype;
  int32_t device_type;
  int32_t device_index;

  ET_CHECK_OK_OR_RETURN_ERROR(aoti_torch_get_sizes(orig_handle, &sizes_ptr));
  ET_CHECK_OK_OR_RETURN_ERROR(
      aoti_torch_get_strides(orig_handle, &strides_ptr));
  ET_CHECK_OK_OR_RETURN_ERROR(aoti_torch_get_dtype(orig_handle, &dtype));
  ET_CHECK_OK_OR_RETURN_ERROR(
      aoti_torch_get_device_type(orig_handle, &device_type));
  ET_CHECK_OK_OR_RETURN_ERROR(
      aoti_torch_get_device_index(orig_handle, &device_index));

  int64_t ndim = orig_handle->dim();

  // Validate dtype
  ET_CHECK_OK_OR_RETURN_ERROR(validate_dtype(dtype));

  // Ensure device_index is always 0
  ET_CHECK_OR_RETURN_ERROR(
      device_index == 0,
      InvalidArgument,
      "device_index must be 0, got: %d",
      device_index);

  // Get the original data pointer from the source tensor
  void* data_ptr = orig_handle->mutable_data_ptr();
  ET_CHECK_OR_RETURN_ERROR(
      data_ptr != nullptr,
      InvalidArgument,
      "Source tensor has null data pointer");

  // Check if the given memory is in the map
  auto memory_it = memory_to_n_tensor.find(data_ptr);
  ET_CHECK_OR_RETURN_ERROR(
      memory_it != memory_to_n_tensor.end(),
      InvalidArgument,
      "Memory address %p is not being tracked by reference counting system",
      data_ptr);

  // Convert sizes and strides to vectors
  std::vector<SizesType> sizes = convert_sizes_to_vector(ndim, sizes_ptr);
  std::vector<StridesType> strides =
      convert_strides_to_vector(ndim, sizes_ptr, strides_ptr);

  // Create new tensor that shares the same memory as the original
  // This is similar to PyTorch's Tensor copy constructor - creates a new
  // tensor object that shares the same underlying storage
  std::shared_ptr<Tensor> tensor = make_tensor(
      sizes, // Same sizes as original
      data_ptr, // Share the same memory from source tensor
      {}, // dim_order (empty, will be auto-generated)
      strides, // Same strides as original
      dtype_to_scalar_type(dtype) // Same dtype as original
  );

  ET_CHECK_OR_RETURN_ERROR(
      tensor != nullptr, InvalidArgument, "Failed to create new tensor handle");

  // Store the tensor so it doesn't get destroyed
  tensors.insert(tensor);

  *new_handle = tensor.get();

  // Increment the reference count for this memory address only if it is owned
  // by tensor
  memory_to_n_tensor[data_ptr] = memory_to_n_tensor[data_ptr] == NOT_OWN
      ? NOT_OWN
      : memory_to_n_tensor[data_ptr] + 1;

  return Error::Ok;
}

AOTITorchError aoti_torch_item_bool(Tensor* tensor, bool* ret_value) {
  // Validate input parameters
  ET_CHECK_OR_RETURN_ERROR(
      tensor != nullptr,
      InvalidArgument,
      "aoti_torch_item_bool failed: tensor is null");

  ET_CHECK_OR_RETURN_ERROR(
      ret_value != nullptr,
      InvalidArgument,
      "aoti_torch_item_bool failed: ret_value is null");

  // Validate that tensor dtype is bool
  int32_t dtype;
  ET_CHECK_OK_OR_RETURN_ERROR(aoti_torch_get_dtype(tensor, &dtype));

  ET_CHECK_OR_RETURN_ERROR(
      dtype == aoti_torch_dtype_bool(),
      InvalidArgument,
      "aoti_torch_item_bool failed: tensor dtype is not bool (got %d)",
      dtype);

  // Get the data pointer
  const void* data_ptr = tensor->const_data_ptr();
  ET_CHECK_OR_RETURN_ERROR(
      data_ptr != nullptr,
      InvalidArgument,
      "aoti_torch_item_bool failed: tensor data pointer is null");

  // Check if tensor is on CUDA or CPU
  cudaPointerAttributes attributes{};
  ET_CUDA_CHECK_OR_RETURN_ERROR(
      cudaPointerGetAttributes(&attributes, data_ptr));

  if (attributes.type == cudaMemoryTypeDevice) {
    // CUDA memory case: copy from device to host
    bool device_value;
    ET_CUDA_CHECK_OR_RETURN_ERROR(cudaMemcpy(
        &device_value, data_ptr, sizeof(bool), cudaMemcpyDeviceToHost));
    *ret_value = device_value;
  } else {
    // CPU memory case: direct access
    const bool* bool_ptr = static_cast<const bool*>(data_ptr);
    *ret_value = *bool_ptr;
  }

  return Error::Ok;
}

AOTITorchError aoti_torch_assign_tensors_out(Tensor* src, Tensor** ret_dst) {
  // Validate input parameters
  ET_CHECK_OR_RETURN_ERROR(
      src != nullptr,
      InvalidArgument,
      "aoti_torch_assign_tensors_out failed: src is null");

  ET_CHECK_OR_RETURN_ERROR(
      ret_dst != nullptr,
      InvalidArgument,
      "aoti_torch_assign_tensors_out failed: ret_dst is null");

  // Get the data pointer from the source tensor
  void* data_ptr = src->mutable_data_ptr();
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

  // Get dtype from source tensor
  int32_t dtype = 0;
  ET_CHECK_OK_OR_RETURN_ERROR(aoti_torch_get_dtype(src, &dtype));

  // Get sizes and strides from source tensor
  int64_t* sizes_ptr;
  int64_t* strides_ptr;
  ET_CHECK_OK_OR_RETURN_ERROR(aoti_torch_get_sizes(src, &sizes_ptr));
  ET_CHECK_OK_OR_RETURN_ERROR(aoti_torch_get_strides(src, &strides_ptr));

  int64_t ndim = src->dim();

  // Convert to vectors
  std::vector<SizesType> sizes = convert_sizes_to_vector(ndim, sizes_ptr);
  std::vector<StridesType> strides =
      convert_strides_to_vector(ndim, sizes_ptr, strides_ptr);

  // Create new tensor view that shares the same memory as source tensor
  std::shared_ptr<Tensor> tensor = make_tensor(
      sizes,
      data_ptr, // Share the same memory from source tensor
      {}, // dim_order (empty, will be auto-generated)
      strides,
      dtype_to_scalar_type(dtype));

  ET_CHECK_OR_RETURN_ERROR(
      tensor != nullptr,
      InvalidArgument,
      "Failed to create tensor view in aoti_torch_assign_tensors_out");

  // Store the tensor so it doesn't get destroyed
  tensors.insert(tensor);

  *ret_dst = tensor.get();

  // Increment the reference count for this memory address only if it is owned
  // by tensor
  memory_to_n_tensor[data_ptr] = memory_to_n_tensor[data_ptr] == NOT_OWN
      ? NOT_OWN
      : memory_to_n_tensor[data_ptr] + 1;

  return Error::Ok;
}
} // extern "C"

} // namespace executorch::backends::cuda
