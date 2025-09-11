/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "memory.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstdlib> // For posix_memalign
#include <cstring>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "tensor_attribute.h"
#include "utils.h"

namespace executorch {
namespace backends {
namespace aoti {

namespace { // Internal namespace for utility functions

// Utility function to print array values in format [val1, val2, ...]
// For use with pointer-based arrays (e.g., int64_t* strides, int64_t* sizes)
template <typename ValueType>
void print_array_values(
    const ValueType* values,
    int64_t count,
    const std::string& name = "values") {
  std::cout << name << ": [";
  for (int i = 0; i < count; i++) {
    std::cout << values[i] << (i < count - 1 ? ", " : "");
  }
  std::cout << "]" << std::endl;
}

// Version 1: For use with int64_t sizes (e.g., from blob creation functions)
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

// Check if tensor is in contiguous memory format (NCHW for 4D tensors) for
// int32_t sizes
bool is_tensor_contiguous(
    int64_t ndim,
    const int32_t* sizes,
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
bool is_tensor_channels_last(
    int64_t ndim,
    const int32_t* sizes,
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

  // Storage offset must always be 0
  AOTITorchError storage_offset_error = validate_storage_offset(storage_offset);
  if (storage_offset_error != Error::Ok) {
    return storage_offset_error;
  }

  // Convert sizes to the format expected by ExecutorTorch
  std::vector<int32_t> sizes(ndim);
  for (int i = 0; i < ndim; i++) {
    sizes[i] = static_cast<int32_t>(sizes_ptr[i]);
  }

  // check the tensor format
  // Only support contiguous format for now
  if (!is_tensor_contiguous(ndim, sizes_ptr, strides_ptr)) {
    std::cout
        << "aoti_torch_create_tensor_from_blob_v2 failed since input stride is not in contiguous format. Return with Error"
        << std::endl;
    return Error::InvalidArgument;
  }

  // Since storage_offset is guaranteed to be 0, use data pointer directly
  void* adjusted_data = data;

  // Create ExecutorTorch tensor that wraps the existing memory
  // Note: We're NOT copying the data, just wrapping it
  auto tensor = executorch::extension::make_tensor_ptr(
      sizes, // tensor dimensions
      adjusted_data, // existing memory (don't copy!)
      executorch::aten::ScalarType::Float // only supported dtype
  );

  if (!tensor) {
    std::cerr << "Failed to create tensor from blob" << std::endl;
    return Error::InvalidArgument;
  }

  // Store the tensor so it doesn't get destroyed
  tensors.insert(tensor);

  *ret_new_tensor = tensor.get();
  is_tensor_own_memory[tensor.get()] = false;

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
  // This requires us to reserve CUDA memory and put it into a ETensor
  void* ptr;
  int64_t numel = 1;
  for (int i = 0; i < ndim; i++) {
    numel *= sizes_ptr[i];
  }

  AOTITorchError dtype_error = validate_dtype(dtype);
  if (dtype_error != Error::Ok) {
    return dtype_error;
  }

  int64_t nbytes = numel * 4;

  if (device_type == 1) { // cuda
    cudaError_t err = cudaMalloc(&ptr, nbytes);
    if (err != cudaSuccess) {
      std::cout << "failed to allocate " << nbytes
                << " error: " << cudaGetErrorString(err) << std::endl;
      throw std::runtime_error("Failed to call cudaMalloc");
    }
  } else if (device_type == 0) { // cpu
    // Ensure 16-byte alignment for CPU memory to match CUDA requirements
    // do we need to do this in cuda backend?
    int result = posix_memalign(&ptr, 16, nbytes);
    if (result != 0) {
      throw std::runtime_error("Failed to allocate aligned CPU memory");
    }
    if (ptr == nullptr) {
      throw std::runtime_error("Failed to call posix_memalign");
    }
  } else {
    throw std::runtime_error(
        "Need to implement empty_strided for non-CUDA non-CPU");
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
  // Check ownership before cleaning up metadata
  auto ownership_it = is_tensor_own_memory.find(tensor);
  bool owns_memory = (ownership_it != is_tensor_own_memory.end())
      ? ownership_it->second
      : false;

  // Clean up ALL metadata maps immediately to prevent use-after-free
  tensor_to_sizes.erase(tensor);
  tensor_to_strides.erase(tensor);
  is_tensor_own_memory.erase(tensor);

  if (!owns_memory) {
    // Don't free memory since the tensor doesn't own it
    return Error::Ok;
  }

  for (auto it = tensors.begin(); it != tensors.end(); ++it) {
    if (it->get() == tensor) {
      // Get the tensor before erasing
      auto tensor_ptr = *it;

      void* data_ptr = tensor_ptr->mutable_data_ptr();

      // Determine if it's GPU memory
      cudaPointerAttributes attributes;
      cudaError_t err = cudaPointerGetAttributes(&attributes, data_ptr);

      // et tensor does not own data; need to free them manually.
      if (err == cudaSuccess && attributes.type == cudaMemoryTypeDevice) {
        // This is GPU memory - free with proper synchronization
        cudaDeviceSynchronize(); // Wait for all operations to complete BEFORE
                                 // freeing
        cudaFree(data_ptr);
      } else {
        // This is CPU memory - free immediately
        free(data_ptr);
      }
      // Remove from set (this will call the destructor if it's the last
      // reference)
      tensors.erase(it);
      return Error::Ok;
    }
  }
  std::cout << "Error: Didn't find tensor " << tensor << std::endl;
  return Error::InvalidArgument;
}

void checkCudaError(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    std::cerr << "Error: " << msg << " (" << cudaGetErrorString(err) << ")"
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

AOTITorchError aoti_torch_copy_(
    AOTITensorHandle self,
    AOTITensorHandle src,
    int32_t non_blocking) {
  std::cout << "aoti_torch_copy_ called: self=" << self << ", src=" << src
            << std::endl;

  // assert same dim for now
  if (self->dim() != src->dim()) {
    std::cout << "Error: dimension mismatch. self.dim()=" << self->dim()
              << ", src.dim()=" << src->dim() << std::endl;
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

  // Check sizes match
  for (int i = 0; i < self->dim(); i++) {
    if (self_sizes[i] != src_sizes[i]) {
      same_schema = false;
      break;
    }
  }

  // Check strides match (only if sizes match)
  if (same_schema) {
    for (int i = 0; i < self->dim(); i++) {
      if (self_strides[i] != src_strides[i]) {
        same_schema = false;
        break;
      }
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
      std::cout
          << "Error: self tensor must be contiguous or channels-last for stride conversion. "
          << std::endl;
      print_array_values(self_strides, self->dim(), "self strides");
      print_array_values(self_sizes, self->dim(), "self_sizes");
      return Error::InvalidArgument;
    }

    if (!src_is_contiguous && !src_is_channels_last) {
      std::cout
          << "Error: src tensor must be contiguous or channels-last for stride conversion."
          << std::endl;
      print_array_values(src_strides, src->dim(), "self strides");
      print_array_values(src_sizes, src->dim(), "src_sizes");
      return Error::InvalidArgument;
    }
  }

  // Determine device locations
  cudaPointerAttributes srcAttributes, dstAttributes;
  cudaError_t err;

  err = cudaPointerGetAttributes(&srcAttributes, src->data_ptr());
  checkCudaError(err, "Failed to get source pointer attributes");

  err = cudaPointerGetAttributes(&dstAttributes, self->data_ptr());
  checkCudaError(err, "Failed to get destination pointer attributes");

  bool srcIsDevice = srcAttributes.type == cudaMemoryTypeDevice;
  bool dstIsDevice = dstAttributes.type == cudaMemoryTypeDevice;

  size_t total_bytes = src->nbytes();

  if (same_schema) {
    // Simple copy since layouts match
    if (srcIsDevice && dstIsDevice) {
      err = cudaMemcpy(
          self->mutable_data_ptr(),
          src->data_ptr(),
          total_bytes,
          cudaMemcpyDeviceToDevice);
      checkCudaError(err, "Failed to copy from device to device");
    } else if (srcIsDevice && !dstIsDevice) {
      err = cudaMemcpy(
          self->mutable_data_ptr(),
          src->data_ptr(),
          total_bytes,
          cudaMemcpyDeviceToHost);
      checkCudaError(err, "Failed to copy from device to host");
    } else if (!srcIsDevice && dstIsDevice) {
      err = cudaMemcpy(
          self->mutable_data_ptr(),
          src->data_ptr(),
          total_bytes,
          cudaMemcpyHostToDevice);
      checkCudaError(err, "Failed to copy from host to device");
    } else {
      std::memcpy(self->mutable_data_ptr(), src->data_ptr(), total_bytes);
    }
  } else {
    // Layout conversion needed (contiguous <-> channels-last)
    std::cout << "Layout conversion needed - doing element-wise copy"
              << std::endl;

    if (self->dim() != 4) {
      std::cout << "Error: Layout conversion only supported for 4D tensors"
                << std::endl;
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
      err = cudaMemcpy(
          src_host_data, src->data_ptr(), total_bytes, cudaMemcpyDeviceToHost);
      checkCudaError(err, "Failed to copy src to host");
      need_free_src = true;
    } else {
      src_host_data = static_cast<float*>(src->data_ptr());
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
      err = cudaMemcpy(
          self->mutable_data_ptr(),
          dst_host_data,
          total_bytes,
          cudaMemcpyHostToDevice);
      checkCudaError(err, "Failed to copy result to device");
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
    err = cudaMemcpy(
        &src_first, src->data_ptr(), sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError(err, "Failed to copy first src element");
  } else {
    src_first = static_cast<const float*>(src->data_ptr())[0];
  }

  if (dstIsDevice) {
    err = cudaMemcpy(
        &dst_first, self->data_ptr(), sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError(err, "Failed to copy first dst element");
  } else {
    dst_first = static_cast<const float*>(self->data_ptr())[0];
  }

  std::cout << "Copy verification: src[0]=" << src_first
            << ", dst[0]=" << dst_first << std::endl;
  std::cout << "aoti_torch_copy_ completed successfully" << std::endl;

  return Error::Ok;
}

AOTITorchError aoti_torch__reinterpret_tensor(
    AOTITensorHandle self,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    AOTITensorHandle* ret_new_tensor) {
  // Check if storage_offset is not 0 - return error if not
  AOTITorchError storage_offset_error = validate_storage_offset(storage_offset);
  if (storage_offset_error != Error::Ok) {
    return storage_offset_error;
  }

  // Check if dimensions match
  if (self->dim() != ndim) {
    std::cout << "Error: tensor dimension mismatch. self->dim(): "
              << self->dim() << ", provided ndim: " << ndim << std::endl;
    return Error::InvalidArgument;
  }

  // Get tensor properties from the input tensor
  int32_t dtype;
  AOTITorchError dtype_err = aoti_torch_get_dtype(self, &dtype);
  if (dtype_err != Error::Ok) {
    std::cout << "Error: failed to get dtype from input tensor" << std::endl;
    return dtype_err;
  }

  int32_t device_type;
  AOTITorchError device_type_err =
      aoti_torch_get_device_type(self, &device_type);
  if (device_type_err != Error::Ok) {
    std::cout << "Error: failed to get device_type from input tensor"
              << std::endl;
    return device_type_err;
  }

  int32_t device_index;
  AOTITorchError device_index_err =
      aoti_torch_get_device_index(self, &device_index);
  if (device_index_err != Error::Ok) {
    std::cout << "Error: failed to get device_index from input tensor"
              << std::endl;
    return device_index_err;
  }

  // Create new tensor with the provided sizes and strides using
  // aoti_torch_empty_strided
  AOTITorchError create_err = aoti_torch_empty_strided(
      ndim,
      sizes_ptr,
      strides_ptr,
      dtype,
      device_type,
      device_index,
      ret_new_tensor);

  if (create_err != Error::Ok) {
    std::cout << "Error: failed to create new tensor with empty_strided"
              << std::endl;
    return create_err;
  }

  // Copy data from source tensor to new tensor
  AOTITorchError copy_err = aoti_torch_copy_(*ret_new_tensor, self, 0);
  if (copy_err != Error::Ok) {
    std::cout << "Error: failed to copy data from source tensor to new tensor"
              << std::endl;
    // Clean up the created tensor on failure
    aoti_torch_delete_tensor_object(*ret_new_tensor);
    *ret_new_tensor = nullptr;
    return copy_err;
  }

  return Error::Ok;
}

// Cleanup function for clearing global state
void cleanup_memory() {
  is_tensor_own_memory.clear();
  if (!tensors.empty()) {
    std::cout << "Warning: tensors not empty" << std::endl;
  }
}

} // extern "C"

} // namespace aoti
} // namespace backends
} // namespace executorch
