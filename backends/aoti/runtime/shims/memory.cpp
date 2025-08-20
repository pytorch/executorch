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

namespace executorch {
namespace backends {
namespace aoti {

using executorch::runtime::Error;
using executorch::runtime::etensor::Tensor;

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
  std::cout << "Creating tensor from data blob " << data << " - ndim: " << ndim
            << ", dtype: " << dtype << ", device_type: " << device_type
            << std::endl;

  // Convert sizes to the format expected by ExecutorTorch
  std::vector<int32_t> sizes(ndim);
  for (int i = 0; i < ndim; i++) {
    sizes[i] = static_cast<int32_t>(sizes_ptr[i]);
    std::cout << "Size[" << i << "] = " << sizes[i] << std::endl;
  }

  // check the tensor format
  // Only support contiguous format for now
  int64_t expected_stride = 1;
  for (int i = ndim - 1; i >= 0; --i) {
    if (strides_ptr[i] != expected_stride) {
      std::cout
          << "aoti_torch_create_tensor_from_blob_v2 failed since input stride is not in contiguous format. Return with Error"
          << std::endl;
      return Error::InvalidArgument;
    }
    expected_stride *= sizes_ptr[i];
  }

  // Adjust data pointer by storage_offset if needed
  void* adjusted_data = data;
  if (storage_offset > 0) {
    // Calculate byte offset based on dtype size
    size_t dtype_size =
        4; // Assuming float32 for now, you may need to handle other dtypes
    if (dtype == 6) { // float32
      dtype_size = 4;
    } else {
      std::cout << "Error: Unhandled dtype " << dtype << std::endl;
      return Error::NotImplemented;
    }
    adjusted_data = static_cast<char*>(data) + (storage_offset * dtype_size);
  }

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

  std::cout << "Successfully created tensor from blob: " << tensor.get()
            << " wrapping data at: " << adjusted_data << std::endl;

  return Error::Ok;
}

AOTITorchError aoti_torch_create_tensor_from_blob(
    void* data,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AOTITensorHandle* ret_new_tensor) {
  throw std::runtime_error("Should never create from blob");
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

  if (dtype != 6) { // throw if not float32
    throw std::runtime_error("Need to implement empty_strided for non-float32");
  }

  int64_t nbytes = numel * 4;

  if (device_type == 1) { // cuda
    std::cout << "Allocating " << nbytes << " bytes on CUDA " << std::endl;
    cudaError_t err = cudaMalloc(&ptr, nbytes);
    if (err != cudaSuccess) {
      std::cout << "failed to allocate " << nbytes
                << " error: " << cudaGetErrorString(err) << std::endl;
      throw std::runtime_error("Failed to call cudaMalloc");
    }
  } else if (device_type == 0) { // cpu
    std::cout << "Allocating " << nbytes << " bytes on CPU " << std::endl;
    // Ensure 16-byte alignment for CPU memory to match CUDA requirements
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
  std::cout << "Allocated " << nbytes << " bytes at " << ptr << ", sizes_ptr "
            << sizes_ptr << std::endl;

  // ETensor sizes
  std::vector<int32_t> sizes(ndim);
  for (int i = 0; i < ndim; i++) {
    sizes[i] = sizes_ptr[i];
  }
  // ETensor creation
  auto tensor = executorch::extension::make_tensor_ptr(sizes, ptr);

  // Store the tensor
  tensors.insert(tensor);

  std::cout << "sizes.data(): " << sizes.data()
            << ", tensor->sizes().data(): " << tensor->sizes().data()
            << std::endl;
  std::cout << "Size[0] of tensor " << tensor.get() << " is "
            << tensor->sizes()[0] << std::endl;
  *ret_new_tensor = tensor.get();
  is_tensor_own_memory[tensor.get()] = true;
  return Error::Ok;
}

AOTITorchError aoti_torch_delete_tensor_object(AOTITensorHandle tensor) {
  std::cout << "Called aoti_torch_delete_tensor_object for tensor " << tensor
            << std::endl;

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
    std::cout << "Tensor " << tensor << " does not own memory. Skipped \n\n"
              << std::endl;
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
        std::cout << "Freeing GPU memory at " << data_ptr << std::endl;
        cudaDeviceSynchronize(); // Wait for all operations to complete BEFORE
                                 // freeing
        cudaFree(data_ptr);
        std::cout << "GPU memory freed successfully" << std::endl;
      } else {
        // This is CPU memory - free immediately
        std::cout << "Freeing CPU memory at " << data_ptr << std::endl;
        free(data_ptr);
        std::cout << "CPU memory freed successfully" << std::endl;
      }

      std::cout << "Memory freed. Now erasing tensor " << tensor << std::endl;

      // Remove from set (this will call the destructor if it's the last
      // reference)
      tensors.erase(it);

      std::cout << "Tensor erased. Now returning \n\n" << std::endl;

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
  // check if size is the same
  if (self->dim() != src->dim()) {
    std::cout << "self.dim() " << self->dim() << ", src.dim() " << src->dim()
              << std::endl;
    throw std::runtime_error("self.dim() != src.dim()");
  }
  std::cout << "self->data_ptr(): " << self->data_ptr()
            << " sizes: " << self->sizes().data() << std::endl;
  std::cout << "src->data_ptr(): " << src->data_ptr()
            << " sizes: " << src->sizes().data() << std::endl;
  for (int i = 0; i < self->dim(); i++) {
    if (self->sizes()[i] != src->sizes()[i]) {
      std::cout << "self.sizes()[i] " << self->sizes()[i] << ", src.sizes()[i] "
                << src->sizes()[i] << std::endl;
      throw std::runtime_error("size mismatch");
    }
  }

  int size = src->nbytes();
  // should check for device
  cudaPointerAttributes srcAttributes, dstAttributes;
  cudaError_t err;
  // Get attributes of the source pointer
  err = cudaPointerGetAttributes(&srcAttributes, src->data_ptr());
  checkCudaError(err, "Failed to get source pointer attributes");
  // Get attributes of the destination pointer
  err = cudaPointerGetAttributes(&dstAttributes, self->data_ptr());
  checkCudaError(err, "Failed to get destination pointer attributes");
  bool srcIsDevice = srcAttributes.type == cudaMemoryTypeDevice;
  bool dstIsDevice = dstAttributes.type == cudaMemoryTypeDevice;
  // Determine the memory locations and perform the appropriate copy
  if (srcIsDevice && dstIsDevice) {
    // Device to Device copy
    err = cudaMemcpy(
        self->mutable_data_ptr(),
        src->data_ptr(),
        size,
        cudaMemcpyDeviceToDevice);
    checkCudaError(err, "Failed to copy from device to device");
  } else if (srcIsDevice && !dstIsDevice) {
    // Device to Host copy
    err = cudaMemcpy(
        self->mutable_data_ptr(),
        src->data_ptr(),
        size,
        cudaMemcpyDeviceToHost);
    std::cout << "Device to Host copy, self data: "
              << ((float*)self->data_ptr())[0] << std::endl;
    checkCudaError(err, "Failed to copy from device to host");
  } else if (!srcIsDevice && dstIsDevice) {
    // Host to Device copy
    err = cudaMemcpy(
        self->mutable_data_ptr(),
        src->data_ptr(),
        size,
        cudaMemcpyHostToDevice);
    std::cout << "Host to Device copy, src data: "
              << ((float*)src->data_ptr())[0] << std::endl;
    checkCudaError(err, "Failed to copy from host to device");
  } else if (!srcIsDevice && !dstIsDevice) {
    // Host to Host copy
    std::cout << "Host to Host copy, src data: " << ((float*)src->data_ptr())[0]
              << std::endl;
    std::memcpy(self->mutable_data_ptr(), src->data_ptr(), size);
  } else {
    std::cerr << "Error: Unknown memory type. self: " << dstAttributes.type
              << ", src: " << srcAttributes.type << std::endl;
    throw std::runtime_error("Unknown memory type");
  }
  // print first value of src and self
  return Error::Ok;
}

AOTITorchError aoti_torch_create_cuda_stream_guard(
    void* stream,
    int32_t device_index,
    CUDAStreamGuardHandle* ret_guard) {
  std::cout << "Entering stream guard for device " << device_index
            << " with stream " << stream << std::endl;

  // Set device
  cudaError_t err = cudaSetDevice(device_index);
  if (err != cudaSuccess) {
    std::cerr << "Failed to set device " << device_index << ": "
              << cudaGetErrorString(err) << std::endl;
    return Error::Internal;
  }

  // Create minimal guard structure
  CUDAStreamGuardOpaque* guard = new CUDAStreamGuardOpaque();
  guard->device_index = device_index;
  guard->original_stream = static_cast<cudaStream_t>(stream);
  guard->sync_event = nullptr;

  std::cout << "Stream guard created successfully for stream " << stream
            << std::endl;

  *ret_guard = guard;
  return Error::Ok;
}

AOTITorchError aoti_torch_delete_cuda_stream_guard(
    CUDAStreamGuardHandle guard) {
  std::cout << "Exiting stream guard" << std::endl;

  if (guard == nullptr) {
    return Error::Ok;
  }

  // Clean up the guard structure
  delete guard;

  std::cout << "Stream guard cleanup completed" << std::endl;
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
