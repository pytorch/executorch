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

namespace { // Internal namespace for utility functions

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

// Check if tensor is in contiguous memory format (NCHW for 4D tensors)
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
  std::cout << "Creating tensor from data blob " << data << " - ndim: " << ndim
            << ", dtype: " << dtype << ", device_type: " << device_type
            << ", storage_offset: " << storage_offset << std::endl;

  // Only float32 tensors are supported
  if (dtype != 6) { // 6 = float32
    std::cout << "ERROR: Only float32 tensors are supported. Got dtype: "
              << dtype << " (expected: 6 for float32)" << std::endl;
    return Error::InvalidArgument;
  }

  // Storage offset must always be 0
  if (storage_offset != 0) {
    std::cout << "ERROR: Storage offset must be 0. Got storage_offset: "
              << storage_offset << std::endl;
    return Error::InvalidArgument;
  }

  // Convert sizes to the format expected by ExecutorTorch
  std::vector<int32_t> sizes(ndim);
  for (int i = 0; i < ndim; i++) {
    sizes[i] = static_cast<int32_t>(sizes_ptr[i]);
    std::cout << "Size[" << i << "] = " << sizes[i] << std::endl;
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
  std::cout << "////Allocated " << nbytes << " bytes at " << ptr
            << ", sizes_ptr " << sizes_ptr << std::endl;

  // ETensor sizes
  std::vector<int32_t> sizes(ndim);
  for (int i = 0; i < ndim; i++) {
    sizes[i] = sizes_ptr[i];
  }

  std::cout << "Sizes: ";
  for (int i = 0; i < ndim; i++) {
    std::cout << sizes[i] << ", ";
  }

  std::cout << std::endl;

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
  std::cout << std::endl;

  // ETensor creation
  auto tensor = executorch::extension::from_blob(ptr, sizes, strides);

  // Store the tensor so it doesn't get destroyed
  tensors.insert(tensor);
  *ret_new_tensor = tensor.get();
  is_tensor_own_memory[tensor.get()] = true;

  std::cout << "Finished. Created tensor " << tensor.get() << " with sizes "
            << std::endl
            << "sizes.data(): " << sizes.data()
            << ", tensor->sizes().data(): " << tensor->sizes().data()
            << std::endl;
  std::cout << "Size[0] of tensor " << tensor.get() << " is "
            << tensor->sizes()[0] << std::endl
            << std::endl;

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

  if (self_dtype != 6 || src_dtype != 6) { // 6 = float32
    std::cout << "Error: Only float32 tensors supported. Got self.dtype="
              << self_dtype << ", src.dtype=" << src_dtype << std::endl;
    return Error::InvalidArgument;
  }

  // Get stride information for layout validation
  int64_t* self_strides;
  int64_t* src_strides;
  aoti_torch_get_strides(self, &self_strides);
  aoti_torch_get_strides(src, &src_strides);

  auto self_sizes = self->sizes();
  auto src_sizes = src->sizes();

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

  if (same_schema) {
    std::cout << "Same tensor schema detected - enabling naive copy"
              << std::endl;
    // For same schema, we don't need to check memory formats - just use direct
    // copy
  } else {
    // Different strides: check memory format and only support contiguous <->
    // channels-last conversion
    std::cout
        << "Different tensor schemas - checking memory format compatibility"
        << std::endl;

    // Check if contiguous (strides decrease from left to right)
    self_is_contiguous =
        is_tensor_contiguous(self->dim(), self_sizes.data(), self_strides);

    src_is_contiguous =
        is_tensor_contiguous(src->dim(), src_sizes.data(), src_strides);

    // Check if channels-last (4D: NHWC format)
    if (!self_is_contiguous) {
      self_is_channels_last =
          is_tensor_channels_last(self->dim(), self_sizes.data(), self_strides);
    }

    if (!src_is_contiguous) {
      src_is_channels_last =
          is_tensor_channels_last(src->dim(), src_sizes.data(), src_strides);
    }

    // Validate layout assumptions only when schemas differ
    if (!self_is_contiguous && !self_is_channels_last) {
      std::cout
          << "Error: self tensor must be contiguous or channels-last for stride conversion. "
          << "Got strides: [";
      for (int i = 0; i < self->dim(); i++) {
        std::cout << self_strides[i] << (i < self->dim() - 1 ? ", " : "");
      }
      std::cout << "]" << std::endl;
      std::cout << "self_sizes: [";
      for (int i = 0; i < self->dim(); i++) {
        std::cout << self_sizes[i] << (i < self->dim() - 1 ? ", " : "");
      }
      std::cout << "]" << std::endl;
      return Error::InvalidArgument;
    }

    if (!src_is_contiguous && !src_is_channels_last) {
      std::cout
          << "Error: src tensor must be contiguous or channels-last for stride conversion. \n"
          << "Got strides: [";
      for (int i = 0; i < src->dim(); i++) {
        std::cout << src_strides[i] << (i < src->dim() - 1 ? ", " : "");
      }
      std::cout << "]" << std::endl;
      std::cout << "src_sizes: [";
      for (int i = 0; i < self->dim(); i++) {
        std::cout << src_sizes[i] << (i < self->dim() - 1 ? ", " : "");
      }
      std::cout << "]" << std::endl;
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

  std::cout << "Copy layout: src="
            << (src_is_contiguous ? "contiguous" : "channels-last") << " ("
            << (srcIsDevice ? "GPU" : "CPU") << ") -> "
            << "dst=" << (self_is_contiguous ? "contiguous" : "channels-last")
            << " (" << (dstIsDevice ? "GPU" : "CPU") << ")" << std::endl;

  size_t total_bytes = src->nbytes();

  if (same_schema) {
    std::cout << "Same layout - doing direct copy of " << total_bytes
              << " bytes" << std::endl;

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

AOTITorchError aoti_torch__reinterpret_tensor(
    AOTITensorHandle self,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    AOTITensorHandle* ret_new_tensor) {
  std::cout << "aoti_torch__reinterpret_tensor called with tensor " << self
            << ", ndim: " << ndim << ", storage_offset: " << storage_offset
            << std::endl;

  for (int i = 0; i < ndim; i++) {
    std::cout << "sizes[" << i << "]: " << sizes_ptr[i] << std::endl;
  }
  for (int i = 0; i < ndim; i++) {
    std::cout << "strides[" << i << "]: " << strides_ptr[i] << std::endl;
  }

  // Check if storage_offset is not 0 - return error if not
  if (storage_offset != 0) {
    std::cout
        << "Error: aoti_torch__reinterpret_tensor does not support non-zero storage_offset: "
        << storage_offset << std::endl;
    return Error::InvalidArgument;
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

  if (dtype != 6) { // 6 = float32
    std::cout
        << "ERROR: Only float32 tensors are supported in reinterpret_tensor. Got dtype: "
        << dtype << " (expected: 6 for float32)" << std::endl;
    return Error::InvalidArgument;
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

  std::cout << "Creating new tensor with dtype: " << dtype
            << ", device_type: " << device_type
            << ", device_index: " << device_index << std::endl;

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

  std::cout << "Successfully created reinterpreted tensor " << *ret_new_tensor
            << " from source tensor " << self << std::endl;

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
