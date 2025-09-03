/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "utils.h"
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace executorch {
namespace backends {
namespace aoti {

namespace internal {
// Constants for file operations
const char* const TENSOR_OUTPUT_FILENAME =
    "/home/gasoonjia/executorch/aoti_intermediate_output.txt";
} // namespace internal

extern "C" {

void aoti_torch_print_tensor_handle(AOTITensorHandle self, const char* msg) {
  printf("Printing tensor handle: %p\n", self);

  if (!self) {
    throw std::runtime_error("Tensor handle is null");
  }

  printf("Tensor handle is not null\n");

  // Get dtype and check if it's float32 (dtype 6 in PyTorch)
  int32_t dtype = 0;
  if (aoti_torch_get_dtype(self, &dtype) != AOTI_TORCH_SUCCESS) {
    throw std::runtime_error("Failed to get tensor dtype");
  }

  printf("Tensor dtype is: %d\n", dtype);

  if (dtype != 6) { // 6 is the dtype code for float32 in PyTorch
    throw std::runtime_error(
        "Tensor dtype is not float32. Expected dtype 6, got: " +
        std::to_string(dtype));
  }

  printf("Tensor dtype is float32\n");

  // Get data pointer
  void* data_ptr = nullptr;
  if (aoti_torch_get_data_ptr(self, &data_ptr) != AOTI_TORCH_SUCCESS ||
      !data_ptr) {
    throw std::runtime_error("Failed to get tensor data pointer");
  }

  printf("Tensor data pointer is %p not null\n", data_ptr);

  // Get dimensions
  int64_t dim = 0;
  if (aoti_torch_get_dim(self, &dim) != AOTI_TORCH_SUCCESS) {
    throw std::runtime_error("Failed to get tensor dimensions");
  }

  printf("Tensor dimensions are: %ld\n", dim);

  // Get sizes
  int64_t* sizes = nullptr;
  if (aoti_torch_get_sizes(self, &sizes) != AOTI_TORCH_SUCCESS || !sizes) {
    throw std::runtime_error("Failed to get tensor sizes");
  }

  printf("Tensor sizes are: %ld\n", sizes);

  // Calculate total number of elements
  int64_t total_elements = 1;
  for (int i = 0; i < dim; i++) {
    total_elements *= sizes[i];
  }

  printf("Total elements in tensor: %ld\n", total_elements);

  // Check device type to handle CUDA tensors properly
  int32_t device_type = 0;
  if (aoti_torch_get_device_type(self, &device_type) != AOTI_TORCH_SUCCESS) {
    throw std::runtime_error("Failed to get tensor device type");
  }

  printf("Tensor device type: %d\n", device_type);

  AtenTensorHandle cpu_tensor = nullptr;
  const float* float_data = nullptr;
  bool need_cleanup = false;

  // Check if tensor is on CUDA (device_type 1 is CUDA)
  if (device_type == 1) {
    printf("Tensor is on CUDA, copying to CPU...\n");

    // Get strides for creating CPU tensor
    int64_t* strides = nullptr;
    if (aoti_torch_get_strides(self, &strides) != AOTI_TORCH_SUCCESS ||
        !strides) {
      throw std::runtime_error("Failed to get tensor strides");
    }

    // Create a CPU tensor with same shape and layout
    if (aoti_torch_empty_strided(
            dim, sizes, strides, dtype, 0, -1, &cpu_tensor) !=
        AOTI_TORCH_SUCCESS) {
      throw std::runtime_error("Failed to create CPU tensor");
    }

    // Copy data from CUDA to CPU tensor
    if (aoti_torch_copy_(cpu_tensor, self, 0) != AOTI_TORCH_SUCCESS) {
      aoti_torch_delete_tensor_object(cpu_tensor);
      throw std::runtime_error("Failed to copy tensor from CUDA to CPU");
    }

    // Get CPU data pointer
    void* cpu_data_ptr = nullptr;
    if (aoti_torch_get_data_ptr(cpu_tensor, &cpu_data_ptr) !=
            AOTI_TORCH_SUCCESS ||
        !cpu_data_ptr) {
      aoti_torch_delete_tensor_object(cpu_tensor);
      throw std::runtime_error("Failed to get CPU tensor data pointer");
    }

    float_data = static_cast<const float*>(cpu_data_ptr);
    need_cleanup = true;
    printf("Successfully copied CUDA tensor to CPU\n");
  } else {
    // Tensor is already on CPU, use original data pointer
    printf("Tensor is on CPU, using original data pointer\n");
    float_data = static_cast<const float*>(data_ptr);
  }

  // Open file for writing (append mode to not overwrite previous outputs)
  printf("Writing tensor to file: %s\n", internal::TENSOR_OUTPUT_FILENAME);

  std::ofstream output_file(
      internal::TENSOR_OUTPUT_FILENAME, std::ios::out | std::ios::app);
  if (!output_file.is_open()) {
    if (need_cleanup) {
      aoti_torch_delete_tensor_object(cpu_tensor);
    }
    throw std::runtime_error(
        "Failed to open output file: " +
        std::string(internal::TENSOR_OUTPUT_FILENAME));
  }

  printf("Successfully opened file for writing\n");

  // Write message and tensor info to file
  output_file << "=== " << msg << " ===" << std::endl;
  output_file << "Device type: " << device_type << std::endl;
  output_file << "Dimensions: " << dim << std::endl;
  output_file << "Sizes: [";
  for (int i = 0; i < dim; i++) {
    output_file << sizes[i];
    if (i < dim - 1)
      output_file << ", ";
  }
  output_file << "]" << std::endl;
  output_file << "Total elements: " << total_elements << std::endl;
  output_file << "Data content:" << std::endl;

  // Write tensor data to file (now safe to access)
  for (int64_t i = 0; i < total_elements; i++) {
    output_file << float_data[i] << " ";
    if (i < total_elements - 1) {
      output_file << ", ";
      // Add newline every 10 elements for readability
      if ((i + 1) % 10 == 0) {
        output_file << std::endl;
      }
    }
  }
  output_file << std::endl << std::endl;

  // Clean up CPU tensor if we created one
  if (need_cleanup) {
    aoti_torch_delete_tensor_object(cpu_tensor);
    printf("Cleaned up temporary CPU tensor\n");
  }

  // File will be automatically closed when output_file goes out of scope
}

// Function to cleanup the tensor output file (to be called from
// aoti_backend.cpp)
void cleanup_aoti_tensor_output() {
  // No cleanup needed since file is opened and closed on each call
}

} // extern "C"

} // namespace aoti
} // namespace backends
} // namespace executorch
