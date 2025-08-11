/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "cuda_runtime.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace executorch {
namespace backends {
namespace aoti {

// Here is where the aoti bouncers are going to be defined.
// I define the globals aoti generated compiled code calls
// They can be backed by ET systems

using namespace std;

using executorch::aten::ScalarType;
using executorch::runtime::ArrayRef;
using executorch::runtime::Backend;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::Result;
using executorch::runtime::etensor::Tensor;

extern "C" {
using AOTITensorHandle = Tensor*;

// TODO: We should get a proper one
struct CUDAStreamGuardOpaque;
using CUDAStreamGuardHandle = CUDAStreamGuardOpaque*;

using AOTIRuntimeError = Error;
using AOTITorchError = Error;

struct AOTInductorModelContainerOpaque;
using AOTInductorModelContainerHandle = AOTInductorModelContainerOpaque*;
using AOTInductorStreamHandle = void*;
using AOTIProxyExecutorHandle = void*;

using AOTInductorModelContainerCreateWithDeviceFunc = AOTIRuntimeError (*)(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    const char* device_str,
    const char* cubin_dir);

using AOTInductorModelContainerDeleteFunc =
    AOTIRuntimeError (*)(AOTInductorModelContainerHandle container_handle);

using AOTInductorModelContainerGetNumInputsFunc = AOTIRuntimeError (*)(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_constants);

using AOTInductorModelContainerGetNumOutputsFunc = AOTIRuntimeError (*)(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_constants);

using AOTInductorModelContainerRunFunc = AOTIRuntimeError (*)(
    AOTInductorModelContainerHandle container_handle,
    AOTITensorHandle* input_handles, // array of input AOTITensorHandle; handles
                                     // are stolen; the array itself is borrowed
    size_t num_inputs,
    AOTITensorHandle*
        output_handles, // array for writing output AOTITensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    size_t num_outputs,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle);

AOTInductorModelContainerCreateWithDeviceFunc
    AOTInductorModelContainerCreateWithDevice = nullptr;
AOTInductorModelContainerDeleteFunc AOTInductorModelContainerDelete = nullptr;
AOTInductorModelContainerGetNumInputsFunc
    AOTInductorModelContainerGetNumInputs = nullptr;
AOTInductorModelContainerGetNumOutputsFunc
    AOTInductorModelContainerGetNumOutputs = nullptr;
AOTInductorModelContainerRunFunc AOTInductorModelContainerRun = nullptr;
std::unordered_map<Tensor*, std::vector<int64_t>> tensor_to_sizes;
std::unordered_map<Tensor*, std::vector<int64_t>> tensor_to_strides;
std::unordered_set<std::shared_ptr<Tensor>> tensors;

int32_t aoti_torch_grad_mode_is_enabled() {
  // No autograd ever
  return false;
}

void aoti_torch_grad_mode_set_enabled(bool enabled) {
  if (enabled) {
    throw std::runtime_error("Cannot enable autograd");
  }
}

AOTITorchError aoti_torch_get_data_ptr(
    AOTITensorHandle tensor,
    void** ret_data_ptr) {
  *ret_data_ptr = tensor->mutable_data_ptr();
  return Error::Ok;
}

AOTITorchError aoti_torch_get_storage_offset(
    AOTITensorHandle tensor,
    int64_t* ret_storage_offset) {
  // Storage offset is always 0 in ET
  *ret_storage_offset = 0;
  return Error::Ok;
}

AOTITorchError aoti_torch_get_strides(
    AOTITensorHandle tensor,
    int64_t** ret_strides) {
  auto it = tensor_to_strides.find(tensor);
  if (it == tensor_to_strides.end()) {
    std::vector<int64_t> strides(tensor->dim());
    auto tensor_strides = tensor->strides();
    for (int i = 0; i < tensor->dim(); i++) {
      strides[i] = tensor_strides[i];
    }
    it = tensor_to_strides.emplace(tensor, std::move(strides)).first;
  }
  *ret_strides = it->second.data();
  std::cout << "getting strides from tensor " << tensor << " with dim "
            << tensor->dim() << std::endl;
  for (int i = 0; i < tensor->dim(); i++) {
    std::cout << "strides " << i << " = " << *ret_strides[i] << std::endl;
  }
  return Error::Ok;
}

AOTITorchError aoti_torch_get_dtype(
    AOTITensorHandle tensor,
    int32_t* ret_dtype) {
  *ret_dtype = static_cast<int32_t>(tensor->scalar_type());
  return Error::Ok;
}

AOTITorchError aoti_torch_get_sizes(
    AOTITensorHandle tensor,
    int64_t** ret_sizes) {
  auto it = tensor_to_sizes.find(tensor);
  if (it == tensor_to_sizes.end()) {
    std::vector<int64_t> sizes(tensor->dim());
    auto tensor_sizes = tensor->sizes();
    for (int i = 0; i < tensor->dim(); i++) {
      sizes[i] = tensor_sizes[i];
    }
    it = tensor_to_sizes.emplace(tensor, std::move(sizes)).first;
  }
  *ret_sizes = it->second.data();
  std::cout << "getting sizes from tensor " << tensor << " with dim "
            << tensor->dim() << std::endl;
  for (int i = 0; i < tensor->dim(); i++) {
    std::cout << "size " << i << " = " << *ret_sizes[i] << std::endl;
  }
  return Error::Ok;
}

AOTITorchError aoti_torch_get_storage_size(
    AOTITensorHandle tensor,
    int64_t* ret_size) {
  throw std::runtime_error("Cannot get storage size on ETensor");
}

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
  throw std::runtime_error("Not creating Tensor from blob here");
}

AOTITorchError aoti_torch_create_cuda_stream_guard(
    void* stream,
    int32_t device_index,
    CUDAStreamGuardHandle* ret_guard) {
  std::cout << "Entering stream guard for device " << device_index << std::endl;
  return Error::Ok;
}

AOTITorchError aoti_torch_delete_cuda_stream_guard(
    CUDAStreamGuardHandle guard) {
  std::cout << "Exiting stream guard" << std::endl;
  return Error::Ok;
}

int aoti_torch_device_type_cpu() {
  // Let's say cpu is 0 for ET as well
  return 0;
}

__attribute__((__visibility__("default"))) int32_t
aoti_torch_device_type_cuda() {
  // Let's say cuda is 1 for ET as well
  return 1;
}

__attribute__((__visibility__("default"))) int32_t aoti_torch_dtype_float32() {
  // Let assume the dtype here is all we will support
  return 6;
}

AOTITorchError aoti_torch_delete_tensor_object(AOTITensorHandle tensor) {
  std::cout << "Deleting " << tensor << std::endl;
  for (auto it = tensors.begin(); it != tensors.end(); ++it) {
    if (it->get() == tensor) {
      tensors.erase(it);
      break; // Exit the loop once the element is found and removed
    }
  }
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
      std::cout << "failed to allocate " << nbytes << std::endl;
      throw std::runtime_error("Failed to call cudaMalloc");
    }
  } else if (device_type == 0) { // cpu
    std::cout << "Allocating " << nbytes << " bytes on CPU " << std::endl;
    ptr = malloc(nbytes);
    if (ptr == nullptr) {
      throw std::runtime_error("Failed to call malloc");
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
  return Error::Ok;
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
}

struct AOTIDelegateHandle {
  void* so_handle;
  AOTInductorModelContainerHandle container_handle;
};

class AOTIBackend final : public ::executorch::runtime::BackendInterface {
 public:
  // Once in program
  AOTIBackend() {
    ET_LOG(Info, "AOTIBackend ctor");
  }

  bool is_available() const override {
    return 1;
  }

  // Once per loaded binary blob
  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed, // This will be the buffer from aoti_backend
      ArrayRef<CompileSpec> compile_specs // This will be my empty list
  ) const override {
    // We could load the .so content directly. But I don't want to deal with
    // relocation. So dumping a file and using dlopen

    // // Create a temporary file
    // std::ofstream outfile("/tmp/test.so", std::ios::binary);

    // // Write the ELF buffer to the temporary file
    // outfile.write((char*)processed->data(), sizeof(void*) * processed->size());

    // // Finish writing the file to disk
    // outfile.close();

    // // Free the in-memory buffer
    // processed->Free();

    const char* so_path = static_cast<const char*>(processed->data());

    printf("so path: %s\n", so_path);

    // Load the ELF using dlopen
    void* so_handle = dlopen(so_path, RTLD_LAZY | RTLD_LOCAL);
    if (so_handle == nullptr) {
      std::cout << dlerror() << std::endl;
      return Error::AccessFailed;
    }

    AOTInductorModelContainerCreateWithDevice =
        reinterpret_cast<AOTInductorModelContainerCreateWithDeviceFunc>(
            dlsym(so_handle, "AOTInductorModelContainerCreateWithDevice"));
    if (AOTInductorModelContainerCreateWithDevice == nullptr) {
      perror("dlsym1");
      return Error::AccessFailed;
    }
    AOTInductorModelContainerDelete =
        reinterpret_cast<AOTInductorModelContainerDeleteFunc>(
            dlsym(so_handle, "AOTInductorModelContainerDelete"));
    if (AOTInductorModelContainerDelete == nullptr) {
      perror("dlsym2");
      return Error::AccessFailed;
    }
    AOTInductorModelContainerGetNumInputs =
        reinterpret_cast<AOTInductorModelContainerGetNumInputsFunc>(
            dlsym(so_handle, "AOTInductorModelContainerGetNumInputs"));
    if (AOTInductorModelContainerGetNumInputs == nullptr) {
      perror("dlsym3");
      return Error::AccessFailed;
    }
    AOTInductorModelContainerGetNumOutputs =
        reinterpret_cast<AOTInductorModelContainerGetNumOutputsFunc>(
            dlsym(so_handle, "AOTInductorModelContainerGetNumOutputs"));
    if (AOTInductorModelContainerGetNumOutputs == nullptr) {
      perror("dlsym4");
      return Error::AccessFailed;
    }
    AOTInductorModelContainerRun =
        reinterpret_cast<AOTInductorModelContainerRunFunc>(
            dlsym(so_handle, "AOTInductorModelContainerRun"));
    if (AOTInductorModelContainerRun == nullptr) {
      perror("dlsym5");
      return Error::AccessFailed;
    }

    AOTInductorModelContainerHandle container_handle = nullptr;

    AOTIRuntimeError err;

    err = AOTInductorModelContainerCreateWithDevice(
        &container_handle, 1, "cuda", nullptr);
    printf("container_handle=%p\n", container_handle);

    AOTIDelegateHandle* handle = new AOTIDelegateHandle();
    handle->so_handle = so_handle;
    handle->container_handle = container_handle;
    return (DelegateHandle*)handle; // Return the handle post-processing
  }

  // Once per execution
  Error execute(
      BackendExecutionContext& context,
      DelegateHandle* handle_,
      EValue** args) const override {
    AOTIDelegateHandle* handle = (AOTIDelegateHandle*)handle_;

    size_t num_inputs;
    AOTInductorModelContainerGetNumInputs(
        handle->container_handle, &num_inputs);

    size_t num_outputs;
    AOTInductorModelContainerGetNumOutputs(
        handle->container_handle, &num_outputs);

    std::vector<AOTITensorHandle> inputs(num_inputs);
    std::vector<AOTITensorHandle> outputs(num_outputs);

    for (int i = 0; i < num_inputs; i++) {
      auto tensor_in = args[i]->toTensor();
      inputs[i] = &tensor_in;
    }

    for (int i = num_inputs; i < num_inputs + num_outputs; i++) {
      auto tensor_out = args[i]->toTensor();
      outputs[i - num_inputs] = &tensor_out;
    }

    AOTInductorModelContainerRun(
        handle->container_handle,
        inputs.data(),
        num_inputs,
        outputs.data(),
        num_outputs,
        // Should these last two be something?
        nullptr,
        nullptr);

    // Still need to copy the output to args, because they are malloc'ed but
    // not using the data_ptr from outputs.
    for (int i = 0; i < num_outputs; i++) {
      auto args_out = args[i + num_inputs]->toTensor();
      aoti_torch_copy_(&args_out, outputs[i], 0);
    }
    return Error::Ok;
  }

  void destroy(DelegateHandle* handle_) const override {
    AOTIDelegateHandle* handle = (AOTIDelegateHandle*)handle_;
    dlclose(handle->so_handle);
    AOTInductorModelContainerDelete(handle->container_handle);
    free(handle);
    tensor_to_sizes.clear();
    tensor_to_strides.clear();
  }
};

} // namespace aoti

namespace {
auto cls = aoti::AOTIBackend();
executorch::runtime::Backend backend{"AotiBackend", &cls};
static executorch::runtime::Error success_with_compiler =
    register_backend(backend);
} // namespace

} // namespace backends
} // namespace executorch
