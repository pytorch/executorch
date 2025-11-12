/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "torch_aoti_model.h"

#include <dlfcn.h>

#include <fstream>
#include <utility>

#include <torch/csrc/inductor/aoti_runtime/utils.h>

#include <executorch/runtime/platform/log.h>

namespace executorch::examples::whisper {
namespace {

using ::executorch::runtime::Error;

template <typename FuncPtr>
Error load_symbol(void* handle, const char* name, FuncPtr* out) {
  if (!handle) {
    ET_LOG(Error, "dlopen handle is null while loading %s", name);
    return Error::Internal;
  }
  dlerror();
  auto* symbol = dlsym(handle, name);
  const char* err = dlerror();
  if (err != nullptr) {
    ET_LOG(Error, "Failed to resolve %s: %s", name, err);
    return Error::Internal;
  }
  *out = reinterpret_cast<FuncPtr>(symbol);
  return Error::Ok;
}

torch::aot_inductor::RAIIAtenTensorHandle tensor_to_handle(
    const torch::Tensor& tensor) {
  // Shallow copy to keep storage alive while giving runtime ownership of the
  // temporary handle.
  auto* cloned = new at::Tensor(tensor);
  return torch::aot_inductor::RAIIAtenTensorHandle(
      reinterpret_cast<AtenTensorHandle>(cloned));
}

torch::Tensor handle_to_tensor(
    const torch::aot_inductor::RAIIAtenTensorHandle& handle) {
  auto* tensor_ptr = reinterpret_cast<at::Tensor*>(handle.get());
  return torch::Tensor(*tensor_ptr);
}

std::vector<uint8_t> load_file(const std::string& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    ET_LOG(Error, "Failed to open %s", path.c_str());
    return {};
  }
  std::streamsize size = stream.tellg();
  stream.seekg(0, std::ios::beg);
  std::vector<uint8_t> buffer(size);
  if (!stream.read(reinterpret_cast<char*>(buffer.data()), size)) {
    ET_LOG(Error, "Failed to read %s", path.c_str());
    return {};
  }
  return buffer;
}

} // namespace

TorchAOTIModel::TorchAOTIModel(
    std::string library_path,
    std::optional<std::string> weights_blob_path,
    std::string device,
    std::optional<std::string> cubin_dir)
    : library_path_(std::move(library_path)),
      weights_blob_path_(std::move(weights_blob_path)),
      device_(std::move(device)),
      cubin_dir_(std::move(cubin_dir)) {}

TorchAOTIModel::TorchAOTIModel(TorchAOTIModel&& other) noexcept {
  *this = std::move(other);
}

TorchAOTIModel& TorchAOTIModel::operator=(TorchAOTIModel&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  unload();
  library_path_ = std::move(other.library_path_);
  weights_blob_path_ = std::move(other.weights_blob_path_);
  device_ = std::move(other.device_);
  cubin_dir_ = std::move(other.cubin_dir_);
  so_handle_ = other.so_handle_;
  container_handle_ = other.container_handle_;
  create_with_device_ = other.create_with_device_;
  delete_container_ = other.delete_container_;
  run_ = other.run_;
  get_num_inputs_ = other.get_num_inputs_;
  get_num_outputs_ = other.get_num_outputs_;
  update_constants_from_blob_ = other.update_constants_from_blob_;
  num_inputs_ = other.num_inputs_;
  num_outputs_ = other.num_outputs_;

  other.so_handle_ = nullptr;
  other.container_handle_ = nullptr;
  other.create_with_device_ = nullptr;
  other.delete_container_ = nullptr;
  other.run_ = nullptr;
  other.get_num_inputs_ = nullptr;
  other.get_num_outputs_ = nullptr;
  other.update_constants_from_blob_ = nullptr;
  other.num_inputs_ = 0;
  other.num_outputs_ = 0;
  return *this;
}

TorchAOTIModel::~TorchAOTIModel() {
  unload();
}

Error TorchAOTIModel::ensure_loaded() const {
  if (so_handle_ == nullptr || container_handle_ == nullptr || run_ == nullptr) {
    return Error::InvalidState;
  }
  return Error::Ok;
}

void TorchAOTIModel::unload() {
  if (delete_container_ && container_handle_) {
    delete_container_(container_handle_);
  }
  container_handle_ = nullptr;
  if (so_handle_) {
    dlclose(so_handle_);
    so_handle_ = nullptr;
  }
  create_with_device_ = nullptr;
  delete_container_ = nullptr;
  run_ = nullptr;
  get_num_inputs_ = nullptr;
  get_num_outputs_ = nullptr;
  update_constants_from_blob_ = nullptr;
  num_inputs_ = 0;
  num_outputs_ = 0;
}

Error TorchAOTIModel::load_symbols() {
  ET_CHECK_OK_OR_RETURN_ERROR(
      load_symbol(so_handle_, "AOTInductorModelContainerCreateWithDevice", &create_with_device_));
  ET_CHECK_OK_OR_RETURN_ERROR(
      load_symbol(so_handle_, "AOTInductorModelContainerDelete", &delete_container_));
  ET_CHECK_OK_OR_RETURN_ERROR(
      load_symbol(so_handle_, "AOTInductorModelContainerRun", &run_));
  ET_CHECK_OK_OR_RETURN_ERROR(
      load_symbol(so_handle_, "AOTInductorModelContainerGetNumInputs", &get_num_inputs_));
  ET_CHECK_OK_OR_RETURN_ERROR(
      load_symbol(so_handle_, "AOTInductorModelContainerGetNumOutputs", &get_num_outputs_));
  auto update_result = load_symbol(
      so_handle_,
      "AOTInductorModelUpdateConstantsFromBlob",
      &update_constants_from_blob_);
  if (update_result != Error::Ok) {
    update_constants_from_blob_ = nullptr;
    ET_LOG(
        Warn,
        "Library %s does not expose AOTInductorModelUpdateConstantsFromBlob. Assuming weights are embedded.",
        library_path_.c_str());
  }
  return Error::Ok;
}

Error TorchAOTIModel::load() {
  if (ensure_loaded() == Error::Ok) {
    return Error::Ok;
  }

  so_handle_ = dlopen(library_path_.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!so_handle_) {
    ET_LOG(Error, "dlopen failed for %s: %s", library_path_.c_str(), dlerror());
    return Error::AccessFailed;
  }

  ET_CHECK_OK_OR_RETURN_ERROR(load_symbols());

  ET_CHECK_OR_RETURN_ERROR(
      create_with_device_ != nullptr,
      Error::InvalidState,
      "create_with_device is null");

  int32_t status = create_with_device_(
      &container_handle_,
      1,
      device_.c_str(),
      cubin_dir_.has_value() ? cubin_dir_->c_str() : nullptr);
  ET_CHECK_OR_RETURN_ERROR(
      status == 0,
      Error::Internal,
      "Failed to create AOTI container for %s",
      library_path_.c_str());

  if (weights_blob_path_ && update_constants_from_blob_) {
    auto blob = load_file(*weights_blob_path_);
    ET_CHECK_OR_RETURN_ERROR(
        !blob.empty(),
        Error::AccessFailed,
        "Failed to read weights blob %s",
        weights_blob_path_->c_str());
    status =
        update_constants_from_blob_(container_handle_, blob.data());
    ET_CHECK_OR_RETURN_ERROR(
        status == 0, Error::Internal, "Failed to load weights blob");
  }

  status = get_num_inputs_(container_handle_, &num_inputs_);
  ET_CHECK_OR_RETURN_ERROR(status == 0, Error::Internal, "get_num_inputs failed");
  status = get_num_outputs_(container_handle_, &num_outputs_);
  ET_CHECK_OR_RETURN_ERROR(status == 0, Error::Internal, "get_num_outputs failed");

  return Error::Ok;
}

::executorch::runtime::Result<std::vector<torch::Tensor>> TorchAOTIModel::run(
    const std::vector<torch::Tensor>& inputs) const {
  ET_CHECK_OK_OR_RETURN_ERROR(ensure_loaded());
  ET_CHECK_OR_RETURN_ERROR(
      inputs.size() == num_inputs_,
      Error::InvalidArgument,
      "Expected %zu inputs, got %zu",
      num_inputs_,
      inputs.size());

  std::vector<torch::aot_inductor::RAIIAtenTensorHandle> owned_inputs;
  owned_inputs.reserve(inputs.size());
  std::vector<AtenTensorHandle> input_handles(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    owned_inputs.push_back(tensor_to_handle(inputs[i]));
    input_handles[i] = owned_inputs.back().get();
  }

  std::vector<AtenTensorHandle> raw_outputs(num_outputs_, nullptr);
  int32_t status = run_(
      container_handle_,
      input_handles.data(),
      input_handles.size(),
      raw_outputs.data(),
      raw_outputs.size(),
      nullptr,
      nullptr);

  ET_CHECK_OR_RETURN_ERROR(status == 0, Error::Internal, "AOTI execution failed");

  std::vector<torch::Tensor> outputs;
  outputs.reserve(raw_outputs.size());
  for (auto* handle : raw_outputs) {
    torch::aot_inductor::RAIIAtenTensorHandle raii_handle(handle);
    outputs.push_back(handle_to_tensor(raii_handle));
  }
  return outputs;
}

} // namespace executorch::examples::whisper
