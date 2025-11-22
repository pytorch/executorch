/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <optional>
#include <string>
#include <vector>

#include <torch/torch.h>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

#include <torch/csrc/inductor/aoti_runtime/interface.h>

namespace executorch::examples::whisper {

/**
 * Lightweight RAII wrapper around an AOTInductor shared library produced by
 * torch.compile(..., backend="aot_inductor").
 *
 * The runner loads the shared library, wires the exported C ABI entry points,
 * optionally feeds an external weights blob, and exposes a simple run() API
 * that takes and returns torch::Tensor objects.
 */
class TorchAOTIModel {
 public:
  TorchAOTIModel(
      std::string library_path,
      std::optional<std::string> weights_blob_path,
      std::string device = "cpu",
      std::optional<std::string> cubin_dir = std::nullopt);
  TorchAOTIModel(const TorchAOTIModel&) = delete;
  TorchAOTIModel& operator=(const TorchAOTIModel&) = delete;
  TorchAOTIModel(TorchAOTIModel&&) noexcept;
  TorchAOTIModel& operator=(TorchAOTIModel&&) noexcept;
  ~TorchAOTIModel();

  ::executorch::runtime::Error load();

  ::executorch::runtime::Result<std::vector<torch::Tensor>> run(
      const std::vector<torch::Tensor>& inputs) const;

  size_t num_inputs() const {
    return num_inputs_;
  }
  size_t num_outputs() const {
    return num_outputs_;
  }

 private:
  using CreateWithDeviceFn = int32_t (*)(
      AOTInductorModelContainerHandle*,
      size_t,
      const char*,
      const char*);
  using DeleteFn =
      int32_t (*)(AOTInductorModelContainerHandle container_handle);
  using RunFn = int32_t (*)(
      AOTInductorModelContainerHandle,
      AtenTensorHandle*,
      size_t,
      AtenTensorHandle*,
      size_t,
      AOTInductorStreamHandle,
      AOTIProxyExecutorHandle);
  using CountFn = int32_t (*)(
      AOTInductorModelContainerHandle,
      size_t* num_entries);
  using UpdateConstantsFromBlobFn = int32_t (*)(
      AOTInductorModelContainerHandle,
      const uint8_t* weight_blob_ptr);

  ::executorch::runtime::Error ensure_loaded() const;
  ::executorch::runtime::Error load_symbols();
  void unload();

  std::string library_path_;
  std::optional<std::string> weights_blob_path_;
  std::string device_;
  std::optional<std::string> cubin_dir_;

  void* so_handle_ = nullptr;
  AOTInductorModelContainerHandle container_handle_ = nullptr;

  CreateWithDeviceFn create_with_device_ = nullptr;
  DeleteFn delete_container_ = nullptr;
  RunFn run_ = nullptr;
  CountFn get_num_inputs_ = nullptr;
  CountFn get_num_outputs_ = nullptr;
  UpdateConstantsFromBlobFn update_constants_from_blob_ = nullptr;

  size_t num_inputs_ = 0;
  size_t num_outputs_ = 0;
};

} // namespace executorch::examples::whisper
