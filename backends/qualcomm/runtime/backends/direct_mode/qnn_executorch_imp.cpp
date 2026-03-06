/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/backends/direct_mode/QnnExecuTorchIdlWrapper.h>
#include "HAP_farf.h"
#include "qnn_executorch.h"

using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::FileDataLoader;
using executorch::runtime::Error;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;

std::unordered_map<
    std::string,
    std::unique_ptr<executorch::backends::qnn::QnnExecuTorchIdlWrapper>>
    loaded_models;

AEEResult qnn_executorch_open(const char* uri, remote_handle64* h) {
  FARF(RUNTIME_HIGH, __func__);
  executorch::runtime::runtime_init();
  return 0;
}

AEEResult qnn_executorch_close(remote_handle64 h) {
  FARF(RUNTIME_HIGH, __func__);
  loaded_models.clear();
  return 0;
}

AEEResult qnn_executorch_load(
    remote_handle64 _h,
    const char* pte_path,
    const int method_index) {
  FARF(RUNTIME_HIGH, __func__);
  std::string key(pte_path);
  if (loaded_models.count(key)) {
    FARF(RUNTIME_HIGH, "%s is already loaded.", pte_path);
  } else {
    loaded_models[key] =
        std::make_unique<executorch::backends::qnn::QnnExecuTorchIdlWrapper>(
            pte_path, method_index);
  }
  return 0;
}

AEEResult qnn_executorch_execute_all(
    remote_handle64 _h,
    const char* pte_path,
    const char* input_list_path,
    const char* output_folder_path,
    int* num_inferences_performed,
    double* total_execute_interval,
    double* total_read_file_interval,
    double* total_save_file_interval) {
  FARF(RUNTIME_HIGH, __func__);
  AEEResult status = 0;
  std::string key(pte_path);
  if (loaded_models.count(key)) {
    Error res = loaded_models[key]->execute_all(
        input_list_path,
        output_folder_path,
        *num_inferences_performed,
        *total_execute_interval,
        *total_read_file_interval,
        *total_save_file_interval);
    if (res != Error::Ok) {
      FARF(RUNTIME_ERROR, "Failed to execute_all.");
      status = -1;
    }
  } else {
    FARF(
        RUNTIME_ERROR,
        "Unable to execute_all, please ensure the model %s is loaded.",
        pte_path);
    status = -1;
  }
  return status;
}

AEEResult qnn_executorch_unload(remote_handle64 _h, const char* pte_path) {
  FARF(RUNTIME_HIGH, __func__);
  AEEResult status = 0;
  std::string key(pte_path);
  if (loaded_models.count(key)) {
    loaded_models.erase(key);
  } else {
    FARF(
        RUNTIME_ERROR,
        "Unable to get unload model, please ensure the model %s is loaded.",
        pte_path);
    status = -1;
  }
  return status;
}

AEEResult qnn_executorch_enable_intermediate_tensor_dump(
    remote_handle64 _h,
    const char* pte_path,
    const int debug_buffer_size) {
  FARF(RUNTIME_HIGH, __func__);
  AEEResult status = 0;
  std::string key(pte_path);
  if (loaded_models.count(key)) {
    if (loaded_models[key]->enable_intermediate_tensor_dump(
            debug_buffer_size) != Error::Ok) {
      status = -1;
    }
  } else {
    FARF(
        RUNTIME_ERROR,
        "Unable to enable_intermediate_tensor_dump, please ensure the model %s is loaded.",
        pte_path);
    status = -1;
  }
  return status;
}

AEEResult qnn_executorch_dump_etdp(
    remote_handle64 _h,
    const char* pte_path,
    const char* etdump_path) {
  FARF(RUNTIME_HIGH, __func__);
  AEEResult status = 0;
  std::string key(pte_path);
  if (loaded_models.count(key)) {
    if (loaded_models[key]->dump_etdp(etdump_path) != Error::Ok) {
      status = -1;
    }
  } else {
    FARF(
        RUNTIME_ERROR,
        "Unable to dump_etdp, please ensure the model %s is loaded.",
        pte_path);
    status = -1;
  }
  return status;
}

AEEResult qnn_executorch_dump_intermediate_tensor(
    remote_handle64 _h,
    const char* pte_path,
    const char* debug_output_path) {
  FARF(RUNTIME_HIGH, __func__);
  AEEResult status = 0;
  std::string key(pte_path);
  if (loaded_models.count(key)) {
    if (loaded_models[key]->dump_intermediate_tensor(debug_output_path) !=
        Error::Ok) {
      status = -1;
    }
  } else {
    FARF(
        RUNTIME_ERROR,
        "Unable to enable_intermediate_tensor_dump, please ensure the model %s is loaded.",
        pte_path);
    status = -1;
  }
  return status;
}
