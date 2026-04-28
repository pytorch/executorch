/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <executorch/backends/qualcomm/runtime/QnnBackendOptions.h>
#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>

#include <QnnTypes.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnBackendFactory.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnBackendUnifiedRegistry.h>
#include <executorch/backends/qualcomm/runtime/backends/ir/IrContext.h>

namespace executorch {
namespace backends {
namespace qnn {

using executorch::runtime::Error;
class QnnDlcManager {
 public:
  QnnDlcManager(
      const QnnExecuTorchContextBinary& qnn_context_blob,
      const QnnExecuTorchOptions* options);

  QnnSystemContext_GraphInfo_t* GetQnnDlcGraphInfoPtr() {
    return graphs_;
  }

  uint32_t GetQnnDlcGraphInfoNum() {
    return num_graphs_;
  }

  std::unique_ptr<BackendConfigParameters> backend_params_ptr_ =
      std::make_unique<BackendConfigParameters>();
  std::unique_ptr<QnnBackendBundle> backend_bundle_ptr_ =
      std::make_unique<QnnBackendBundle>();

  void Destroy();

  Error SetUpDlcEnvironment(
      const Qnn_Version_t& coreApiVersion,
      const std::vector<std::string>& graph_names);

  Error RegisterGraphsFromDLC(
      QnnImplementation* implementation,
      QnnSystemImplementation* system_implementation,
      QnnBackend* backend,
      QnnContext* context,
      QnnBackendCache* cache) {
    const QnnSystemInterface& system_interface =
        system_implementation->GetQnnSystemInterface();

    // create dlc_handle
    QnnSystemDlc_Handle_t dlc_handle = nullptr;
    backend_bundle_ptr_->qnn_logger_ptr = std::make_unique<QnnLogger>(
        implementation,
        LoggingCallback,
        get_option(options_->log_level(), QNN_RUNTIME_LOG_LEVEL));

    Qnn_ErrorHandle_t error =
        system_interface.qnn_system_dlc_create_from_binary(
            /*logger=*/backend_bundle_ptr_->qnn_logger_ptr->GetHandle(),
            /*buffer=*/(const uint8_t*)qnn_context_blob_.buffer,
            /*bufferSize=*/qnn_context_blob_.nbytes,
            /*dlcHandle=*/&dlc_handle);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Can't create dlc from binary. Error %d.", QNN_GET_ERROR_CODE(error));
      return Error::Internal;
    }

    // compose graphs from dlc
    const QnnInterface_t* interface =
        implementation->GetQnnInterface().GetInterface();
    error = system_interface.qnn_system_dlc_compose_graphs(
        /*dlcHandle=*/dlc_handle,
        /*graphConfigs=*/nullptr,
        /*numGraphConfigs=*/0,
        /*backend=*/backend->GetHandle(),
        /*context=*/context->GetHandle(),
        /*backendInterface=*/*interface,
        /*graphVersion=*/QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1,
        /*graphs=*/&graphs_,
        /*numGraphs=*/&num_graphs_);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Can't compose graph from dlc. Error %d.", QNN_GET_ERROR_CODE(error));
      return Error::Internal;
    }

    for (uint32_t i = 0; i < num_graphs_; ++i) {
      auto& graphInfo = graphs_[i].graphInfoV1;
      cache->SetGraphNames(graphInfo.graphName);
    }

    error = system_interface.qnn_system_dlc_free(/*dlcHandle=*/dlc_handle);
    return Error::Ok;
  }

 private:
  static constexpr const char* library_name_ = "libQnnIr.so";

  const QnnExecuTorchContextBinary& qnn_context_blob_;
  const QnnExecuTorchOptions* options_;

  QnnSystemContext_GraphInfo_t* graphs_ = nullptr;
  uint32_t num_graphs_ = 0;

  Error LoadQnnIrLibrary();

  Error Create();

  Error Configure(const std::vector<std::string>& graph_names);
};
} // namespace qnn
} // namespace backends
} // namespace executorch
