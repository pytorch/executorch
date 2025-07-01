/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>

#include <QnnTypes.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnBackendFactory.h>
#include <executorch/backends/qualcomm/runtime/backends/ir/IrContext.h>

#include "QnnWrapperUtils.hpp"
namespace executorch {
namespace backends {
namespace qnn {

using executorch::runtime::Error;
using QnnModel_composeGraphsFromDlc = qnn_wrapper_api::ModelError_t (*)(...);
class QnnDlcManager {
 public:
  QnnDlcManager(
      const QnnExecuTorchContextBinary& qnn_context_blob,
      const QnnExecuTorchOptions* options);

  qnn_wrapper_api::GraphInfoPtr_t* GetQnnDlcGraphInfoPtr() {
    return qnn_dlc_graph_info_;
  }

  uint32_t GetQnnDlcGraphInfoNum() {
    return qnn_dlc_graph_info_num_;
  }

  std::unique_ptr<BackendConfigParameters> backend_params_ptr_ =
      std::make_unique<BackendConfigParameters>();

  void ResetBackendParams();
  void ResetLogger();
  void TerminateAllBackends();

  Error SetUpDlcEnvironment(const Qnn_Version_t& coreApiVersion);

  Error RegisterGraphsFromDLC(
      const QnnImplementation& implementation,
      QnnBackend* backend,
      QnnContext* context,
      QnnBackendCache* cache);

 private:
  static constexpr const char* library_name_ = "libQnnIr.so";
  QnnImplementation qnn_loaded_backend_;
  std::unique_ptr<QnnLogger> logger_;

  const QnnExecuTorchContextBinary& qnn_context_blob_;
  const QnnExecuTorchOptions* options_;

  static constexpr const char* dlc_lib_ = "libQnnModelDlc.so";
  qnn_wrapper_api::GraphInfoPtr_t* qnn_dlc_graph_info_ = nullptr;
  uint32_t qnn_dlc_graph_info_num_ = 0;

  Error LoadQnnIrLibrary();

  Error Create();

  Error Configure();
};
} // namespace qnn
} // namespace backends
} // namespace executorch
