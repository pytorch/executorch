/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/qc_compiler_spec_generated.h>
#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnBackendCache.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnBackendCommon.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnContextCommon.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnDeviceCommon.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnGraphCommon.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnImplementation.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnLogger.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnMemManager.h>
#include <executorch/backends/qualcomm/runtime/backends/gpu/GpuBackend.h>
#include <executorch/backends/qualcomm/runtime/backends/gpu/GpuContext.h>
#include <executorch/backends/qualcomm/runtime/backends/gpu/GpuDevice.h>
#include <executorch/backends/qualcomm/runtime/backends/gpu/GpuGraph.h>
#include <executorch/backends/qualcomm/runtime/backends/htp/HtpBackend.h>
#include <executorch/backends/qualcomm/runtime/backends/htp/HtpBackendCache.h>
#include <executorch/backends/qualcomm/runtime/backends/htp/HtpContext.h>
#include <executorch/backends/qualcomm/runtime/backends/htp/HtpDevice.h>
#include <executorch/backends/qualcomm/runtime/backends/htp/HtpGraph.h>

#include <memory>
namespace executorch {
namespace backends {
namespace qnn {

class QnnDlcManager;
typedef enum { UNINITIALIZED, INITIALIZED } BackendInitializeState;

// @brief Struct containing all handles for a given QNN backend
typedef struct BackendConfigParameters {
  std::unique_ptr<QnnBackend> qnn_backend_ptr_;
  BackendInitializeState backend_init_state_;
  std::unique_ptr<QnnContext> qnn_context_ptr_;
  std::unique_ptr<QnnDevice> qnn_device_ptr_;
  std::unique_ptr<QnnGraph> qnn_graph_ptr_;
  std::unique_ptr<QnnMemManager> qnn_mem_manager_ptr_;
  std::unique_ptr<QnnBackendCache> qnn_backend_cache_ptr_;

  // Default ctor
  BackendConfigParameters()
      : qnn_backend_ptr_(nullptr),
        backend_init_state_(BackendInitializeState::UNINITIALIZED),
        qnn_context_ptr_(nullptr),
        qnn_device_ptr_(nullptr),
        qnn_graph_ptr_(nullptr),
        qnn_mem_manager_ptr_(nullptr),
        qnn_backend_cache_ptr_(nullptr) {}
  // Default dtor
  ~BackendConfigParameters() {
    qnn_graph_ptr_.reset();
    qnn_backend_cache_ptr_.reset();
    qnn_mem_manager_ptr_.reset();
    qnn_context_ptr_.reset();
    qnn_device_ptr_.reset();
    qnn_backend_ptr_.reset();
    backend_init_state_ = BackendInitializeState::UNINITIALIZED;
  }

} BackendConfigParameters;

class QnnBackendFactory {
 public:
  std::unique_ptr<BackendConfigParameters> Create(
      const QnnImplementation& implementation,
      QnnLogger* logger,
      const QnnExecuTorchContextBinary& qnn_context_blob,
      const QnnExecuTorchOptions* options,
      QnnDlcManager* qnn_dlc_manager);
};
} // namespace qnn
} // namespace backends
} // namespace executorch
