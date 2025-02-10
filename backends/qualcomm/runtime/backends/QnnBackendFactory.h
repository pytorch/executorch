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
#include <executorch/backends/qualcomm/runtime/backends/htpbackend/HtpBackend.h>
#include <executorch/backends/qualcomm/runtime/backends/htpbackend/HtpBackendCache.h>
#include <executorch/backends/qualcomm/runtime/backends/htpbackend/HtpContext.h>
#include <executorch/backends/qualcomm/runtime/backends/htpbackend/HtpDevice.h>
#include <executorch/backends/qualcomm/runtime/backends/htpbackend/HtpGraph.h>

#include <memory>
namespace executorch {
namespace backends {
namespace qnn {
typedef enum { UNINITIALIZED, INITIALIZED } BackendInitializeState;

// @brief Struct containing all handles for a given QNN backend
class BackendConfigurator {
 public:
  struct BackendConfigParameters {
    // backend
    std::unique_ptr<QnnBackend> qnn_backend_ptr_;
    BackendInitializeState backend_init_state_;
    std::unique_ptr<QnnContext> qnn_context_ptr_;
    std::unique_ptr<QnnDevice> qnn_device_ptr_;
    std::unique_ptr<QnnGraph> qnn_graph_ptr_;
    std::unique_ptr<QnnMemManager> qnn_mem_manager_ptr_;
    std::unique_ptr<QnnBackendCache> qnn_backend_cache_ptr_;

    // Ir backend
    std::unique_ptr<QnnBackend> qnn_comm_backend_ptr_;
    std::unique_ptr<QnnContext> qnn_comm_context_ptr_;
    std::unique_ptr<QnnDevice> qnn_comm_device_ptr_;
    std::unique_ptr<QnnGraph> qnn_comm_graph_ptr_;
    std::unique_ptr<QnnBackendCache> qnn_comm_backend_cache_ptr_;

    BackendConfigParameters()
        : qnn_backend_ptr_(nullptr),
          backend_init_state_(BackendInitializeState::UNINITIALIZED),
          qnn_context_ptr_(nullptr),
          qnn_device_ptr_(nullptr),
          qnn_graph_ptr_(nullptr),
          qnn_mem_manager_ptr_(nullptr),
          qnn_backend_cache_ptr_(nullptr),
          qnn_comm_backend_ptr_(nullptr),
          qnn_comm_context_ptr_(nullptr),
          qnn_comm_device_ptr_(nullptr),
          qnn_comm_graph_ptr_(nullptr),
          qnn_comm_backend_cache_ptr_(nullptr) {}

    ~BackendConfigParameters() {
      qnn_graph_ptr_.reset();
      qnn_backend_cache_ptr_.reset();
      qnn_mem_manager_ptr_.reset();
      qnn_context_ptr_.reset();
      qnn_device_ptr_.reset();
      qnn_backend_ptr_.reset();
      backend_init_state_ = BackendInitializeState::UNINITIALIZED;
      qnn_comm_graph_ptr_.reset();
      qnn_comm_backend_cache_ptr_.reset();
      qnn_comm_context_ptr_.reset();
      qnn_comm_device_ptr_.reset();
      qnn_comm_backend_ptr_.reset();
    }
  };

  std::unique_ptr<BackendConfigParameters> backend_params_ptr_;

  BackendConfigurator()
      : backend_params_ptr_(std::make_unique<BackendConfigParameters>()) {}

  bool IsBackendInitState() {
    return backend_params_ptr_->backend_init_state_ ==
        BackendInitializeState::UNINITIALIZED;
  }

  Error configure_qnn_backend_cache() {
    if (backend_params_ptr_->qnn_comm_backend_cache_ptr_ != nullptr) {
      ET_CHECK_OR_RETURN_ERROR(
          backend_params_ptr_->qnn_comm_backend_cache_ptr_->Configure() ==
              Error::Ok,
          Internal,
          "Fail to configure Qnn common backend cache");
    }
    return backend_params_ptr_->qnn_backend_cache_ptr_->Configure();
  }

  Error configure_qnn_backend() {
    if (backend_params_ptr_->qnn_comm_backend_ptr_ != nullptr) {
      ET_CHECK_OR_RETURN_ERROR(
          backend_params_ptr_->qnn_comm_backend_ptr_->Configure() == Error::Ok,
          Internal,
          "Fail to configure Qnn common backend");
    }
    return backend_params_ptr_->qnn_backend_ptr_->Configure();
  }

  Error configure_qnn_device() {
    return backend_params_ptr_->qnn_device_ptr_->Configure();
  }

  Error configure_qnn_context() {
    if (backend_params_ptr_->qnn_comm_context_ptr_ != nullptr) {
      ET_CHECK_OR_RETURN_ERROR(
          backend_params_ptr_->qnn_comm_context_ptr_->Configure() == Error::Ok,
          Internal,
          "Fail to configure Qnn common context");
    }
    return backend_params_ptr_->qnn_context_ptr_->Configure();
  }

  Error configure_qnn_graph(const bool skip_create) {
    QnnContext* qnn_context_ptr = backend_params_ptr_->qnn_context_ptr_.get();
    QnnGraph* qnn_graph_ptr = backend_params_ptr_->qnn_graph_ptr_.get();
    for (const std::string& graph_name : qnn_context_ptr->GetGraphNames()) {
      if (qnn_graph_ptr->Configure(graph_name, skip_create) != Error::Ok) {
        return Error::Internal;
      }
    }
    if (backend_params_ptr_->qnn_comm_context_ptr_ != nullptr) {
      qnn_context_ptr = backend_params_ptr_->qnn_comm_context_ptr_.get();
      qnn_graph_ptr = backend_params_ptr_->qnn_comm_graph_ptr_.get();
    }
    for (const std::string& graph_name : qnn_context_ptr->GetGraphNames()) {
      if (qnn_graph_ptr->Configure(graph_name, skip_create) != Error::Ok) {
        return Error::Internal;
      }
    }
    return executorch::runtime::Error::Ok;
  }
};

struct QnnBackends {
  QnnImplementation* qnn_loaded_backend_ptr_;
  QnnImplementation* qnn_loaded_comm_backend_ptr_;

  QnnBackends()
      : qnn_loaded_backend_ptr_(nullptr),
        qnn_loaded_comm_backend_ptr_(nullptr) {}

  void TerminateAllBackends() {
    if (qnn_loaded_backend_ptr_) {
      qnn_loaded_backend_ptr_->TerminateAllBackends();
    }
    if (qnn_loaded_comm_backend_ptr_) {
      qnn_loaded_comm_backend_ptr_->TerminateAllBackends();
    }
  }
};

class QnnBackendFactory {
 public:
  std::unique_ptr<BackendConfigurator> Create(
      const QnnBackends& implementations,
      QnnLogger* logger,
      const QnnExecuTorchContextBinary& qnn_context_blob,
      const QnnExecuTorchOptions* options);
};
} // namespace qnn
} // namespace backends
} // namespace executorch
