/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef EXECUTORCH_QNN_EXECUTORCH_QNN_BACKEND_FACTORY_H_
#define EXECUTORCH_QNN_EXECUTORCH_QNN_BACKEND_FACTORY_H_
#include <executorch/backends/qnn/runtime/QnnExecuTorch.h>
#include <executorch/backends/qnn/runtime/backends/QnnBackendCommon.h>
#include <executorch/backends/qnn/runtime/backends/QnnContextCommon.h>
#include <executorch/backends/qnn/runtime/backends/QnnDeviceCommon.h>
#include <executorch/backends/qnn/runtime/backends/QnnFactory.h>
#include <executorch/backends/qnn/runtime/backends/QnnGraphCommon.h>
#include <executorch/backends/qnn/runtime/backends/QnnImplementation.h>
#include <executorch/backends/qnn/runtime/backends/QnnLogger.h>
#include <executorch/backends/qnn/runtime/backends/htpbackend/HtpBackend.h>
#include <executorch/backends/qnn/runtime/backends/htpbackend/HtpContext.h>
#include <executorch/backends/qnn/runtime/backends/htpbackend/HtpDevice.h>
#include <executorch/backends/qnn/runtime/backends/htpbackend/HtpGraph.h>

#include <memory>
namespace torch {
namespace executor {
namespace qnn {
typedef enum { UNINITIALIZED, INITIALIZED } BackendInitializeState;

// @brief Struct containing all handles for a given QNN backend
typedef struct BackendConfigParameters {
  std::unique_ptr<QnnBackend> qnn_backend_ptr_;
  BackendInitializeState backend_init_state_;
  std::unique_ptr<QnnContext> qnn_context_ptr_;
  std::unique_ptr<QnnDevice> qnn_device_ptr_;
  std::unique_ptr<QnnGraph> qnn_graph_ptr_;

  // Default ctor
  BackendConfigParameters()
      : qnn_backend_ptr_(nullptr),
        backend_init_state_(BackendInitializeState::UNINITIALIZED),
        qnn_context_ptr_(nullptr),
        qnn_device_ptr_(nullptr),
        qnn_graph_ptr_(nullptr) {}
  // Default dtor
  ~BackendConfigParameters() {
    qnn_graph_ptr_.reset();
    qnn_context_ptr_.reset();
    qnn_device_ptr_.reset();
    qnn_backend_ptr_.reset();
    backend_init_state_ = BackendInitializeState::UNINITIALIZED;
  }

} BackendConfigParameters;

class QnnBackendFactory : public QnnFactory {
 public:
  std::unique_ptr<BackendConfigParameters> Create(
      const QnnImplementation& implementation, QnnLogger* logger,
      const QnnExecuTorchContextBinary& qnn_context_blob,
      const QnnExecuTorchBackendType& backend_type,
      const std::string& graph_name,
      const QnnExecuTorchHtpBackendOptions& htp_options);
};
}  // namespace qnn
}  // namespace executor
}  // namespace torch

#endif  // EXECUTORCH_QNN_EXECUTORCH_QNN_BACKEND_FACTORY_H_
