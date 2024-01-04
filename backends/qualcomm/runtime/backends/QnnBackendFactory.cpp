/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/backends/QnnBackendFactory.h>
namespace torch {
namespace executor {
namespace qnn {
std::unique_ptr<BackendConfigParameters> QnnBackendFactory::Create(
    const QnnImplementation& implementation,
    QnnLogger* logger,
    const QnnExecuTorchContextBinary& qnn_context_blob,
    const QnnExecuTorchBackendType& backend_type,
    const std::string& graph_name,
    const QnnExecuTorchHtpBackendOptions& htp_options) {
  auto backend_params = std::make_unique<BackendConfigParameters>();
  switch (backend_type) {
    case kHtpBackend:
      backend_params->qnn_backend_ptr_ =
          std::make_unique<HtpBackend>(implementation, logger);
      backend_params->qnn_device_ptr_ =
          std::make_unique<HtpDevice>(implementation, logger, htp_options);

      backend_params->qnn_context_ptr_ = std::make_unique<HtpContext>(
          implementation,
          backend_params->qnn_backend_ptr_.get(),
          backend_params->qnn_device_ptr_.get(),
          qnn_context_blob,
          htp_options);

      backend_params->qnn_graph_ptr_ = std::make_unique<HtpGraph>(
          implementation,
          backend_params->qnn_context_ptr_.get(),
          graph_name,
          htp_options);
      backend_params->backend_init_state_ = BackendInitializeState::INITIALIZED;
      return backend_params;
      break;
    case kGpuBackend:
    case kDspBackend:
    case kUndefinedBackend:
    default:
      return nullptr;
  }

  // should not reach here
  return nullptr;
}
} // namespace qnn
} // namespace executor
} // namespace torch
