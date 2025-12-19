/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnBackendUnifiedRegistry.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnLogger.h>
#include <executorch/backends/qualcomm/runtime/backends/gpu/GpuBackend.h>
#include <executorch/backends/qualcomm/runtime/backends/gpu/GpuDevice.h>
#include <executorch/backends/qualcomm/runtime/backends/htp/HtpBackend.h>
#include <executorch/backends/qualcomm/runtime/backends/htp/HtpDevice.h>

#include <string>

namespace executorch {
namespace backends {
namespace qnn {
using executorch::runtime::Error;

// Static instance for the singleton
QnnBackendUnifiedRegistry& QnnBackendUnifiedRegistry::GetInstance() {
  static QnnBackendUnifiedRegistry instance;
  return instance;
}

// Private constructor
QnnBackendUnifiedRegistry::QnnBackendUnifiedRegistry() = default;

// Destructor
QnnBackendUnifiedRegistry::~QnnBackendUnifiedRegistry() {
  CleanupExpired();
}

Error QnnBackendUnifiedRegistry::GetOrCreateBackendBundle(
    const QnnExecuTorchOptions* options,
    std::shared_ptr<QnnBackendBundle>& bundle) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Extract relevant parameters from options for creation and validation
  std::string current_lib_path = options->library_path()->str();
  QnnExecuTorchLogLevel current_log_level = get_option(options->log_level());
  QnnExecuTorchBackendType backend_type =
      options->backend_options()->backend_type();

  if (current_lib_path.empty()) {
    switch (backend_type) {
      case QnnExecuTorchBackendType::kHtpBackend: {
        current_lib_path = htp_library_name_;
        break;
      }
      case QnnExecuTorchBackendType::kGpuBackend: {
        current_lib_path = gpu_library_name_;
        break;
      }
      case QnnExecuTorchBackendType::kDspBackend:
      case QnnExecuTorchBackendType::kUndefinedBackend:
      default:
        QNN_EXECUTORCH_LOG_ERROR(
            "Unsupported backend type: %s",
            EnumNameQnnExecuTorchBackendType(backend_type));
        return Error::NotFound;
    }
  }

  // Check if resources already exist
  auto it = qnn_backend_bundles_map_.find(backend_type);
  if (it != qnn_backend_bundles_map_.end()) {
    // Create new shared_ptr that shares ownership of the managed object.
    if (auto existing_bundle = it->second.lock()) {
      bundle = existing_bundle;
      if (bundle->qnn_logger_ptr->GetLogLevel() != current_log_level) {
        bundle->qnn_logger_ptr = std::make_unique<QnnLogger>(
            bundle->implementation.get(), LoggingCallback, current_log_level);
      }
      QNN_EXECUTORCH_LOG_INFO(
          "Use cached backend bundle for current backend: %s",
          EnumNameQnnExecuTorchBackendType(backend_type));
      return Error::Ok;
    }
  }

  QNN_EXECUTORCH_LOG_INFO("Creating new backend bundle.");

  // 1. Create QnnImplementation and load qnn library
  std::unique_ptr<QnnImplementation> implementation =
      std::make_unique<QnnImplementation>(current_lib_path);
  auto config = GetImplementationConfig(options);
  Error ret = implementation->Load(config.get());
  ET_CHECK_OR_RETURN_ERROR(
      ret == Error::Ok, Internal, "Fail to load Qnn library");

  // 2. Create QnnLogger
  std::unique_ptr<QnnLogger> logger = std::make_unique<QnnLogger>(
      implementation.get(), LoggingCallback, current_log_level);

  // 3. Create QnnBackend (specific type based on options)
  // 4. Create QnnDevice (specific type based on options)
  std::unique_ptr<QnnBackend> backend = nullptr;
  std::unique_ptr<QnnDevice> device = nullptr;

  switch (backend_type) {
    case QnnExecuTorchBackendType::kHtpBackend: {
      auto htp_options = options->backend_options()->htp_options();
      backend =
          std::make_unique<HtpBackend>(implementation.get(), logger.get());
      device = std::make_unique<HtpDevice>(
          implementation.get(), logger.get(), options->soc_info(), htp_options);
      break;
    }
    case QnnExecuTorchBackendType::kGpuBackend: {
      auto gpu_options = options->backend_options()->gpu_options();
      backend = std::make_unique<GpuBackend>(
          implementation.get(), logger.get(), gpu_options);
      device = std::make_unique<GpuDevice>(implementation.get(), logger.get());
      break;
    }
    case QnnExecuTorchBackendType::kDspBackend:
    case QnnExecuTorchBackendType::kUndefinedBackend:
    default:
      return Error::NotFound;
  }
  ET_CHECK_OR_RETURN_ERROR(
      backend->Configure(options->op_package_options()) == Error::Ok,
      Internal,
      "Fail to configure Qnn backend");
  ET_CHECK_OR_RETURN_ERROR(
      device->Configure() == Error::Ok,
      Internal,
      "Fail to configure Qnn device");

  if (backend->VerifyQNNSDKVersion() != Error::Ok) {
    return Error::Internal;
  }

  bundle->implementation = std::move(implementation);
  bundle->qnn_logger_ptr = std::move(logger);
  bundle->qnn_backend_ptr = std::move(backend);
  bundle->qnn_device_ptr = std::move(device);
  qnn_backend_bundles_map_.emplace(
      backend_type, bundle); // Store weak_ptr to the bundle

  return Error::Ok;
}

void QnnBackendUnifiedRegistry::CleanupExpired() {
  std::lock_guard<std::mutex> lock(mutex_);

  for (auto it = qnn_backend_bundles_map_.begin();
       it != qnn_backend_bundles_map_.end();) {
    if (it->second.expired()) {
      it = qnn_backend_bundles_map_.erase(it);
    } else {
      ++it;
    }
  }
}

} // namespace qnn
} // namespace backends
} // namespace executorch
