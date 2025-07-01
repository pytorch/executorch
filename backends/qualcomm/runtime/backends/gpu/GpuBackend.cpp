/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/backends/gpu/GpuBackend.h>

#include "GPU/QnnGpuCommon.h"

namespace executorch {
namespace backends {
namespace qnn {

using executorch::runtime::Error;

GpuBackend::GpuBackend(
    const QnnImplementation& implementation,
    QnnLogger* logger,
    const QnnExecuTorchGpuBackendOptions* gpu_options)
    : QnnBackend(implementation, logger) {
  gpu_backend_custom_config_ =
      std::make_unique<GpuBackendCustomConfig>(gpu_options);
}

Qnn_Version_t GpuBackend::GetExpectedBackendVersion() const {
  Qnn_Version_t backend_version;
  backend_version.major = QNN_GPU_API_VERSION_MAJOR;
  backend_version.minor = QNN_GPU_API_VERSION_MINOR;
  backend_version.patch = QNN_GPU_API_VERSION_PATCH;
  return backend_version;
}

bool GpuBackend::IsProfileEventTypeParentOfNodeTime(
    QnnProfile_EventType_t event_type) {
  return (event_type == QNN_PROFILE_EVENTTYPE_EXECUTE);
}

Error GpuBackend::MakeConfig(std::vector<const QnnBackend_Config_t*>& config) {
  const std::vector<QnnBackend_CustomConfig_t>& backend_custom_config =
      gpu_backend_custom_config_->CreateBackendCustomConfig();

  uint32_t num_custom_configs = backend_custom_config.size();
  backend_config_.resize(num_custom_configs);
  // +1 for null terminated
  config.reserve(num_custom_configs + 1);

  for (std::size_t i = 0; i < num_custom_configs; ++i) {
    backend_config_[i].option = QNN_BACKEND_CONFIG_OPTION_CUSTOM;
    backend_config_[i].customConfig = backend_custom_config[i];
    config.push_back(&backend_config_[i]);
  }

  config.push_back(nullptr);
  return Error::Ok;
}

} // namespace qnn
} // namespace backends
} // namespace executorch
