/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/backends/gpu/GpuBackendCustomConfig.h>

namespace executorch {
namespace backends {
namespace qnn {

GpuBackendCustomConfig::GpuBackendCustomConfig(
    const QnnExecuTorchGpuBackendOptions* gpu_options)
    : gpu_options_(gpu_options) {}

QnnGpuBackend_CustomConfig_t*
GpuBackendCustomConfig::AllocBackendCustomConfig() {
  gpu_backend_config_.emplace_back(
      std::make_unique<QnnGpuBackend_CustomConfig_t>());
  gpu_backend_config_.back()->option = QNN_GPU_BACKEND_CONFIG_OPTION_UNDEFINED;
  return gpu_backend_config_.back().get();
}

std::vector<QnnBackend_CustomConfig_t>
GpuBackendCustomConfig::CreateBackendCustomConfig() {
  std::vector<QnnBackend_CustomConfig_t> ret;
  QnnGpuBackend_CustomConfig_t* p_custom_config = nullptr;

  if (gpu_options_->use_weight_sharing()) {
    p_custom_config = AllocBackendCustomConfig();
    p_custom_config->option =
        QNN_GPU_BACKEND_CONFIG_OPTION_WEIGHT_SHARING_ENABLED;
    p_custom_config->weightSharingEnabled = 1;
    ret.push_back(static_cast<QnnBackend_CustomConfig_t>(p_custom_config));
  }
  return ret;
}

} // namespace qnn
} // namespace backends
} // namespace executorch
