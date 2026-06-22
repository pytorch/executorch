/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/backends/lpai/LpaiBackendCustomConfig.h>
#include <unordered_map>

namespace executorch {
namespace backends {
namespace qnn {

LpaiBackendCustomConfig::LpaiBackendCustomConfig(
    const SocInfo* soc_info,
    const QnnExecuTorchLpaiBackendOptions* lpai_options)
    : soc_info_(soc_info), lpai_options_(lpai_options) {}

QnnLpaiBackend_CustomConfig_t*
LpaiBackendCustomConfig::AllocBackendCustomConfig() {
  lpai_backend_config_.emplace_back(
      std::make_unique<QnnLpaiBackend_CustomConfig_t>());
  lpai_backend_config_.back()->option = QNN_LPAI_BACKEND_CUSTOM_CFG_UNDEFINED;
  return lpai_backend_config_.back().get();
}

QnnLpaiBackend_CustomConfigHwInfo_t* LpaiBackendCustomConfig::AllocHwInfo() {
  lpai_hw_info_.emplace_back(
      std::make_unique<QnnLpaiBackend_CustomConfigHwInfo_t>());
  lpai_hw_info_.back()->hwVersion = QNN_LPAI_BACKEND_HW_VERSION_UNKNOWN;
  lpai_hw_info_.back()->lpaiTarget = QNN_LPAI_BACKEND_TARGET_UNKNOWN;
  return lpai_hw_info_.back().get();
}

std::vector<QnnBackend_CustomConfig_t>
LpaiBackendCustomConfig::CreateBackendCustomConfig() {
  std::vector<QnnBackend_CustomConfig_t> ret;
  QnnLpaiBackend_CustomConfig_t* p_custom_config = nullptr;

  std::unordered_map<LpaiHardwareVersion, QnnLpaiBackend_HwVersion_t>
      lpai_hw_ver = {
          {LpaiHardwareVersion::V6, QNN_LPAI_BACKEND_HW_VERSION_V6},
#if (QNN_API_VERSION_MAJOR >= 2 && QNN_API_VERSION_MINOR >= 29)
          {LpaiHardwareVersion::V7, QNN_LPAI_BACKEND_HW_VERSION_V7},
#endif
      };

  p_custom_config = AllocBackendCustomConfig();
  auto p_hw_info = AllocHwInfo();
  p_custom_config->option = QNN_LPAI_BACKEND_CUSTOM_CFG_HW_INFO;
  auto lpai_info = soc_info_->lpai_info();
  if (lpai_info && lpai_hw_ver.count(lpai_info->lpai_hardware_version())) {
    p_hw_info->hwVersion = lpai_hw_ver[lpai_info->lpai_hardware_version()];
  }
  p_hw_info->lpaiTarget =
      static_cast<QnnLpaiBackend_Target_t>(lpai_options_->target_env());
  p_custom_config->config = p_hw_info;
  ret.push_back(static_cast<QnnBackend_CustomConfig_t>(p_custom_config));
  return ret;
}

} // namespace qnn
} // namespace backends
} // namespace executorch
