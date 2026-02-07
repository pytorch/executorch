/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/backends/lpai/LpaiBackendCustomConfig.h>

namespace executorch {
namespace backends {
namespace qnn {

LpaiBackendCustomConfig::LpaiBackendCustomConfig(const SocInfo* soc_info)
    : soc_info_(soc_info) {}

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
          {LpaiHardwareVersion::V1, QNN_LPAI_BACKEND_HW_VERSION_V1},
          {LpaiHardwareVersion::V2, QNN_LPAI_BACKEND_HW_VERSION_V2},
          {LpaiHardwareVersion::V3, QNN_LPAI_BACKEND_HW_VERSION_V3},
          {LpaiHardwareVersion::V4, QNN_LPAI_BACKEND_HW_VERSION_V4},
          {LpaiHardwareVersion::V5, QNN_LPAI_BACKEND_HW_VERSION_V5},
          {LpaiHardwareVersion::V5_1, QNN_LPAI_BACKEND_HW_VERSION_V5_1},
          {LpaiHardwareVersion::V6, QNN_LPAI_BACKEND_HW_VERSION_V6},
          {LpaiHardwareVersion::V7, QNN_LPAI_BACKEND_HW_VERSION_V7},
      };

  p_custom_config = AllocBackendCustomConfig();
  auto p_hw_info = AllocHwInfo();
  p_custom_config->option = QNN_LPAI_BACKEND_CUSTOM_CFG_HW_INFO;
  auto lpai_info = soc_info_->lpai_info();
  if (lpai_info && lpai_hw_ver.count(lpai_info->lpai_hardware_version())) {
    p_hw_info->hwVersion = lpai_hw_ver[lpai_info->lpai_hardware_version()];
  }
  p_hw_info->lpaiTarget = QNN_LPAI_BACKEND_TARGET_ADSP;
  p_custom_config->config = p_hw_info;
  ret.push_back(static_cast<QnnBackend_CustomConfig_t>(p_custom_config));
  return ret;
}

} // namespace qnn
} // namespace backends
} // namespace executorch
