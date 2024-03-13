/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/backends/htpbackend/HtpDeviceCustomConfig.h>
namespace torch {
namespace executor {
namespace qnn {
std::vector<QnnDevice_CustomConfig_t>
HtpDeviceCustomConfig::CreateDeviceCustomConfig(
    const SocInfo* qcom_target_soc_info) {
  std::vector<QnnDevice_CustomConfig_t> ret;
  QnnHtpDevice_CustomConfig_t* p_custom_config = nullptr;

  p_custom_config = AllocDeviceCustomConfig();
  p_custom_config->option = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
  p_custom_config->socModel =
      static_cast<uint32_t>(qcom_target_soc_info->soc_model());
  ret.push_back(static_cast<QnnDevice_CustomConfig_t>(p_custom_config));

  return ret;
}
} // namespace qnn
} // namespace executor
} // namespace torch
