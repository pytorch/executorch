/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/backends/htpbackend/HtpDevicePlatformInfoConfig.h>
namespace torch {
namespace executor {
namespace qnn {
std::vector<QnnDevice_PlatformInfo_t*>
HtpDevicePlatformInfoConfig::CreateDevicePlatformInfo(
    const SocInfo* qcom_target_soc_info) {
  std::vector<QnnDevice_PlatformInfo_t*> ret;
  QnnDevice_PlatformInfo_t* p_platform_info = nullptr;
  QnnDevice_HardwareDeviceInfo_t* p_hw_device_info = nullptr;
  QnnHtpDevice_DeviceInfoExtension_t* p_device_info_extension = nullptr;
  QnnDevice_CoreInfo_t* p_core_info = nullptr;

  p_platform_info = AllocDevicePlatformInfo();
  p_platform_info->version = QNN_DEVICE_PLATFORM_INFO_VERSION_1;
  p_platform_info->v1.numHwDevices = 1;

  p_hw_device_info = AllocHwDeviceInfo();
  p_hw_device_info->version = QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1;
  p_hw_device_info->v1.deviceId = 0;
  p_hw_device_info->v1.deviceType = 0;
  p_hw_device_info->v1.numCores = 1;

  p_device_info_extension = AllocDeviceInfoExtension();
  p_device_info_extension->devType = QNN_HTP_DEVICE_TYPE_ON_CHIP;
  p_device_info_extension->onChipDevice.vtcmSize =
      qcom_target_soc_info->htp_info()->vtcm_size_in_mb();
  // Given by user, default value is unsigned pd
  p_device_info_extension->onChipDevice.signedPdSupport =
      htp_options_->pd_session() == QnnExecuTorchHtpPdSession::kHtpSignedPd;
  p_device_info_extension->onChipDevice.socModel =
      static_cast<uint32_t>(qcom_target_soc_info->soc_model());
  p_device_info_extension->onChipDevice.arch = static_cast<QnnHtpDevice_Arch_t>(
      qcom_target_soc_info->htp_info()->htp_arch());
  // For Htp, dlbcSupport is true
  p_device_info_extension->onChipDevice.dlbcSupport = true;
  p_hw_device_info->v1.deviceInfoExtension = p_device_info_extension;

  p_core_info = AllocCoreInfo();
  p_core_info->version = QNN_DEVICE_CORE_INFO_VERSION_1;
  p_core_info->v1.coreId = 0;
  p_core_info->v1.coreType = 0;
  p_core_info->v1.coreInfoExtension = nullptr;
  p_hw_device_info->v1.cores = p_core_info;

  p_platform_info->v1.hwDevices = p_hw_device_info;
  ret.push_back(p_platform_info);

  return ret;
}
} // namespace qnn
} // namespace executor
} // namespace torch
