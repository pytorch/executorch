/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/backends/htp/HtpDevicePlatformInfoConfig.h>
namespace executorch {
namespace backends {
namespace qnn {
std::vector<QnnDevice_PlatformInfo_t*>
HtpDevicePlatformInfoConfig::CreateDevicePlatformInfo(
    const SocInfo* /*qcom_target_soc_info*/) {
  std::vector<QnnDevice_PlatformInfo_t*> ret;
  auto error = implementation_->GetQnnInterface().qnn_device_get_platform_info(
      logger_->GetHandle(), &platform_info_);
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Failed to get platform info via QNN with error %lu",
        QNN_GET_ERROR_CODE(error));
    return ret;
  }
  if (platform_info_->v1.numHwDevices <= htp_options_->device_id()) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Device id (%d) exceeds current hardware capability: %d",
        htp_options_->device_id(),
        platform_info_->v1.numHwDevices);
    return ret;
  }
  size_t num_device_cores =
      platform_info_->v1.hwDevices[htp_options_->device_id()].v1.numCores;
  if (htp_options_->core_ids()->size() >= num_device_cores) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Number of cores (%d) exceeds current hardware capability: %d",
        htp_options_->core_ids()->size(),
        num_device_cores);
    return ret;
  }

  QnnDevice_PlatformInfo_t* p_platform_info = AllocDevicePlatformInfo();
  p_platform_info->version = QNN_DEVICE_PLATFORM_INFO_VERSION_1;
  p_platform_info->v1.numHwDevices = 1;

  QnnDevice_HardwareDeviceInfo_t* p_hw_device_info = AllocHwDeviceInfo();
  p_hw_device_info->version = QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1;
  p_hw_device_info->v1.deviceId = htp_options_->device_id();
  // TODO: support off-chip device type if necessary
  p_hw_device_info->v1.deviceType = 0;
  p_hw_device_info->v1.numCores = htp_options_->core_ids()->size();

  for (size_t i = 0; i < p_hw_device_info->v1.numCores; ++i) {
    QnnDevice_CoreInfo_t* p_core_info = AllocCoreInfo();
    // memory of core info is maintained by QNN, should be safe to use it
    // directly
    if (htp_options_->core_ids()->data()[i] >= num_device_cores) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Core id (%d) exceeds current hardware capability: %d",
          htp_options_->core_ids()->data()[i],
          num_device_cores);
      return ret;
    }
    *p_core_info = platform_info_->v1.hwDevices[htp_options_->device_id()]
                       .v1.cores[htp_options_->core_ids()->data()[i]];
  }
  p_hw_device_info->v1.cores = htp_core_info_.data();

  p_platform_info->v1.hwDevices = p_hw_device_info;
  ret.push_back(p_platform_info);

  return ret;
}
} // namespace qnn
} // namespace backends
} // namespace executorch
