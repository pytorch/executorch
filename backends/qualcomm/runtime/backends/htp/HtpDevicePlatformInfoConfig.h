/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/qc_compiler_spec_generated.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnDeviceCommon.h>

#include <memory>
#include <vector>

#include "HTP/QnnHtpDevice.h"
namespace executorch {
namespace backends {
namespace qnn {
using namespace qnn_delegate;
class HtpDevicePlatformInfoConfig {
 public:
  explicit HtpDevicePlatformInfoConfig(
      const QnnExecuTorchHtpBackendOptions* htp_options,
      QnnImplementation* implementation,
      QnnLogger* logger)
      : htp_options_(htp_options),
        implementation_(implementation),
        logger_(logger) {}

  ~HtpDevicePlatformInfoConfig() {
    // to prevent misleading error message: e.g. <E> Invalid platform info (nil)
    // check if platform_info_ has been queried first
    if (platform_info_) {
      implementation_->GetQnnInterface().qnn_device_free_platform_info(
          logger_->GetHandle(), platform_info_);
    }
  }

  std::vector<QnnDevice_PlatformInfo_t*> CreateDevicePlatformInfo(
      const SocInfo* qcom_target_soc_info);

 private:
  QnnDevice_PlatformInfo_t* AllocDevicePlatformInfo() {
    htp_platform_info_.emplace_back(
        std::make_unique<QnnDevice_PlatformInfo_t>());
    htp_platform_info_.back()->version =
        QNN_DEVICE_PLATFORM_INFO_VERSION_UNDEFINED;
    return htp_platform_info_.back().get();
  }

  QnnDevice_HardwareDeviceInfo_t* AllocHwDeviceInfo() {
    htp_hw_device_info_.emplace_back(
        std::make_unique<QnnDevice_HardwareDeviceInfo_t>());
    htp_hw_device_info_.back()->version =
        QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_UNDEFINED;
    return htp_hw_device_info_.back().get();
  }

  QnnDevice_CoreInfo_t* AllocCoreInfo() {
    htp_core_info_.emplace_back(QnnDevice_CoreInfo_t());
    htp_core_info_.back().version = QNN_DEVICE_CORE_INFO_VERSION_UNDEFINED;
    return &htp_core_info_.back();
  }

  QnnHtpDevice_DeviceInfoExtension_t* AllocDeviceInfoExtension() {
    htp_device_info_extension_.emplace_back(
        std::make_unique<QnnHtpDevice_DeviceInfoExtension_t>());
    htp_device_info_extension_.back()->devType = QNN_HTP_DEVICE_TYPE_UNKNOWN;
    return htp_device_info_extension_.back().get();
  }

  [[maybe_unused]] const QnnExecuTorchHtpBackendOptions* htp_options_;

  std::vector<std::unique_ptr<QnnDevice_PlatformInfo_t>> htp_platform_info_;
  std::vector<std::unique_ptr<QnnDevice_HardwareDeviceInfo_t>>
      htp_hw_device_info_;
  std::vector<QnnDevice_CoreInfo_t> htp_core_info_;
  std::vector<std::unique_ptr<QnnHtpDevice_DeviceInfoExtension_t>>
      htp_device_info_extension_;
  const QnnDevice_PlatformInfo_t* platform_info_{nullptr};
  QnnImplementation* implementation_;
  QnnLogger* logger_;
};
} // namespace qnn
} // namespace backends
} // namespace executorch
