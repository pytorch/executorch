/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/schema_generated.h>

#include <memory>
#include <vector>

#include "HTP/QnnHtpDevice.h"
namespace torch {
namespace executor {
namespace qnn {
using namespace qnn_delegate;
class HtpDeviceCustomConfig {
 public:
  explicit HtpDeviceCustomConfig(
      const QnnExecuTorchHtpBackendOptions* htp_options)
      : htp_options_(htp_options) {}
  std::vector<QnnDevice_CustomConfig_t> CreateDeviceCustomConfig(
      const SocInfo* qcom_target_soc_info);

  QnnHtpDevice_CustomConfig_t* AllocDeviceCustomConfig() {
    htp_device_config_.emplace_back(
        std::make_unique<QnnHtpDevice_CustomConfig_t>());
    htp_device_config_.back()->option = QNN_HTP_DEVICE_CONFIG_OPTION_UNKNOWN;
    return htp_device_config_.back().get();
  }

 private:
  [[maybe_unused]] const QnnExecuTorchHtpBackendOptions* htp_options_;
  std::vector<std::unique_ptr<QnnHtpDevice_CustomConfig_t>> htp_device_config_;
};
} // namespace qnn
} // namespace executor
} // namespace torch
