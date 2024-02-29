/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/runtime/Utils.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnDeviceCommon.h>
#include <executorch/backends/qualcomm/runtime/backends/htpbackend/HtpDevicePlatformInfoConfig.h>

#include <memory>

#include "HTP/QnnHtpDevice.h"
namespace torch {
namespace executor {
namespace qnn {
class HtpDevice : public QnnDevice {
 public:
  HtpDevice(
      const QnnImplementation& implementation,
      QnnLogger* logger,
      const QnnExecuTorchHtpBackendOptions& htp_options)
      : QnnDevice(implementation, logger), htp_options_(htp_options) {
    htp_device_platform_info_config_ =
        std::make_unique<HtpDevicePlatformInfoConfig>(htp_options);
  }
  ~HtpDevice();

  // Defines Qnn performance mode vote types for htpbackend
  enum PerformanceModeVoteType {
    kNoVote = 0,
    kUpVote = 1,
    kDownVote = 2,
  };

 protected:
  Error MakeConfig(std::vector<const QnnDevice_Config_t*>& config) override;

  Error AfterCreateDevice() override;

 private:
  void PerformanceVote();
  void ReleasePerformanceVote();

  inline bool IsPerfModeEnabled() {
    return htp_options_.performance_mode !=
        QnnExecuTorchHtpPerformanceMode::kHtpDefault;
  }

  template <typename T>
  std::vector<std::add_pointer_t<std::add_const_t<T>>> ObtainNullTermPtrVector(
      const std::vector<T>& vec) {
    std::vector<std::add_pointer_t<std::add_const_t<T>>> ret;
    for (auto& elem : vec) {
      ret.push_back(&elem);
    }
    ret.push_back(nullptr);
    return ret;
  }

  QnnHtpDevice_CustomConfig_t* AllocDeviceCustomConfig();
  std::unique_ptr<HtpDevicePlatformInfoConfig> htp_device_platform_info_config_;

  std::vector<std::unique_ptr<QnnHtpDevice_CustomConfig_t>> htp_device_config_;
  std::vector<QnnDevice_Config_t> device_config_;

  std::uint32_t powerconfig_client_id_{0};
  QnnHtpDevice_PerfInfrastructure_t owned_htp_perf_infra_ =
      QNN_HTP_DEVICE_PERF_INFRASTRUCTURE_INIT;
  QnnHtpDevice_PerfInfrastructure_t* htp_perf_infra_{nullptr};
  std::vector<QnnHtpPerfInfrastructure_PowerConfig_t> perf_power_configs_;
  std::vector<QnnHtpPerfInfrastructure_PowerConfig_t> down_vote_power_configs_;
  std::vector<QnnHtpPerfInfrastructure_PowerConfig_t> rpc_power_configs_;
  std::vector<const QnnHtpPerfInfrastructure_PowerConfig_t*>
      rpc_power_configs_ptr_;
  std::vector<const QnnHtpPerfInfrastructure_PowerConfig_t*>
      perf_power_configs_ptr_;
  std::vector<const QnnHtpPerfInfrastructure_PowerConfig_t*>
      down_vote_power_configs_ptr_;

  QnnExecuTorchHtpBackendOptions htp_options_;
};
} // namespace qnn
} // namespace executor
} // namespace torch
