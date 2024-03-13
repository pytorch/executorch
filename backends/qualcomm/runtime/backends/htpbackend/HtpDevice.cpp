/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/backends/htpbackend/HtpDevice.h>

#include "HTP/QnnHtpCommon.h"
#include "Saver/QnnSaverCommon.h"

namespace torch {
namespace executor {
namespace qnn {

// constexpr config values
constexpr const int kSleepMinLatency = 40;
constexpr const int kSleepLowLatency = 100;
constexpr const int kSleepMediumLatency = 1000;
constexpr const int kSleepHighLatency = 2000;
constexpr const int kDcvsDisable = 0;
constexpr const int kDcvsEnable = 1;

// default rpc control latency - 100 us
constexpr const int kRpcControlLatency = 100;
// default rpc polling time for high power modes - 9999 us
constexpr const int kRpcPollingTimeHighPower = 9999;
// default rpc polling time for low power modes - 0 us
constexpr const int kRpcPollingTimeLowPower = 0;

// the number of Rpc Polling config
constexpr const int kNumRpcPollingPowerConfigs = 2;

namespace {
template <typename... Args>
Qnn_ErrorHandle_t HtpPerfInfraStubForSaver(Args... args) {
  return QNN_SUCCESS;
}

Error GetPerfInfra(
    const QnnInterface& qnn_interface,
    QnnHtpDevice_PerfInfrastructure_t* p_out) {
  if (qnn_interface.GetBackendId() == QNN_BACKEND_ID_SAVER) {
    p_out->createPowerConfigId = HtpPerfInfraStubForSaver;
    p_out->destroyPowerConfigId = HtpPerfInfraStubForSaver;
    p_out->setPowerConfig = HtpPerfInfraStubForSaver;
    p_out->setMemoryConfig = HtpPerfInfraStubForSaver;
    return Error::Ok;
  }

  QnnDevice_Infrastructure_t device_infra = nullptr;
  Qnn_ErrorHandle_t error =
      qnn_interface.qnn_device_get_infrastructure(&device_infra);

  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "HTP backend perf_infrastructure "
        "creation failed. Error %d",
        QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }

  auto* htp_infra = static_cast<QnnHtpDevice_Infrastructure_t*>(device_infra);
  if (htp_infra->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) {
    QNN_EXECUTORCH_LOG_ERROR(
        "HTP infra type = %d, which is "
        "not perf infra type.",
        htp_infra->infraType);
    return Error::Internal;
  }

  *p_out = htp_infra->perfInfra;
  return Error::Ok;
}

std::vector<QnnHtpPerfInfrastructure_PowerConfig_t> SetVotePowerConfig(
    const std::uint32_t power_config_id,
    const QnnExecuTorchHtpPerformanceMode perf_mode,
    const HtpDevice::PerformanceModeVoteType vote_type) {
  constexpr const int kNumConfigs = 1;
  std::vector<QnnHtpPerfInfrastructure_PowerConfig_t> power_configs(
      kNumConfigs);

  QnnHtpPerfInfrastructure_PowerConfig_t& dcvs_config = power_configs[0];

  dcvs_config.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  QnnHtpPerfInfrastructure_DcvsV3_t& dcvs_v3 = dcvs_config.dcvsV3Config;
  dcvs_v3.contextId = power_config_id;

  // Check DownVote before performance mode
  if (vote_type == HtpDevice::PerformanceModeVoteType::kDownVote) {
    dcvs_v3.setSleepDisable = 0; // false
    dcvs_v3.sleepDisable = 0;

    dcvs_v3.setDcvsEnable = 1; // true
    dcvs_v3.dcvsEnable = kDcvsEnable;

    dcvs_v3.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE;

    dcvs_v3.setSleepLatency = 1; // true
    dcvs_v3.sleepLatency = kSleepHighLatency;

    dcvs_v3.setBusParams = 1;
    dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS2;
    dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS2;
    dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS2;

    dcvs_v3.setCoreParams = 1;
    dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS2;
    dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS2;
    dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS2;

    return power_configs;
  }

  // Upvote
  dcvs_v3.setSleepDisable = 0;
  dcvs_v3.sleepDisable = 0;

  dcvs_v3.setDcvsEnable = 1;
  dcvs_v3.dcvsEnable = kDcvsDisable;

  dcvs_v3.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;

  // choose performance mode
  switch (perf_mode) {
    case QnnExecuTorchHtpPerformanceMode::kHtpBurst:
      dcvs_v3.setSleepLatency = 1; // true
      dcvs_v3.sleepLatency = kSleepMinLatency;

      dcvs_v3.setBusParams = 1;
      dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

      dcvs_v3.setCoreParams = 1;
      dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      break;
    case QnnExecuTorchHtpPerformanceMode::kHtpSustainedHighPerformance:
    case QnnExecuTorchHtpPerformanceMode::kHtpHighPerformance:
      dcvs_v3.setSleepLatency = 1; // true
      dcvs_v3.sleepLatency = kSleepLowLatency;

      dcvs_v3.setBusParams = 1;
      dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO;
      dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO;
      dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO;

      dcvs_v3.setCoreParams = 1;
      dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO;
      dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO;
      dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO;
      break;
    case QnnExecuTorchHtpPerformanceMode::kHtpPowerSaver:
      dcvs_v3.setSleepLatency = 1; // true
      dcvs_v3.sleepLatency = kSleepMediumLatency;

      dcvs_v3.setBusParams = 1;
      dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS;
      dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS;
      dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS;

      dcvs_v3.setCoreParams = 1;
      dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS;
      dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS;
      dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS;
      break;
    case QnnExecuTorchHtpPerformanceMode::kHtpLowPowerSaver:
      dcvs_v3.setSleepLatency = 1; // true
      dcvs_v3.sleepLatency = kSleepMediumLatency;

      dcvs_v3.setBusParams = 1;
      dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS2;
      dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS2;
      dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS2;

      dcvs_v3.setCoreParams = 1;
      dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS2;
      dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS2;
      dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS2;
      break;
    case QnnExecuTorchHtpPerformanceMode::kHtpHighPowerSaver:
      dcvs_v3.setSleepLatency = 1; // true
      dcvs_v3.sleepLatency = kSleepMediumLatency;

      dcvs_v3.setBusParams = 1;
      dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS_PLUS;

      dcvs_v3.setCoreParams = 1;
      dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      break;
    case QnnExecuTorchHtpPerformanceMode::kHtpLowBalanced:
      dcvs_v3.setSleepLatency = 1; // true
      dcvs_v3.sleepLatency = kSleepMediumLatency;

      dcvs_v3.setBusParams = 1;
      dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_NOM;
      dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_NOM;
      dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_NOM;

      dcvs_v3.setCoreParams = 1;
      dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_NOM;
      dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_NOM;
      dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_NOM;
      break;
    case QnnExecuTorchHtpPerformanceMode::kHtpBalanced:
      dcvs_v3.setSleepLatency = 1; // true
      dcvs_v3.sleepLatency = kSleepMediumLatency;

      dcvs_v3.setBusParams = 1;
      dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
      dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
      dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_NOM_PLUS;

      dcvs_v3.setCoreParams = 1;
      dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
      dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
      dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
      break;
    default:
      QNN_EXECUTORCH_LOG_ERROR(
          "Invalid performance profile "
          "%d to set power configs",
          perf_mode);
      break;
  }

  return power_configs;
}

std::vector<QnnHtpPerfInfrastructure_PowerConfig_t> SetRpcPollingPowerConfig(
    QnnExecuTorchHtpPerformanceMode perf_mode) {
  std::vector<QnnHtpPerfInfrastructure_PowerConfig_t> power_configs(
      kNumRpcPollingPowerConfigs);

  QnnHtpPerfInfrastructure_PowerConfig_t& rpc_control_latency =
      power_configs[0];
  QnnHtpPerfInfrastructure_PowerConfig_t& rpc_polling_time = power_configs[1];

  // configs
  rpc_control_latency.option =
      QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
  rpc_polling_time.option =
      QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;

  rpc_control_latency.rpcControlLatencyConfig = kRpcControlLatency;
  switch (perf_mode) {
    case QnnExecuTorchHtpPerformanceMode::kHtpBurst:
    case QnnExecuTorchHtpPerformanceMode::kHtpSustainedHighPerformance:
    case QnnExecuTorchHtpPerformanceMode::kHtpHighPerformance:
      rpc_polling_time.rpcPollingTimeConfig = kRpcPollingTimeHighPower;
      break;
    case QnnExecuTorchHtpPerformanceMode::kHtpPowerSaver:
    case QnnExecuTorchHtpPerformanceMode::kHtpLowPowerSaver:
    case QnnExecuTorchHtpPerformanceMode::kHtpHighPowerSaver:
    case QnnExecuTorchHtpPerformanceMode::kHtpLowBalanced:
    case QnnExecuTorchHtpPerformanceMode::kHtpBalanced:
    case QnnExecuTorchHtpPerformanceMode::kHtpDefault:
      rpc_polling_time.rpcPollingTimeConfig = kRpcPollingTimeLowPower;
      break;
    default:
      QNN_EXECUTORCH_LOG_ERROR(
          "Invalid performance profile "
          "%d to set power configs",
          perf_mode);
      break;
  }
  return power_configs;
}

} // namespace

HtpDevice::~HtpDevice() {
  if (htp_perf_infra_ != nullptr && powerconfig_client_id_ != 0 &&
      !down_vote_power_configs_ptr_.empty()) {
    htp_perf_infra_->setPowerConfig(
        powerconfig_client_id_, down_vote_power_configs_ptr_.data());
    htp_perf_infra_->destroyPowerConfigId(powerconfig_client_id_);
  } else if (htp_perf_infra_ != nullptr && powerconfig_client_id_ != 0) {
    htp_perf_infra_->destroyPowerConfigId(powerconfig_client_id_);
  }
}

Error HtpDevice::MakeConfig(std::vector<const QnnDevice_Config_t*>& config) {
  std::vector<QnnDevice_CustomConfig_t> device_custom_config =
      htp_device_custom_config_->CreateDeviceCustomConfig(
          qcom_target_soc_info_);
  QnnHtpDevice_CustomConfig_t* p_custom_config = nullptr;

  if (QNN_HTP_API_VERSION_MAJOR <= QNN_HTP_DEPRECATED_HTP_ARCH_VERSION_MAJOR &&
      QNN_HTP_API_VERSION_MINOR <= QNN_HTP_DEPRECATED_HTP_ARCH_VERSION_MINOR) {
    p_custom_config = htp_device_custom_config_->AllocDeviceCustomConfig();
    p_custom_config->option = QNN_HTP_DEVICE_CONFIG_OPTION_ARCH;
    p_custom_config->arch.deviceId = 0;
    p_custom_config->arch.arch = static_cast<QnnHtpDevice_Arch_t>(
        qcom_target_soc_info_->htp_info()->htp_arch());
    device_custom_config.push_back(
        static_cast<QnnDevice_CustomConfig_t>(p_custom_config));
  }

  switch (htp_options_->pd_session()) {
    case QnnExecuTorchHtpPdSession::kHtpSignedPd:
      p_custom_config = htp_device_custom_config_->AllocDeviceCustomConfig();
      p_custom_config->option = QNN_HTP_DEVICE_CONFIG_OPTION_SIGNEDPD;
      p_custom_config->useSignedProcessDomain.useSignedProcessDomain = true;
      p_custom_config->useSignedProcessDomain.deviceId = 0;
      device_custom_config.push_back(
          static_cast<QnnDevice_CustomConfig_t>(p_custom_config));
      break;
    case QnnExecuTorchHtpPdSession::kHtpUnsignedPd:
    default:
      break;
  }

  const std::vector<QnnDevice_PlatformInfo_t*>& device_platform_info =
      htp_device_platform_info_config_->CreateDevicePlatformInfo(
          qcom_target_soc_info_);

  uint32_t num_custom_configs =
      device_platform_info.size() + device_custom_config.size();
  device_config_.resize(num_custom_configs);
  // +1 for null terminated
  config.reserve(num_custom_configs + 1);

  for (std::size_t i = 0; i < device_custom_config.size(); ++i) {
    device_config_[i].option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
    device_config_[i].customConfig = device_custom_config[i];
    config.push_back(&device_config_[i]);
  }

  if (!device_platform_info.empty()) {
    // Below codes use `Device_config_[device_custom_config.size()]` which imply
    // the length of platform info can only be 1.
    ET_CHECK_OR_RETURN_ERROR(
        device_platform_info.size() == 1u,
        Internal,
        "Error! Device platform info size != 1, got %zu",
        device_platform_info.size());
    device_config_[device_custom_config.size()].option =
        QNN_DEVICE_CONFIG_OPTION_PLATFORM_INFO;
    device_config_[device_custom_config.size()].hardwareInfo =
        device_platform_info.back();
    config.push_back(&device_config_[device_custom_config.size()]);
  }

  // null terminated
  config.push_back(nullptr);

  return Error::Ok;
}

void HtpDevice::PerformanceVote() {
  if (IsPerfModeEnabled()) {
    htp_perf_infra_->setPowerConfig(
        powerconfig_client_id_, perf_power_configs_ptr_.data());
  }
};

void HtpDevice::ReleasePerformanceVote() {
  if (IsPerfModeEnabled()) {
    htp_perf_infra_->setPowerConfig(
        powerconfig_client_id_, down_vote_power_configs_ptr_.data());
  }
};

Error HtpDevice::AfterCreateDevice() {
  if (IsPerfModeEnabled()) {
    const QnnInterface& qnn_interface = implementation_.GetQnnInterface();
    Qnn_ErrorHandle_t error = QNN_SUCCESS;

    // Get htp_perf_infra
    htp_perf_infra_ = &owned_htp_perf_infra_;
    if (GetPerfInfra(qnn_interface, htp_perf_infra_) != Error::Ok) {
      return Error::Internal;
    }

    // Get power client id
    error = htp_perf_infra_->createPowerConfigId(
        /*device_id=*/0, /*core_id=*/0, &powerconfig_client_id_);

    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "HTP backend unable to create "
          "power config. Error %d",
          QNN_GET_ERROR_CODE(error));
      return Error::Internal;
    }

    // Set vector of PowerConfigs and map it to a vector of pointers.
    perf_power_configs_ = SetVotePowerConfig(
        powerconfig_client_id_,
        htp_options_->performance_mode(),
        PerformanceModeVoteType::kUpVote);
    perf_power_configs_ptr_ = ObtainNullTermPtrVector(perf_power_configs_);

    down_vote_power_configs_ = SetVotePowerConfig(
        powerconfig_client_id_,
        QnnExecuTorchHtpPerformanceMode::kHtpDefault,
        PerformanceModeVoteType::kDownVote);
    down_vote_power_configs_ptr_ =
        ObtainNullTermPtrVector(down_vote_power_configs_);

    // vote immediately
    PerformanceVote();

    // Set Rpc polling mode
    rpc_power_configs_ =
        SetRpcPollingPowerConfig(htp_options_->performance_mode());
    rpc_power_configs_ptr_ = ObtainNullTermPtrVector(rpc_power_configs_);

    htp_perf_infra_->setPowerConfig(
        powerconfig_client_id_, rpc_power_configs_ptr_.data());
  }

  return Error::Ok;
}

} // namespace qnn
} // namespace executor
} // namespace torch
