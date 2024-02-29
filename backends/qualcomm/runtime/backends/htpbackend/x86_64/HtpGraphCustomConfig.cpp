/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/backends/htpbackend/HtpGraphCustomConfig.h>
namespace torch {
namespace executor {
namespace qnn {
std::vector<QnnGraph_CustomConfig_t>
HtpGraphCustomConfig::CreateGraphCustomConfig(
    const HtpInfo& qcom_target_soc_info) {
  std::vector<QnnGraph_CustomConfig_t> ret;
  QnnHtpGraph_CustomConfig_t* p_custom_config = nullptr;

  if (!htp_options_.use_conv_hmx) {
    p_custom_config = AllocGraphCustomConfig();
    p_custom_config->option =
        QNN_HTP_GRAPH_CONFIG_OPTION_SHORT_DEPTH_CONV_ON_HMX_OFF;
    p_custom_config->shortDepthConvOnHmxOff = true;
    ret.push_back(static_cast<QnnGraph_CustomConfig_t>(p_custom_config));
  }

  if (!htp_options_.use_fold_relu) {
    p_custom_config = AllocGraphCustomConfig();
    p_custom_config->option =
        QNN_HTP_GRAPH_CONFIG_OPTION_FOLD_RELU_ACTIVATION_INTO_CONV_OFF;
    p_custom_config->foldReluActivationIntoConvOff = true;
    ret.push_back(static_cast<QnnGraph_CustomConfig_t>(p_custom_config));
  }

  switch (htp_options_.precision) {
    case kHtpFp16:
      p_custom_config = AllocGraphCustomConfig();
      p_custom_config->option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
      p_custom_config->precision = QNN_PRECISION_FLOAT16;
      ret.push_back(static_cast<QnnGraph_CustomConfig_t>(p_custom_config));
      break;
    case kHtpQuantized:
    default:
      break;
  }

  QNN_EXECUTORCH_LOG(
      kLogLevelInfo, "[Qnn ExecuTorch] Running level=3 optimization.");

  p_custom_config = AllocGraphCustomConfig();
  p_custom_config->option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
  p_custom_config->optimizationOption.type =
      QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
  p_custom_config->optimizationOption.floatValue = 3;
  ret.push_back(static_cast<QnnGraph_CustomConfig_t>(p_custom_config));

  p_custom_config = AllocGraphCustomConfig();
  p_custom_config->option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
  p_custom_config->vtcmSizeInMB = qcom_target_soc_info.m_vtcmSizeinMB;
  ret.push_back(static_cast<QnnGraph_CustomConfig_t>(p_custom_config));

  return ret;
}
} // namespace qnn
} // namespace executor
} // namespace torch
