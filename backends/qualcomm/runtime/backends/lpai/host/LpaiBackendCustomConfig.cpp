/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnSdkCompatibility.h>
#include <executorch/backends/qualcomm/runtime/backends/lpai/LpaiBackendCustomConfig.h>

namespace executorch {
namespace backends {
namespace qnn {

std::vector<QnnBackend_CustomConfig_t>
LpaiBackendCustomConfig::CreateBackendCustomConfig() {
  std::vector<QnnBackend_CustomConfig_t> ret =
      CreateBackendCustomConfigCommon();

  const auto& custom_prepare_config =
      lpai_options_->custom_prepare_config_file();
  if (custom_prepare_config && !custom_prepare_config->str().empty()) {
#if QNN_EXECUTORCH_QNN_API_VERSION_AT_LEAST(2, 40)
    auto* p_custom_config = AllocBackendCustomConfig();
    p_custom_config->option = QNN_LPAI_BACKEND_CUSTOM_CFG_PREPARE_CFG;
    p_custom_config->config = const_cast<char*>(custom_prepare_config->c_str());
    ret.push_back(static_cast<QnnBackend_CustomConfig_t>(p_custom_config));
#else
    QNN_EXECUTORCH_LOG_WARN(
        "lpai_custom_prepare_config_file (\"%s\") was set but requires QNN API "
        ">= 2.40 (QAIRT SDK 2.51+); ignoring on this SDK.",
        custom_prepare_config->c_str());
#endif
  }
  return ret;
}

} // namespace qnn
} // namespace backends
} // namespace executorch
