/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/backends/lpai/LpaiContext.h>

namespace executorch {
namespace backends {
namespace qnn {

using executorch::runtime::Error;

LpaiContext::LpaiContext(
    QnnImplementation* implementation,
    QnnBackend* backend,
    QnnDevice* device,
    QnnBackendCache* cache,
    QnnDlcManager* qnn_dlc_manager)
    : QnnContext(implementation, backend, device, cache, qnn_dlc_manager) {
  lpai_context_custom_config_ = std::make_unique<LpaiContextCustomConfig>();
}

Error LpaiContext::MakeConfig(std::vector<const QnnContext_Config_t*>& config) {
  const std::vector<QnnContext_CustomConfig_t>& context_custom_config =
      lpai_context_custom_config_->CreateContextCustomConfig();

  uint32_t num_custom_configs = context_custom_config.size();
  context_config_.resize(num_custom_configs);
  // +1 for null terminated
  config.reserve(num_custom_configs + 1);

  for (std::size_t i = 0; i < num_custom_configs; ++i) {
    context_config_[i].option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
    context_config_[i].customConfig = context_custom_config[i];
    config.push_back(&context_config_[i]);
  }

#ifdef __hexagon__
  QnnContext_Config_t adsp_context_config;
  adsp_context_config.option = QNN_CONTEXT_CONFIG_PERSISTENT_BINARY;
  adsp_context_config.isPersistentBinary = 1;
  context_config_.push_back(adsp_context_config);
  config.push_back(&context_config_.back());
#endif

  config.push_back(nullptr);
  return Error::Ok;
}

} // namespace qnn
} // namespace backends
} // namespace executorch
