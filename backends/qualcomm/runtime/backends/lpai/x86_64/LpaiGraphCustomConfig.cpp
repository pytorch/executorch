/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/backends/lpai/LpaiGraphCustomConfig.h>

namespace executorch {
namespace backends {
namespace qnn {

std::vector<QnnGraph_CustomConfig_t>
LpaiGraphCustomConfig::CreateGraphCustomConfig(const std::string& graph_name) {
  std::vector<QnnGraph_CustomConfig_t> configs;
  QnnLpaiGraph_CustomConfig_t* p_custom_config = nullptr;

  p_custom_config = AllocGraphCustomConfig();
  auto p_core_prepare = AllocPrepare();
  static char core_selection = lpai_options_->core_selection() + '0';
  p_custom_config->option = QNN_LPAI_GRAPH_SET_CFG_PREPARE;
  p_core_prepare->enableCoreSelection = &core_selection;
  p_custom_config->config = p_core_prepare;
  configs.push_back(static_cast<QnnBackend_CustomConfig_t>(p_custom_config));
  return configs;
}

} // namespace qnn
} // namespace backends
} // namespace executorch
