/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/backends/lpai/LpaiContextCustomConfig.h>

namespace executorch {
namespace backends {
namespace qnn {

std::vector<QnnContext_CustomConfig_t>
LpaiContextCustomConfig::CreateContextCustomConfig() {
  std::vector<QnnContext_CustomConfig_t> ret;
  QnnLpaiContext_CustomConfig_t* p_custom_config = nullptr;

  // TODO: support graph based execution in island mode
  p_custom_config = AllocContextCustomConfig();
  p_custom_config->option = QNN_LPAI_CONTEXT_SET_CFG_ENABLE_ISLAND;
  p_custom_config->config = nullptr;
  ret.push_back(static_cast<QnnContext_CustomConfig_t>(p_custom_config));
  return ret;
}

} // namespace qnn
} // namespace backends
} // namespace executorch
