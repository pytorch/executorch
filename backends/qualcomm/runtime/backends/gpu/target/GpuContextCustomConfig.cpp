/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/backends/gpu/GpuContextCustomConfig.h>

namespace executorch {
namespace backends {
namespace qnn {

std::vector<QnnContext_CustomConfig_t>
GpuContextCustomConfig::CreateContextCustomConfig() {
  std::vector<QnnContext_CustomConfig_t> ret;
  QnnGpuContext_CustomConfig_t* p_custom_config = nullptr;

  p_custom_config = AllocContextCustomConfig();
  p_custom_config->option = QNN_GPU_CONTEXT_CONFIG_OPTION_PERF_HINT;
  p_custom_config->perfHint =
      static_cast<QnnGpuContext_PerfHint_t>(gpu_options_->performance_mode());
  ret.push_back(static_cast<QnnContext_CustomConfig_t>(p_custom_config));
  return ret;
}

} // namespace qnn
} // namespace backends
} // namespace executorch
