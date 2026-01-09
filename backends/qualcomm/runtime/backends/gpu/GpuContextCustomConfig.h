/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/qualcomm/qc_compiler_spec_generated.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnContextCommon.h>

#include <memory>
#include <vector>

#include "GPU/QnnGpuContext.h"

namespace executorch {
namespace backends {
namespace qnn {

using namespace qnn_delegate;

class GpuContextCustomConfig {
 public:
  explicit GpuContextCustomConfig(
      const QnnExecuTorchGpuBackendOptions* gpu_options)
      : gpu_options_(gpu_options) {}

  std::vector<QnnContext_CustomConfig_t> CreateContextCustomConfig();

 private:
  QnnGpuContext_CustomConfig_t* AllocContextCustomConfig() {
    gpu_context_config_.emplace_back(
        std::make_unique<QnnGpuContext_CustomConfig_t>());
    gpu_context_config_.back()->option =
        QNN_GPU_CONTEXT_CONFIG_OPTION_UNDEFINED;
    return gpu_context_config_.back().get();
  }
  std::vector<std::unique_ptr<QnnGpuContext_CustomConfig_t>>
      gpu_context_config_;
  [[maybe_unused]] const QnnExecuTorchGpuBackendOptions* gpu_options_;
};

} // namespace qnn
} // namespace backends
} // namespace executorch
