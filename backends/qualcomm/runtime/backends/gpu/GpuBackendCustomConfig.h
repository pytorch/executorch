/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/qualcomm/qc_compiler_spec_generated.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnBackendCommon.h>

#include <memory>
#include <vector>

#include "GPU/QnnGpuBackend.h"

namespace executorch {
namespace backends {
namespace qnn {

using namespace qnn_delegate;

class GpuBackendCustomConfig {
 public:
  explicit GpuBackendCustomConfig(
      const QnnExecuTorchGpuBackendOptions* gpu_options);

  std::vector<QnnBackend_CustomConfig_t> CreateBackendCustomConfig();

 private:
  QnnGpuBackend_CustomConfig_t* AllocBackendCustomConfig();
  std::vector<std::unique_ptr<QnnGpuBackend_CustomConfig_t>>
      gpu_backend_config_;
  const QnnExecuTorchGpuBackendOptions* gpu_options_;
};

} // namespace qnn
} // namespace backends
} // namespace executorch
