/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/qualcomm/qc_compiler_spec_generated.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnGraphCommon.h>

#include <memory>
#include <vector>

#include "GPU/QnnGpuGraph.h"

namespace executorch {
namespace backends {
namespace qnn {

using namespace qnn_delegate;

class GpuGraphCustomConfig {
 public:
  explicit GpuGraphCustomConfig(
      const QnnExecuTorchGpuBackendOptions* gpu_options);

  std::vector<QnnGraph_CustomConfig_t> CreateGraphCustomConfig();

 private:
  QnnGpuGraph_CustomConfig_t* AllocGraphCustomConfig();
  std::vector<std::unique_ptr<QnnGpuGraph_CustomConfig_t>> gpu_graph_config_;
  const QnnExecuTorchGpuBackendOptions* gpu_options_;
};

} // namespace qnn
} // namespace backends
} // namespace executorch
