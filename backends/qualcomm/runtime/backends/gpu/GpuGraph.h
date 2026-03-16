/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/qualcomm/runtime/backends/gpu/GpuGraphCustomConfig.h>

namespace executorch {
namespace backends {
namespace qnn {

class GpuGraph : public QnnGraph {
 public:
  GpuGraph(
      QnnImplementation* implementation,
      QnnBackend* backend,
      QnnContext* context,
      const QnnExecuTorchProfileLevel& profile_level,
      const QnnExecuTorchGpuBackendOptions* gpu_options);

 protected:
  executorch::runtime::Error MakeConfig(
      std::vector<const QnnGraph_Config_t*>& config) override;

 private:
  std::vector<QnnGraph_Config_t> graph_config_;
  std::unique_ptr<GpuGraphCustomConfig> gpu_graph_custom_config_;
};

} // namespace qnn
} // namespace backends
} // namespace executorch
