/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/qualcomm/runtime/backends/gpu/GpuContextCustomConfig.h>

namespace executorch {
namespace backends {
namespace qnn {

class QnnDlcManager;
class GpuContext : public QnnContext {
 public:
  GpuContext(
      const QnnImplementation& implementation,
      QnnBackend* backend,
      QnnDevice* device,
      QnnBackendCache* cache,
      QnnDlcManager* qnn_dlc_manager,
      const QnnExecuTorchGpuBackendOptions* gpu_options);

 protected:
  executorch::runtime::Error MakeConfig(
      std::vector<const QnnContext_Config_t*>& config) override;

 private:
  std::vector<QnnContext_Config_t> context_config_;
  std::unique_ptr<GpuContextCustomConfig> gpu_context_custom_config_;
};

} // namespace qnn
} // namespace backends
} // namespace executorch
