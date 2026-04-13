/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/qualcomm/runtime/backends/gpu/GpuBackendCustomConfig.h>

namespace executorch {
namespace backends {
namespace qnn {

class GpuBackend : public QnnBackend {
 public:
  GpuBackend(
      QnnImplementation* implementation,
      QnnLogger* logger,
      const QnnExecuTorchGpuBackendOptions* gpu_options);

  Qnn_Version_t GetExpectedBackendVersion() const override;

  bool IsProfileEventTypeParentOfNodeTime(
      QnnProfile_EventType_t event_type) override;

 protected:
  executorch::runtime::Error MakeConfig(
      std::vector<const QnnBackend_Config_t*>& config) override;

 private:
  std::vector<QnnBackend_Config_t> backend_config_;
  std::unique_ptr<GpuBackendCustomConfig> gpu_backend_custom_config_;
};

} // namespace qnn
} // namespace backends
} // namespace executorch
