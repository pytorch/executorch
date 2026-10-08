/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/qualcomm/qc_compiler_spec_generated.h>
#include <executorch/backends/qualcomm/runtime/backends/lpai/LpaiBackendCustomConfig.h>

#include <string>

namespace executorch {
namespace backends {
namespace qnn {

class LpaiBackend : public QnnBackend {
 public:
  LpaiBackend(
      QnnImplementation* implementation,
      QnnLogger* logger,
      const SocInfo* soc_info,
      const QnnExecuTorchLpaiBackendOptions* lpai_options);

  Qnn_Version_t GetExpectedBackendVersion() const override;

  bool IsProfileEventTypeParentOfNodeTime(
      QnnProfile_EventType_t event_type) override;

  // Stored as a copy, since this instance can outlive the QnnExecuTorchOptions
  // buffer lpai_options points into.
  const std::string& custom_prepare_config_file() const {
    return custom_prepare_config_file_;
  }

 protected:
  executorch::runtime::Error MakeConfig(
      std::vector<const QnnBackend_Config_t*>& config) override;

 private:
  std::vector<QnnBackend_Config_t> backend_config_;
  std::unique_ptr<LpaiBackendCustomConfig> lpai_backend_custom_config_;
  std::string custom_prepare_config_file_;
};

} // namespace qnn
} // namespace backends
} // namespace executorch
