/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/qualcomm/runtime/backends/lpai/LpaiBackendCustomConfig.h>

namespace executorch {
namespace backends {
namespace qnn {

class LpaiBackend : public QnnBackend {
 public:
  LpaiBackend(
      QnnImplementation* implementation,
      QnnLogger* logger,
      const SocInfo* soc_info);

  Qnn_Version_t GetExpectedBackendVersion() const override;

  bool IsProfileEventTypeParentOfNodeTime(
      QnnProfile_EventType_t event_type) override;

 protected:
  executorch::runtime::Error MakeConfig(
      std::vector<const QnnBackend_Config_t*>& config) override;

 private:
  std::vector<QnnBackend_Config_t> backend_config_;
  std::unique_ptr<LpaiBackendCustomConfig> lpai_backend_custom_config_;
};

} // namespace qnn
} // namespace backends
} // namespace executorch
