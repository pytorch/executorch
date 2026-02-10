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
#include <unordered_map>
#include <vector>

#include "LPAI/QnnLpaiBackend.h"

namespace executorch {
namespace backends {
namespace qnn {

using namespace qnn_delegate;

class LpaiBackendCustomConfig {
 public:
  explicit LpaiBackendCustomConfig(const SocInfo* soc_info);

  std::vector<QnnBackend_CustomConfig_t> CreateBackendCustomConfig();

 private:
  QnnLpaiBackend_CustomConfig_t* AllocBackendCustomConfig();
  std::vector<std::unique_ptr<QnnLpaiBackend_CustomConfig_t>>
      lpai_backend_config_;
  const SocInfo* soc_info_;

  std::vector<std::unique_ptr<QnnLpaiBackend_CustomConfigHwInfo_t>>
      lpai_hw_info_;
  QnnLpaiBackend_CustomConfigHwInfo_t* AllocHwInfo();
};

} // namespace qnn
} // namespace backends
} // namespace executorch
