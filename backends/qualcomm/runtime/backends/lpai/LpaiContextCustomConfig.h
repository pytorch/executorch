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

#include "LPAI/QnnLpaiContext.h"

namespace executorch {
namespace backends {
namespace qnn {

using namespace qnn_delegate;

class LpaiContextCustomConfig {
 public:
  explicit LpaiContextCustomConfig() {}

  std::vector<QnnContext_CustomConfig_t> CreateContextCustomConfig();

 private:
  QnnLpaiContext_CustomConfig_t* AllocContextCustomConfig() {
    lpai_context_config_.emplace_back(
        std::make_unique<QnnLpaiContext_CustomConfig_t>());
    lpai_context_config_.back()->option = QNN_LPAI_CONTEXT_SET_CFG_UNDEFINED;
    return lpai_context_config_.back().get();
  }
  std::vector<std::unique_ptr<QnnLpaiContext_CustomConfig_t>>
      lpai_context_config_;
};

} // namespace qnn
} // namespace backends
} // namespace executorch
