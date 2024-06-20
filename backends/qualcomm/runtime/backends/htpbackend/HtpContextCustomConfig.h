/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/qualcomm/runtime/backends/QnnContextCommon.h>
#include <executorch/backends/qualcomm/schema_generated.h>

#include <memory>
#include <vector>

#include "HTP/QnnHtpContext.h"

namespace torch {
namespace executor {
namespace qnn {

using namespace qnn_delegate;

class HtpContextCustomConfig {
 public:
  explicit HtpContextCustomConfig(
      const QnnContext* context,
      const QnnExecuTorchHtpBackendOptions* htp_options)
      : context_(context), htp_options_(htp_options) {}

  std::vector<QnnContext_CustomConfig_t> CreateContextCustomConfig();

 private:
  QnnHtpContext_CustomConfig_t* AllocContextCustomConfig() {
    htp_context_config_.emplace_back(
        std::make_unique<QnnHtpContext_CustomConfig_t>());
    htp_context_config_.back()->option = QNN_HTP_CONTEXT_CONFIG_OPTION_UNKNOWN;
    return htp_context_config_.back().get();
  }

  [[maybe_unused]] const QnnContext* context_;
  std::vector<std::unique_ptr<QnnHtpContext_CustomConfig_t>>
      htp_context_config_;
  [[maybe_unused]] const QnnExecuTorchHtpBackendOptions* htp_options_;
};

} // namespace qnn
} // namespace executor
} // namespace torch
