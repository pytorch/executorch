/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/backends/htpbackend/HtpContext.h>

#include "HTP/QnnHtpCommon.h"
#include "Saver/QnnSaverCommon.h"

namespace torch {
namespace executor {
namespace qnn {

Error HtpContext::MakeConfig(std::vector<const QnnContext_Config_t*>& config) {
  const std::vector<QnnContext_CustomConfig_t>& context_custom_config =
      htp_context_custom_config_->CreateContextCustomConfig();

  uint32_t num_custom_configs = context_custom_config.size();
  context_config_.resize(num_custom_configs);
  // +1 for null terminated
  config.reserve(num_custom_configs + 1);

  for (std::size_t i = 0; i < num_custom_configs; ++i) {
    context_config_[i].option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
    context_config_[i].customConfig = context_custom_config[i];
    config.push_back(&context_config_[i]);
  }

  config.push_back(nullptr);
  return Error::Ok;
}

Error HtpContext::AfterConfigure() {
  // update sf_handle with first context handle encounterded as group handle
  // TODO: should handle the thread safety if needed
  if (sf_handle_ == 0x0) {
    sf_handle_ = GetHandle();
  }
  return Error::Ok;
}

} // namespace qnn
} // namespace executor
} // namespace torch
