/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/backends/htp/HtpContext.h>
#include <executorch/backends/qualcomm/runtime/backends/htp/HtpContextCustomConfig.h>

namespace executorch {
namespace backends {
namespace qnn {

std::vector<QnnContext_CustomConfig_t>
HtpContextCustomConfig::CreateContextCustomConfig() {
  std::vector<QnnContext_CustomConfig_t> ret;
  QnnHtpContext_CustomConfig_t* p_custom_config = nullptr;
  const HtpContext* htp_ctx = static_cast<const HtpContext*>(context_);

  // TODO: Verify heap profile works with kProfileBasic once enabled.
  if (profile_level_ != QnnExecuTorchProfileLevel::kProfileOff) {
    QnnHtpContext_CustomConfig_t* p_custom_config_profile = nullptr;
    p_custom_config_profile = AllocContextCustomConfig();
    p_custom_config_profile->option =
        QNN_HTP_CONTEXT_CONFIG_OPTION_DSP_MEMORY_PROFILING_ENABLED;
    p_custom_config_profile->dspMemoryProfilingEnabled = true;
    ret.push_back(
        static_cast<QnnContext_CustomConfig_t>(p_custom_config_profile));
  }

  if (htp_options_->use_multi_contexts() &&
      htp_options_->max_sf_buf_size() != 0) {
    p_custom_config = AllocContextCustomConfig();
    p_custom_config->option =
        QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS;
    QnnHtpContext_GroupRegistration_t group_info;
    group_info.firstGroupHandle = htp_ctx->GetSpillFillHandle();
    group_info.maxSpillFillBuffer = htp_options_->max_sf_buf_size();
    p_custom_config->groupRegistration = group_info;
    ret.push_back(static_cast<QnnContext_CustomConfig_t>(p_custom_config));
  }

  return ret;
}

} // namespace qnn
} // namespace backends
} // namespace executorch
