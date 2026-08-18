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

  if (htp_options_->use_weight_sharing()) {
    p_custom_config = AllocContextCustomConfig();
    p_custom_config->option =
        QNN_HTP_CONTEXT_CONFIG_OPTION_WEIGHT_SHARING_ENABLED;
    p_custom_config->weightSharingEnabled = true;
    ret.push_back(static_cast<QnnContext_CustomConfig_t>(p_custom_config));
  }

#if (QNN_HTP_API_VERSION_MAJOR >= 5 && QNN_HTP_API_VERSION_MINOR >= 49)
  if (htp_options_->use_graph_splitting()) {
    p_custom_config = AllocContextCustomConfig();
    p_custom_config->option =
        QNN_HTP_CONTEXT_CONFIG_OPTION_GRAPH_SPLITTING_CONFIGS;
    QnnHtpContext_GraphSplit_t graph_split_info;
    graph_split_info.graphSplittingEnabled = true;
    p_custom_config->graphSplittingConfigs = graph_split_info;
    ret.push_back(static_cast<QnnContext_CustomConfig_t>(p_custom_config));
  }
#else
  if (htp_options_->use_graph_splitting()) {
    QNN_EXECUTORCH_LOG_WARN(
        "use_graph_splitting is enabled but the QNN SDK used for this build "
        "(API %d.%d) does not support graph splitting, which requires HTP API "
        "5.49 or newer. The option will be ignored.",
        QNN_API_VERSION_MAJOR,
        QNN_API_VERSION_MINOR);
  }
#endif

  return ret;
}

} // namespace qnn
} // namespace backends
} // namespace executorch
