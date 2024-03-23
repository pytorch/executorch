/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/backends/htpbackend/HtpGraph.h>
namespace torch {
namespace executor {
namespace qnn {

Error HtpGraph::MakeConfig(std::vector<const QnnGraph_Config_t*>& config) {
  const std::vector<QnnGraph_CustomConfig_t>& graph_custom_config =
      htp_graph_custom_config_->CreateGraphCustomConfig(qcom_target_soc_info_);

  uint32_t num_custom_configs = graph_custom_config.size();
  graph_config_.resize(num_custom_configs);
  // +1 for null terminated
  config.reserve(num_custom_configs + 1);

  for (std::size_t i = 0; i < num_custom_configs; ++i) {
    graph_config_[i].option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    graph_config_[i].customConfig = graph_custom_config[i];
    config.push_back(&graph_config_[i]);
  }

  config.push_back(nullptr);

  return Error::Ok;
}
} // namespace qnn
} // namespace executor
} // namespace torch
