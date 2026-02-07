/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/qualcomm/runtime/backends/lpai/LpaiGraphCustomConfig.h>

namespace executorch {
namespace backends {
namespace qnn {

class LpaiGraph : public QnnGraph {
 public:
  LpaiGraph(
      QnnImplementation* implementation,
      QnnBackend* backend,
      QnnContext* context,
      const QnnExecuTorchProfileLevel& profile_level,
      const QnnExecuTorchLpaiBackendOptions* lpai_options)
      : QnnGraph(implementation, backend, context, profile_level) {
    lpai_graph_custom_config_ =
        std::make_unique<LpaiGraphCustomConfig>(lpai_options, this);
  };

  executorch::runtime::Error Configure(const std::string& graph_name) override {
    Error configure_status = QnnGraph::Configure(graph_name);
    if (configure_status != Error::Ok) {
      return configure_status;
    }
    const std::vector<QnnGraph_CustomConfig_t>& graph_custom_config =
        lpai_graph_custom_config_->CreateGraphCustomConfig(graph_name);

    std::vector<const QnnGraph_Config_t*> config;
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

    // LPAI specific > configs can only be set after graph create
    const QnnInterface& qnn_interface = implementation_->GetQnnInterface();
    Qnn_ErrorHandle_t error =
        qnn_interface.qnn_graph_set_config(handle_[graph_name], config.data());
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "qnn_graph_set_config failed. Error  %d", QNN_GET_ERROR_CODE(error));
      return Error::Internal;
    }

    // platform specific behavior
    return AfterConfigure(graph_name);
  }

  friend LpaiGraphCustomConfig;

 protected:
  executorch::runtime::Error MakeConfig(
      std::vector<const QnnGraph_Config_t*>& config) override {
    return {};
  }

 private:
  executorch::runtime::Error AfterConfigure(const std::string& graph_name);
  std::vector<QnnGraph_Config_t> graph_config_;
  std::unique_ptr<LpaiGraphCustomConfig> lpai_graph_custom_config_;
};

} // namespace qnn
} // namespace backends
} // namespace executorch
