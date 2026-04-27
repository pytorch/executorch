/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/QnnBackendOptions.h>
#include <executorch/backends/qualcomm/runtime/backends/lpai/LpaiGraph.h>

namespace executorch {
namespace backends {
namespace qnn {

using executorch::runtime::Error;
Error LpaiGraph::AfterRetrieveGraph(const std::string& graph_name) {
  std::vector<QnnGraph_CustomConfig_t> graph_custom_config;
  QnnLpaiGraph_CustomConfig_t* p_custom_config = nullptr;

  // perf config
  p_custom_config = AllocGraphCustomConfig();
  auto p_perf_cfg = AllocPerfCfg();
  p_custom_config->option = QNN_LPAI_GRAPH_SET_CFG_PERF_CFG;
  p_perf_cfg->fps = get_option(lpai_options_->fps(), QNN_RUNTIME_LPAI_FPS);
  p_perf_cfg->ftrtRatio =
      get_option(lpai_options_->ftrt_ratio(), QNN_RUNTIME_LPAI_FTRT_RATIO);
  p_perf_cfg->clientType =
      static_cast<QnnLpaiGraph_ClientPerfType_t>(get_option(
          lpai_options_->client_perf_type(),
          QNN_RUNTIME_LPAI_CLIENT_PERF_TYPE));
  p_custom_config->config = p_perf_cfg;
  graph_custom_config.push_back(p_custom_config);
  // core affinity
  p_custom_config = AllocGraphCustomConfig();
  auto p_core_affinity = AllocCoreAffinity();
  p_custom_config->option = QNN_LPAI_GRAPH_SET_CFG_CORE_AFFINITY;
  p_core_affinity->affinity = static_cast<QnnLpaiGraph_CoreAffinityType_t>(
      get_option(lpai_options_->affinity(), QNN_RUNTIME_LPAI_AFFINITY));
  p_core_affinity->coreSelection = get_option(
      lpai_options_->core_selection(), QNN_RUNTIME_LPAI_CORE_SELECTION);
  p_custom_config->config = p_core_affinity;
  graph_custom_config.push_back(p_custom_config);

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

  // LPAI specific configs can only be set after graph create
  const QnnInterface& qnn_interface = implementation_->GetQnnInterface();
  Qnn_ErrorHandle_t error =
      qnn_interface.qnn_graph_set_config(handle_[graph_name], config.data());
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "qnn_graph_set_config failed. Error  %d", QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }

  error = GraphFinalize(graph_name);
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Failed to finalize Qnn Graph with error: %d",
        QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }
  return Error::Ok;
};

Error LpaiGraph::AfterCreateGraph(const std::string& graph_name) {
  std::vector<QnnGraph_CustomConfig_t> graph_custom_config;
#if (QNN_API_VERSION_MAJOR >= 2 && QNN_API_VERSION_MINOR >= 29)
  QnnLpaiGraph_CustomConfig_t* p_custom_config = nullptr;

  p_custom_config = AllocGraphCustomConfig();
  auto p_core_prepare = AllocPrepare();

  p_custom_config->option = QNN_LPAI_GRAPH_SET_CFG_PREPARE;
  p_core_prepare->enableCoreSelection =
      const_cast<char*>(default_core_selection_);
  p_custom_config->config = p_core_prepare;
  graph_custom_config.push_back(
      static_cast<QnnBackend_CustomConfig_t>(p_custom_config));
#endif

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

  // LPAI specific configs can only be set after graph create
  const QnnInterface& qnn_interface = implementation_->GetQnnInterface();
  Qnn_ErrorHandle_t error =
      qnn_interface.qnn_graph_set_config(handle_[graph_name], config.data());
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "qnn_graph_set_config failed. Error  %d", QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }

  return Error::Ok;
};
} // namespace qnn
} // namespace backends
} // namespace executorch
