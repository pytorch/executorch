/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/backends/lpai/LpaiGraph.h>

namespace executorch {
namespace backends {
namespace qnn {

std::vector<QnnGraph_CustomConfig_t>
LpaiGraphCustomConfig::CreateGraphCustomConfig(const std::string& graph_name) {
  std::vector<QnnGraph_CustomConfig_t> configs;
  QnnLpaiGraph_CustomConfig_t* p_custom_config = nullptr;

#ifdef __hexagon__
  uint32_t scratch_size = 0;
  uint32_t persistent_size = 0;
  QnnLpaiGraph_CustomProperty_t custom_props[2];
  custom_props[0].option = QNN_LPAI_GRAPH_GET_PROP_SCRATCH_MEM_SIZE;
  custom_props[0].property = &scratch_size;
  custom_props[1].option = QNN_LPAI_GRAPH_GET_PROP_PERSISTENT_MEM_SIZE;
  custom_props[1].property = &persistent_size;

  QnnGraph_Property_t graph_props[2];
  graph_props[0].option = QNN_GRAPH_PROPERTY_OPTION_CUSTOM;
  graph_props[0].customProperty = &custom_props[0];
  graph_props[1].option = QNN_GRAPH_PROPERTY_OPTION_CUSTOM;
  graph_props[1].customProperty = &custom_props[1];
  QnnGraph_Property_t* graph_prop_ptrs[3] = {0};
  graph_prop_ptrs[0] = &graph_props[0];
  graph_prop_ptrs[1] = &graph_props[1];

  const QnnInterface& qnn_interface =
      graph_->implementation_->GetQnnInterface();
  Qnn_ErrorHandle_t error = qnn_interface.qnn_graph_get_property(
      graph_->handle_[graph_name], graph_prop_ptrs);

  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "failed to get graph property: %d", QNN_GET_ERROR_CODE(error));
    return {};
  }

  scratch_buf_.resize(scratch_size);
  p_custom_config = AllocGraphCustomConfig();
  p_custom_config->option = QNN_LPAI_GRAPH_SET_CFG_SCRATCH_MEM;
  auto p_scratch_config = AllocMem();
  p_scratch_config->memType = QNN_LPAI_MEM_TYPE_DDR;
  p_scratch_config->size = scratch_size;
  p_scratch_config->addr = scratch_buf_.data();
  p_custom_config->config = p_scratch_config;
  configs.push_back(p_custom_config);

  persistent_buf_.resize(persistent_size);
  p_custom_config = AllocGraphCustomConfig();
  p_custom_config->option = QNN_LPAI_GRAPH_SET_CFG_PERSISTENT_MEM_DEFAULT;
  auto p_persistent_config = AllocMem();
  p_persistent_config->memType = QNN_LPAI_MEM_TYPE_DDR;
  p_persistent_config->size = persistent_size;
  p_persistent_config->addr = persistent_buf_.data();
  p_custom_config->config = p_persistent_config;
  configs.push_back(p_custom_config);
  // TODO: figure out how to add perf control (internal enum required)
  //       e.g. QNN_LPAI_GRAPH_SET_ENPU_CLOCK
#endif
  // perf config
  p_custom_config = AllocGraphCustomConfig();
  auto p_perf_cfg = AllocPerfCfg();
  p_custom_config->option = QNN_LPAI_GRAPH_SET_CFG_PERF_CFG;
  p_perf_cfg->fps = lpai_options_->fps();
  p_perf_cfg->ftrtRatio = lpai_options_->ftrt_ratio();
  p_perf_cfg->clientType = static_cast<QnnLpaiGraph_ClientPerfType_t>(
      lpai_options_->client_perf_type());
  p_custom_config->config = p_perf_cfg;
  configs.push_back(p_custom_config);
  // core affinity
  p_custom_config = AllocGraphCustomConfig();
  auto p_core_affinity = AllocCoreAffinity();
  p_custom_config->option = QNN_LPAI_GRAPH_SET_CFG_CORE_AFFINITY;
  p_core_affinity->affinity =
      static_cast<QnnLpaiGraph_CoreAffinityType_t>(lpai_options_->affinity());
  p_core_affinity->coreSelection = lpai_options_->core_selection();
  p_custom_config->config = p_core_affinity;
  configs.push_back(p_custom_config);
  return configs;
}

} // namespace qnn
} // namespace backends
} // namespace executorch
