/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/qualcomm/qc_compiler_spec_generated.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnGraphCommon.h>

#include <memory>
#include <vector>

#include "LPAI/QnnLpaiGraph.h"
#include "LPAI/QnnLpaiGraphPrepare.h"
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
      : QnnGraph(implementation, backend, context, profile_level),
        lpai_options_(lpai_options){};

 protected:
  executorch::runtime::Error MakeConfig(
      std::vector<const QnnGraph_Config_t*>& config) override {
    return {};
  }
  executorch::runtime::Error AfterRetrieveGraph(
      const std::string& graph_name) override;
  executorch::runtime::Error AfterCreateGraph(
      const std::string& graph_name) override;

 private:
  static constexpr const char* default_core_selection_ = "0,1";
  const QnnExecuTorchLpaiBackendOptions* lpai_options_;
  std::vector<QnnGraph_Config_t> graph_config_;
  std::vector<std::unique_ptr<QnnLpaiGraph_Mem_t>> lpai_mem_;
  std::vector<std::unique_ptr<QnnLpaiGraph_CustomConfig_t>> lpai_graph_config_;
  std::vector<std::unique_ptr<QnnLpaiGraph_PerfCfg_t>> lpai_perf_cfg_;
  std::vector<std::unique_ptr<QnnLpaiGraph_CoreAffinity_t>> lpai_core_affinity_;
  std::vector<std::unique_ptr<QnnLpaiGraph_CustomConfigPrepare_t>>
      lpai_prepare_;

  QnnLpaiGraph_Mem_t* AllocMem() {
    lpai_mem_.emplace_back(std::make_unique<QnnLpaiGraph_Mem_t>());
    lpai_mem_.back()->memType = QNN_LPAI_MEM_TYPE_UNDEFINED;
    lpai_mem_.back()->size = 0;
    lpai_mem_.back()->addr = nullptr;
    return lpai_mem_.back().get();
  }

  QnnLpaiGraph_CustomConfig_t* AllocGraphCustomConfig() {
    lpai_graph_config_.emplace_back(
        std::make_unique<QnnLpaiGraph_CustomConfig_t>());
    lpai_graph_config_.back()->option = QNN_LPAI_GRAPH_SET_CFG_UNDEFINED;
    return lpai_graph_config_.back().get();
  }

  QnnLpaiGraph_PerfCfg_t* AllocPerfCfg() {
    lpai_perf_cfg_.emplace_back(std::make_unique<QnnLpaiGraph_PerfCfg_t>());
    lpai_perf_cfg_.back()->fps = 0;
    lpai_perf_cfg_.back()->ftrtRatio = 0;
    lpai_perf_cfg_.back()->clientType =
        QNN_LPAI_GRAPH_CLIENT_PERF_TYPE_UNDEFINED;
    return lpai_perf_cfg_.back().get();
  }

  QnnLpaiGraph_CoreAffinity_t* AllocCoreAffinity() {
    lpai_core_affinity_.emplace_back(
        std::make_unique<QnnLpaiGraph_CoreAffinity_t>());
    lpai_core_affinity_.back()->affinity =
        QNN_LPAI_GRAPH_CORE_AFFINITY_UNDEFINED;
    lpai_core_affinity_.back()->coreSelection = 0;
    return lpai_core_affinity_.back().get();
  }

  QnnLpaiGraph_CustomConfigPrepare_t* AllocPrepare() {
    lpai_prepare_.emplace_back(
        std::make_unique<QnnLpaiGraph_CustomConfigPrepare_t>());
    lpai_prepare_.back()->enablePerLayer = 0;
#if (QNN_API_VERSION_MAJOR >= 2 && QNN_API_VERSION_MINOR >= 29)
    lpai_prepare_.back()->enableCoreSelection = nullptr;
#endif
    return lpai_prepare_.back().get();
  }
};

} // namespace qnn
} // namespace backends
} // namespace executorch
