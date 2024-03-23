/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/runtime/backends/QnnGraphCommon.h>
#include <executorch/backends/qualcomm/runtime/backends/htpbackend/HtpGraphCustomConfig.h>

#include <memory>

#include "HTP/QnnHtpGraph.h"
namespace torch {
namespace executor {
namespace qnn {
class HtpGraph : public QnnGraph {
 public:
  HtpGraph(
      const QnnImplementation& implementation,
      QnnBackend* backend,
      QnnContext* context,
      const QnnExecuTorchProfileLevel& profile_level,
      const std::string& graph_name,
      const SocInfo* soc_info,
      const QnnExecuTorchHtpBackendOptions* htp_options)
      : QnnGraph(implementation, backend, context, profile_level, graph_name),
        qcom_target_soc_info_(soc_info),
        htp_options_(htp_options) {
    htp_graph_custom_config_ =
        std::make_unique<HtpGraphCustomConfig>(htp_options, context);
  }
  ~HtpGraph() {}

 protected:
  Error MakeConfig(std::vector<const QnnGraph_Config_t*>& config) override;

 private:
  std::vector<QnnGraph_Config_t> graph_config_;
  std::unique_ptr<HtpGraphCustomConfig> htp_graph_custom_config_;
  const SocInfo* qcom_target_soc_info_;
  [[maybe_unused]] const QnnExecuTorchHtpBackendOptions* htp_options_;
};
} // namespace qnn
} // namespace executor
} // namespace torch
