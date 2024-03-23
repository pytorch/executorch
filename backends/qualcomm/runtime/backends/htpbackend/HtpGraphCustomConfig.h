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

#include "HTP/QnnHtpGraph.h"
namespace torch {
namespace executor {
namespace qnn {
using namespace qnn_delegate;
class HtpGraphCustomConfig {
 public:
  explicit HtpGraphCustomConfig(
      const QnnExecuTorchHtpBackendOptions* htp_options,
      const QnnContext* context)
      : htp_options_(htp_options), context_(context) {}

  std::vector<QnnGraph_CustomConfig_t> CreateGraphCustomConfig(
      const SocInfo* qcom_target_soc_info);

 private:
  QnnHtpGraph_CustomConfig_t* AllocGraphCustomConfig() {
    htp_graph_config_.emplace_back(
        std::make_unique<QnnHtpGraph_CustomConfig_t>());
    htp_graph_config_.back()->option = QNN_HTP_GRAPH_CONFIG_OPTION_UNKNOWN;
    return htp_graph_config_.back().get();
  }

  [[maybe_unused]] const QnnExecuTorchHtpBackendOptions* htp_options_;
  std::vector<std::unique_ptr<QnnHtpGraph_CustomConfig_t>> htp_graph_config_;
  const QnnContext* context_;
};
} // namespace qnn
} // namespace executor
} // namespace torch
