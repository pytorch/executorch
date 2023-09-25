/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef EXECUTORCH_QNN_EXECUTORCH_BACKENDS_HTP_BACKEND_HTP_GRAPH_H_
#define EXECUTORCH_QNN_EXECUTORCH_BACKENDS_HTP_BACKEND_HTP_GRAPH_H_
#include <executorch/backends/qnn/runtime/backends/QnnGraphCommon.h>
#include <executorch/backends/qnn/runtime/backends/htpbackend/HtpGraphCustomConfig.h>

#include <memory>

#include "HTP/QnnHtpGraph.h"
namespace torch {
namespace executor {
namespace qnn {
class HtpGraph : public QnnGraph {
 public:
  HtpGraph(const QnnImplementation& implementation, QnnContext* context,
           const std::string& graph_name,
           const QnnExecuTorchHtpBackendOptions& htp_options)
      : QnnGraph(implementation, context, graph_name),
        htp_options_(htp_options) {
    htp_graph_custom_config_ =
        std::make_unique<HtpGraphCustomConfig>(htp_options);
  }
  ~HtpGraph() {}

 protected:
  Error MakeConfig(std::vector<const QnnGraph_Config_t*>& config) override;

 private:
  std::vector<QnnGraph_Config_t> graph_config_;
  std::unique_ptr<HtpGraphCustomConfig> htp_graph_custom_config_;
  [[maybe_unused]] QnnExecuTorchHtpBackendOptions htp_options_;
};
}  // namespace qnn
}  // namespace executor
}  // namespace torch
#endif  // EXECUTORCH_QNN_EXECUTORCH_BACKENDS_HTP_BACKEND_HTP_GRAPH_H_
