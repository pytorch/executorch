/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/aot/wrappers/TensorWrapper.h>
#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnContextCommon.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnImplementation.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnProfiler.h>

#include <vector>

#include "QnnCommon.h"
namespace executorch {
namespace backends {
namespace qnn {
// qnn graph
class QnnGraph {
 public:
  explicit QnnGraph(
      QnnImplementation* implementation,
      QnnBackend* backend,
      QnnContext* context,
      const QnnExecuTorchProfileLevel& profile_level)
      : implementation_(implementation),
        backend_(backend),
        context_(context),
        profile_level_(profile_level) {}

  virtual ~QnnGraph(){};

  executorch::runtime::Error Configure(const std::string& graph_name);

  Qnn_ErrorHandle_t GraphExecute(
      const std::string& graph_name,
      const std::vector<Qnn_Tensor_t>& input_tensor_structs,
      std::vector<Qnn_Tensor_t>& output_tensor_structs);

  Qnn_ErrorHandle_t GraphAddNode(
      const std::string& graph_name,
      const Qnn_OpConfig_t& op_config) {
    return implementation_->GetQnnInterface().qnn_graph_add_node(
        handle_[graph_name], op_config);
  };
  executorch::runtime::Error EnsureTensorInQnnGraph(
      const std::string& graph_name,
      const std::shared_ptr<TensorWrapper>& tensor_wrapper);

  Qnn_ErrorHandle_t GraphFinalize(const std::string& graph_name) {
    return implementation_->GetQnnInterface().qnn_graph_finalize(
        handle_[graph_name],
        profile_[graph_name]->GetHandle(),
        nullptr /* signal_handle */);
  };
  Qnn_ErrorHandle_t ProfileExecuteData(
      const std::string& graph_name,
      executorch::runtime::EventTracer* event_tracer) {
    return profile_[graph_name]->ProfileData(event_tracer);
  };
  Qnn_GraphHandle_t GetHandle(const std::string& graph_name) {
    return handle_[graph_name];
  }

  void SetGraphHandle(
      const std::string& graph_name,
      Qnn_GraphHandle_t graph_handle) {
    handle_[graph_name] = graph_handle;
  }

  QnnProfile* GetProfile(const std::string& graph_name) {
    return profile_[graph_name].get();
  }

 protected:
  virtual executorch::runtime::Error MakeConfig(
      std::vector<const QnnGraph_Config_t*>& config) {
    return executorch::runtime::Error::Ok;
  };

 private:
  std::unordered_map<std::string, Qnn_GraphHandle_t> handle_;
  QnnImplementation* implementation_;
  QnnBackend* backend_;
  QnnContext* context_;
  QnnExecuTorchProfileLevel profile_level_;
  std::unordered_map<std::string, std::unique_ptr<QnnProfile>> profile_;
};
} // namespace qnn
} // namespace backends
} // namespace executorch
