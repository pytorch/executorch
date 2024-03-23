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
namespace torch {
namespace executor {
namespace qnn {
// qnn graph
class QnnGraph {
 public:
  explicit QnnGraph(
      const QnnImplementation& implementation,
      QnnBackend* backend,
      QnnContext* context,
      const QnnExecuTorchProfileLevel& profile_level,
      const std::string& graph_name)
      : handle_(nullptr),
        implementation_(implementation),
        backend_(backend),
        context_(context),
        profile_level_(profile_level),
        graph_name_(graph_name) {}

  virtual ~QnnGraph(){};

  Error Configure();

  Qnn_ErrorHandle_t GraphExecute(
      const std::vector<Qnn_Tensor_t>& input_tensor_structs,
      std::vector<Qnn_Tensor_t>& output_tensor_structs);

  Qnn_ErrorHandle_t GraphAddNode(const Qnn_OpConfig_t& op_config) {
    return implementation_.GetQnnInterface().qnn_graph_add_node(
        handle_, op_config);
  };
  Error EnsureTensorInQnnGraph(
      const std::shared_ptr<TensorWrapper>& tensor_wrapper);

  Qnn_ErrorHandle_t GraphFinalize() {
    return implementation_.GetQnnInterface().qnn_graph_finalize(
        handle_, nullptr /* profile_handle */, nullptr /* signal_handle */);
  };
  Qnn_ErrorHandle_t ProfileExecuteData(EventTracer* event_tracer) {
    return profile_->ProfileData(event_tracer);
  };
  Qnn_GraphHandle_t GetHandle() {
    return handle_;
  }

 protected:
  virtual Error MakeConfig(std::vector<const QnnGraph_Config_t*>& config) {
    return Error::Ok;
  };

 private:
  Qnn_GraphHandle_t handle_;
  const QnnImplementation& implementation_;
  QnnBackend* backend_;
  QnnContext* context_;
  QnnExecuTorchProfileLevel profile_level_;
  std::string graph_name_;
  std::unique_ptr<QnnProfile> profile_;
};
} // namespace qnn
} // namespace executor
} // namespace torch
