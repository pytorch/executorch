/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/backends/QnnGraphCommon.h>
namespace torch {
namespace executor {
namespace qnn {
Error QnnGraph::Configure() {
  // create qnn backend
  const QnnInterface& qnn_interface = implementation_.GetQnnInterface();
  Qnn_ErrorHandle_t error = QNN_SUCCESS;

  std::vector<const QnnGraph_Config_t*> temp_graph_config;
  ET_CHECK_OR_RETURN_ERROR(
      MakeConfig(temp_graph_config) == Error::Ok,
      Internal,
      "Fail to make graph config.");

  if (context_->GetCacheState() == QnnBackendCache::DESERIALIZE) {
    // retrieve QNN Graph
    error = qnn_interface.qnn_graph_retrieve(
        context_->GetHandle(), context_->GetGraphName().c_str(), &handle_);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Can't retrieve graph "
          "%s from context. Error %d.",
          context_->GetGraphName().c_str(),
          QNN_GET_ERROR_CODE(error));
      return Error::Internal;
    }
  } else if (
      context_->GetCacheState() == QnnBackendCache::SERIALIZE ||
      context_->GetCacheState() == QnnBackendCache::ONLINE_PREPARE) {
    Qnn_ErrorHandle_t error = qnn_interface.qnn_graph_create(
        context_->GetHandle(),
        graph_name_.c_str(),
        temp_graph_config.empty() ? nullptr : temp_graph_config.data(),
        &handle_);

    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "qnn_graph_create failed. Error  %d", QNN_GET_ERROR_CODE(error));
      return Error::Internal;
    }
  } else {
    QNN_EXECUTORCH_LOG_ERROR("QNN context cache is invalid.");
    return Error::Internal;
  }

  // The profiler needs to be created after the backend is created.
  profile_ =
      std::make_unique<QnnProfile>(implementation_, backend_, profile_level_);
  return Error::Ok;
}

Qnn_ErrorHandle_t QnnGraph::GraphExecute(
    const std::vector<Qnn_Tensor_t>& input_tensor_structs,
    std::vector<Qnn_Tensor_t>& output_tensor_structs) {
  return implementation_.GetQnnInterface().qnn_graph_execute(
      handle_,
      input_tensor_structs.data(),
      input_tensor_structs.size(),
      output_tensor_structs.data(),
      output_tensor_structs.size(),
      profile_->GetHandle(),
      /*signalHandle=*/nullptr);
};

Error QnnGraph::EnsureTensorInQnnGraph(
    const std::shared_ptr<TensorWrapper>& tensor_wrapper) {
  const QnnInterface& qnn_interface = implementation_.GetQnnInterface();
  Qnn_ErrorHandle_t error = QNN_SUCCESS;

  if (!tensor_wrapper->IsTensorCreated()) {
    Qnn_Tensor_t tensor = tensor_wrapper->CloneTensorStruct();

    error = qnn_interface.qnn_tensor_create_graph_tensor(handle_, &tensor);

    int name_conflict_count = 0;
    while (error == QNN_TENSOR_ERROR_NAME_HASH_COLLISION) {
      const std::string& old_name = tensor_wrapper->GetName();

      std::string new_name =
          old_name + "_" + std::to_string(name_conflict_count);
      tensor_wrapper->SetName(new_name);
      QNN_VER_PTR(tensor)->name = new_name.c_str();

      QNN_EXECUTORCH_LOG_INFO(
          "tensor name %s hash collision, change to %s",
          old_name.c_str(),
          new_name.c_str());

      // update
      name_conflict_count++;
      error = qnn_interface.qnn_tensor_create_graph_tensor(handle_, &tensor);
    }
    tensor_wrapper->UpdateQnnTensorMeta(tensor);
    tensor_wrapper->SetTensorCreated();
  }
  return Error::Ok;
}
} // namespace qnn
} // namespace executor
} // namespace torch
