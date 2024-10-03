/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/aot/ir/qcir_utils.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnBackendCache.h>
namespace torch {
namespace executor {
namespace qnn {
Error QnnBackendCache::GetQnnGraphInfoFromBinary() {
  const QnnSystemInterface& qnn_sys_interface =
      qnn_sys_impl_.GetQnnSystemInterface();
  std::uint32_t num_graphs;
  QnnSystemContext_GraphInfo_t* graph = nullptr;
  const QnnSystemContext_BinaryInfo_t* binaryinfo{nullptr};
  Qnn_ContextBinarySize_t binaryinfo_size = 0;
  Qnn_ErrorHandle_t error = QNN_SUCCESS;

  error = qnn_sys_interface.qnn_system_context_get_binary_info(
      sys_context_handle_,
      qnn_context_blob_.buffer,
      qnn_context_blob_.nbytes,
      &binaryinfo,
      &binaryinfo_size);

  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_WARN(
        "Failed to interpret QNN context "
        "binary. Error code %d. "
        "Try verifying binary with online-prepare format.",
        QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }

  Error status = RetrieveBackendBinaryInfo(binaryinfo);
  if (status == Error::Internal) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Failed to retrieve backend binary info from QNN context binary.");
    return Error::Internal;
  }

  if (binaryinfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
    num_graphs = binaryinfo->contextBinaryInfoV1.numGraphs;
    graph = binaryinfo->contextBinaryInfoV1.graphs;
  } else if (binaryinfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
    num_graphs = binaryinfo->contextBinaryInfoV2.numGraphs;
    graph = binaryinfo->contextBinaryInfoV2.graphs;
  } else {
    QNN_EXECUTORCH_LOG_WARN(
        "Unknown QNN BinaryInfo version %d.", binaryinfo->version);
    return Error::Internal;
  }

  if (num_graphs > 1) {
    QNN_EXECUTORCH_LOG_WARN(
        "The context binary contains %lu graphs. But now "
        "assume that one context binary contains one graph.",
        num_graphs);
    return Error::Internal;
  }

  // only have version_1 now
  if (graph[0].version != QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
    QNN_EXECUTORCH_LOG_WARN(
        "Unknown QNN GraphInfo version %d.", graph[0].version);
    return Error::Internal;
  }
  // get graph name from metadata
  graph_name_ = graph->graphInfoV1.graphName;

  // get graph inputs from metadata
  uint32_t numGraphInputs = graph->graphInfoV1.numGraphInputs;
  input_tensor_structs_.reserve(numGraphInputs);
  for (std::uint32_t i = 0; i < numGraphInputs; ++i) {
    input_tensor_structs_.emplace_back(graph->graphInfoV1.graphInputs[i]);
  }

  // get graph outputs from metadata
  uint32_t numGraphOutputs = graph->graphInfoV1.numGraphOutputs;
  output_tensor_structs_.reserve(numGraphOutputs);
  for (std::uint32_t i = 0; i < numGraphOutputs; ++i) {
    output_tensor_structs_.emplace_back(graph->graphInfoV1.graphOutputs[i]);
  }

  return Error::Ok;
}

Error QnnBackendCache::Configure() {
  if (qnn_context_blob_.buffer == nullptr) {
    state_ = SERIALIZE;
    QNN_EXECUTORCH_LOG_INFO("Caching: Caching is in SAVE MODE.");
    return Error::Ok;
  }

  if (qnn_sys_impl_.Load() != Error::Ok) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Failed to Load QnnSystem "
        "APIs. Caching mechanism is being disabled.");
    return Error::Internal;
  }

  Qnn_ErrorHandle_t error = QNN_SUCCESS;

  // create QNN SystemContext
  const QnnSystemInterface& qnn_sys_interface =
      qnn_sys_impl_.GetQnnSystemInterface();
  error = qnn_sys_interface.qnn_system_context_create(&sys_context_handle_);

  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Failed to create Qnn "
        "SystemContext. Caching mechanism will be disabled. Error code %d",
        QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }

  // DO DESERIALIZE
  state_ = DESERIALIZE;
  QNN_EXECUTORCH_LOG_INFO("Caching: Caching is in RESTORE MODE.");
  Error status = GetQnnGraphInfoFromBinary();
  if (status == Error::Internal) {
    // check if context binary came from flatbuffer
    flatbuffers::FlatBufferBuilder builder;
    flatbuffers::Verifier verifier(
        static_cast<const uint8_t* const>(qnn_context_blob_.buffer),
        qnn_context_blob_.nbytes);

    if (qcir::VerifyGraphBuffer(verifier)) {
      state_ = ONLINE_PREPARE;
      return Error::Ok;
    }

    QNN_EXECUTORCH_LOG_ERROR(
        "Failed to parse QNN Graph Info. The cache "
        "might be broken. Please consider to re-generate the "
        "cache.");
    InvalidateCache();
  }
  return Error::Ok;
}

QnnBackendCache::~QnnBackendCache() {
  Qnn_ErrorHandle_t error = QNN_SUCCESS;
  if (sys_context_handle_ != nullptr) {
    const QnnSystemInterface& qnn_sys_interface =
        qnn_sys_impl_.GetQnnSystemInterface();
    error = qnn_sys_interface.qnn_system_context_free(sys_context_handle_);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_WARN("Failed to free QNN system context.");
    }
    sys_context_handle_ = nullptr;
  }
  qnn_sys_impl_.Unload();
}

std::vector<Qnn_Tensor_t> QnnBackendCache::GetGraphInputs() {
  if (state_ != DESERIALIZE)
    return {};

  return input_tensor_structs_;
}

std::vector<Qnn_Tensor_t> QnnBackendCache::GetGraphOutputs() {
  if (state_ != DESERIALIZE)
    return {};

  return output_tensor_structs_;
}
} // namespace qnn
} // namespace executor
} // namespace torch
