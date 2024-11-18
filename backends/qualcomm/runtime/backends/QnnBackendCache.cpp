/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/aot/ir/qcir_utils.h>
#include <executorch/backends/qualcomm/qc_binary_info_generated.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnBackendCache.h>

namespace executorch {
namespace backends {
namespace qnn {

using executorch::runtime::Error;

Error QnnBackendCache::GetQnnGraphInfoFromBinary(
    void* buffer,
    uint32_t nbytes) {
  const QnnSystemInterface& qnn_sys_interface =
      qnn_sys_impl_.GetQnnSystemInterface();
  std::uint32_t num_graphs;
  QnnSystemContext_GraphInfo_t* graphs = nullptr;
  const QnnSystemContext_BinaryInfo_t* binaryinfo{nullptr};
  Qnn_ContextBinarySize_t binaryinfo_size = 0;
  Qnn_ErrorHandle_t error = QNN_SUCCESS;

  error = qnn_sys_interface.qnn_system_context_get_binary_info(
      sys_context_handle_, buffer, nbytes, &binaryinfo, &binaryinfo_size);

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
    graphs = binaryinfo->contextBinaryInfoV1.graphs;
  } else if (binaryinfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
    num_graphs = binaryinfo->contextBinaryInfoV2.numGraphs;
    graphs = binaryinfo->contextBinaryInfoV2.graphs;
  } else {
    QNN_EXECUTORCH_LOG_WARN(
        "Unknown QNN BinaryInfo version %d.", binaryinfo->version);
    return Error::Internal;
  }

  for (std::uint32_t i = 0; i < num_graphs; ++i) {
    if (graphs->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
      RetrieveGraphInfo<QnnSystemContext_GraphInfoV1_t>(graphs[i].graphInfoV1);
    } else if (graphs->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
      RetrieveGraphInfo<QnnSystemContext_GraphInfoV2_t>(graphs[i].graphInfoV2);
    } else {
      QNN_EXECUTORCH_LOG_WARN(
          "Unknown QNN GraphInfo version %d.", binaryinfo->version);
      return Error::Internal;
    }
  }

  return Error::Ok;
}

Error QnnBackendCache::Configure() {
  if (qnn_context_blob_.buffer == nullptr) {
    state_ = SERIALIZE;
    // use aot_graph_name if we're lowering graph on host side
    graph_names_.push_back(aot_graph_name_);
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
  flatbuffers::Verifier verifier_binary_info(
      static_cast<const uint8_t* const>(qnn_context_blob_.buffer),
      qnn_context_blob_.nbytes);
  if (!qnn_delegate::VerifyBinaryInfoBuffer(verifier_binary_info)) {
    QNN_EXECUTORCH_LOG_ERROR("Fail to verify binary info");
    return Error::Internal;
  }

  auto binary_info = GetBinaryInfo(qnn_context_blob_.buffer);
  Error status = GetQnnGraphInfoFromBinary(
      const_cast<uint8_t*>(binary_info->data()->data()),
      binary_info->data()->size());

  if (status == Error::Internal) {
    // check if context binary came from flatbuffer
    flatbuffers::Verifier verifier(
        binary_info->data()->data(), binary_info->data()->size());

    if (qcir::VerifyContextBuffer(verifier)) {
      state_ = ONLINE_PREPARE;
      auto context = qcir::GetContext(binary_info->data()->data());
      for (const auto& graph : *context->graphs()) {
        graph_names_.emplace_back(graph->name()->str());
      }
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

std::vector<Qnn_Tensor_t> QnnBackendCache::GetGraphInputs(
    const std::string& graph_name) {
  if (state_ != DESERIALIZE)
    return {};

  return input_tensor_structs_[graph_name];
}

std::vector<Qnn_Tensor_t> QnnBackendCache::GetGraphOutputs(
    const std::string& graph_name) {
  if (state_ != DESERIALIZE)
    return {};

  return output_tensor_structs_[graph_name];
}

template <typename INFO>
void QnnBackendCache::RetrieveGraphInfo(const INFO& info) {
  // get graph name from metadata
  graph_names_.push_back(info.graphName);
  // get graph inputs from metadata
  uint32_t numGraphInputs = info.numGraphInputs;
  input_tensor_structs_[graph_names_.back()].reserve(numGraphInputs);
  for (std::uint32_t i = 0; i < numGraphInputs; ++i) {
    input_tensor_structs_[graph_names_.back()].emplace_back(
        info.graphInputs[i]);
  }
  // get graph outputs from metadata
  uint32_t numGraphOutputs = info.numGraphOutputs;
  output_tensor_structs_[graph_names_.back()].reserve(numGraphOutputs);
  for (std::uint32_t i = 0; i < numGraphOutputs; ++i) {
    output_tensor_structs_[graph_names_.back()].emplace_back(
        info.graphOutputs[i]);
  }
}

} // namespace qnn
} // namespace backends
} // namespace executorch
