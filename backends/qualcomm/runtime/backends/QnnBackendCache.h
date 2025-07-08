/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnSysImplementation.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace executorch {
namespace backends {
namespace qnn {
class QnnBackendCache {
 public:
  enum CacheState {
    INVALID = 0,
    SERIALIZE = 1,
    DESERIALIZE = 2,
    ONLINE_PREPARE = 3,
    MULTI_GRAPH = 4,
  };
  explicit QnnBackendCache(const QnnExecuTorchContextBinary& qnn_context_blob)
      : qnn_context_blob_(qnn_context_blob) {}
  virtual ~QnnBackendCache();
  QnnBackendCache(const QnnBackendCache&) = delete;
  QnnBackendCache(QnnBackendCache&&) = delete;
  QnnBackendCache& operator=(const QnnBackendCache&) = delete;
  QnnBackendCache& operator=(QnnBackendCache&&) = delete;

  std::vector<Qnn_Tensor_t> GetGraphInputs(const std::string& graph_name);

  std::vector<Qnn_Tensor_t> GetGraphOutputs(const std::string& graph_name);

  const QnnExecuTorchContextBinary& GetQnnContextBlob() {
    return qnn_context_blob_;
  };

  QnnBackendCache::CacheState GetCacheState() {
    return state_;
  };

  void InvalidateCache() {
    state_ = INVALID;
  }

  std::vector<std::string> GetGraphNames() {
    return graph_names_;
  }

  void SetGraphNames(const std::string& graph_name) {
    graph_names_.emplace_back(graph_name);
  }

  executorch::runtime::Error Configure(
      const std::vector<std::string>& graph_names);

 protected:
  virtual executorch::runtime::Error RetrieveBackendBinaryInfo(
      __ET_UNUSED const QnnSystemContext_BinaryInfo_t* binaryinfo) {
    return executorch::runtime::Error::Ok;
  }

 private:
  executorch::runtime::Error GetQnnGraphInfoFromBinary(
      void* buffer,
      uint32_t nbytes);

  template <typename INFO>
  void RetrieveGraphInfo(const INFO& info);

  CacheState state_{INVALID};

  QnnExecuTorchContextBinary qnn_context_blob_;
  QnnSystemContext_Handle_t sys_context_handle_{nullptr};
  QnnSystemImplementation qnn_sys_impl_{"libQnnSystem.so"};
  std::vector<std::string> graph_names_;
  std::unordered_map<std::string, std::vector<Qnn_Tensor_t>>
      input_tensor_structs_;
  std::unordered_map<std::string, std::vector<Qnn_Tensor_t>>
      output_tensor_structs_;
};
} // namespace qnn
} // namespace backends
} // namespace executorch
