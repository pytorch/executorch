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
#include <vector>
namespace torch {
namespace executor {
namespace qnn {
class QnnBackendCache {
 public:
  enum CacheState {
    INVALID = 0,
    SERIALIZE = 1,
    DESERIALIZE = 2,
    ONLINE_PREPARE = 3,
  };
  explicit QnnBackendCache(const QnnExecuTorchContextBinary& qnn_context_blob)
      : qnn_context_blob_(qnn_context_blob) {}
  virtual ~QnnBackendCache();
  QnnBackendCache(const QnnBackendCache&) = delete;
  QnnBackendCache(QnnBackendCache&&) = delete;
  QnnBackendCache& operator=(const QnnBackendCache&) = delete;
  QnnBackendCache& operator=(QnnBackendCache&&) = delete;

  std::vector<Qnn_Tensor_t> GetGraphInputs();

  std::vector<Qnn_Tensor_t> GetGraphOutputs();

  const QnnExecuTorchContextBinary& GetQnnContextBlob() {
    return qnn_context_blob_;
  };

  QnnBackendCache::CacheState GetCacheState() {
    return state_;
  };

  void InvalidateCache() {
    state_ = INVALID;
  }

  std::string GetGraphName() {
    return graph_name_;
  }

  Error Configure();

 protected:
  virtual Error RetrieveBackendBinaryInfo(
      __ET_UNUSED const QnnSystemContext_BinaryInfo_t* binaryinfo) {
    return Error::Ok;
  }

 private:
  Error GetQnnGraphInfoFromBinary();

  CacheState state_{INVALID};

  QnnExecuTorchContextBinary qnn_context_blob_;
  QnnSystemContext_Handle_t sys_context_handle_{nullptr};
  QnnSystemImplementation qnn_sys_impl_{"libQnnSystem.so"};
  std::string graph_name_;
  std::vector<Qnn_Tensor_t> input_tensor_structs_;
  std::vector<Qnn_Tensor_t> output_tensor_structs_;
};
} // namespace qnn
} // namespace executor
} // namespace torch
