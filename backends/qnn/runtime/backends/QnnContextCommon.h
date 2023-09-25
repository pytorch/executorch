/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef EXECUTORCH_QNN_EXECUTORCH_BACKENDS_QNN_CONTEXT_COMMON_H_
#define EXECUTORCH_QNN_EXECUTORCH_BACKENDS_QNN_CONTEXT_COMMON_H_

#include <executorch/backends/qnn/runtime/Logging.h>
#include <executorch/backends/qnn/runtime/backends/QnnBackendCache.h>
#include <executorch/backends/qnn/runtime/backends/QnnBackendCommon.h>
#include <executorch/backends/qnn/runtime/backends/QnnDeviceCommon.h>

#include <memory>
namespace torch {
namespace executor {
namespace qnn {
class QnnContext {
 public:
  explicit QnnContext(const QnnImplementation& implementation,
                      QnnBackend* backend, QnnDevice* device,
                      const QnnExecuTorchContextBinary& qnn_context_blob)
      : handle_(nullptr),
        implementation_(implementation),
        backend_(backend),
        device_(device) {
    cache_ = std::make_unique<QnnBackendCache>(qnn_context_blob);
  }

  virtual ~QnnContext();
  Error Configure();

  Qnn_ContextHandle_t GetHandle() { return handle_; }

  std::string GetGraphName() { return cache_->GetGraphName(); }

  std::vector<Qnn_Tensor_t> GetGraphInputs() {
    return cache_->GetGraphInputs();
  }
  std::vector<Qnn_Tensor_t> GetGraphOutputs() {
    return cache_->GetGraphOutputs();
  }
  QnnBackendCache::CacheState GetCacheState() {
    return cache_->GetCacheState();
  };

  Error GetContextBinary(
      QnnExecuTorchContextBinary& qnn_executorch_context_binary);

 protected:
  virtual Error MakeConfig(std::vector<const QnnContext_Config_t*>& config) {
    return Error::Ok;
  };

 private:
  Qnn_ContextHandle_t handle_;
  const QnnImplementation& implementation_;
  QnnBackend* backend_;
  QnnDevice* device_;
  std::unique_ptr<QnnBackendCache> cache_;
  std::vector<char> binary_buffer_;
};
}  // namespace qnn
}  // namespace executor
}  // namespace torch

#endif  // EXECUTORCH_QNN_EXECUTORCH_BACKENDS_QNN_CONTEXT_COMMON_H_
