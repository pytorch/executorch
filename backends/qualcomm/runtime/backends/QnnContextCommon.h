/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnBackendCache.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnBackendCommon.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnCustomProtocol.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnDeviceCommon.h>

#include <executorch/backends/qualcomm/runtime/backends/QnnProfiler.h>

#include <memory>
#include <mutex>

namespace executorch {
namespace backends {
namespace qnn {

class QnnDlcManager;

class QnnContext {
 public:
  explicit QnnContext(
      QnnImplementation* implementation,
      QnnBackend* backend,
      QnnDevice* device,
      QnnBackendCache* cache,
      QnnDlcManager* qnn_dlc_manager,
      const QnnExecuTorchProfileLevel& profile_level)
      : handle_(nullptr),
        implementation_(implementation),
        backend_(backend),
        device_(device),
        cache_(cache),
        qnn_dlc_manager_(qnn_dlc_manager),
        profile_level_(profile_level),
        is_htp_backend_(
            implementation->GetQnnInterface().GetBackendId() ==
            QNN_BACKEND_ID_HTP),
        need_to_profile_(
            profile_level != QnnExecuTorchProfileLevel::kProfileOff) {
    qnn_profiler_ =
        std::make_unique<QnnProfile>(implementation_, backend_, profile_level_);
  }

  virtual ~QnnContext();

  executorch::runtime::Error Configure();

  Qnn_ContextHandle_t GetHandle() const {
    return handle_;
  }

  std::vector<std::string> inline GetGraphNames() {
    return cache_->GetGraphNames();
  }

  std::vector<Qnn_Tensor_t> inline GetGraphInputs(
      const std::string& graph_name) {
    return cache_->GetGraphInputs(graph_name);
  }
  std::vector<Qnn_Tensor_t> inline GetGraphOutputs(
      const std::string& graph_name) {
    return cache_->GetGraphOutputs(graph_name);
  }
  QnnBackendCache::CacheState GetCacheState() const {
    return cache_->GetCacheState();
  };

  virtual executorch::runtime::Error GetContextBinary(
      QnnExecuTorchContextBinary& qnn_executorch_context_binary);

 protected:
  virtual executorch::runtime::Error MakeConfig(
      std::vector<const QnnContext_Config_t*>& config) {
    return executorch::runtime::Error::Ok;
  };
  virtual executorch::runtime::Error AfterConfigure() {
    return executorch::runtime::Error::Ok;
  };

 private:
  void WriteHeapProfile();
  Qnn_ContextHandle_t handle_;
  QnnImplementation* implementation_;
  QnnBackend* backend_;
  QnnDevice* device_;
  QnnBackendCache* cache_;
  QnnContextCustomProtocol qnn_context_custom_protocol_;
  QnnDlcManager* qnn_dlc_manager_;

  QnnExecuTorchProfileLevel profile_level_;
  std::unique_ptr<QnnProfile> qnn_profiler_;
  bool is_htp_backend_;
  bool need_to_profile_;
  static std::mutex htp_context_mutex_;
  static int htp_context_count_;
};
} // namespace qnn
} // namespace backends
} // namespace executorch
