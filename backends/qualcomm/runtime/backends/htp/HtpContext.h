/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/qualcomm/runtime/backends/QnnBackendCommon.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnContextCommon.h>
#include <executorch/backends/qualcomm/runtime/backends/htp/HtpContextCustomConfig.h>

namespace executorch {
namespace backends {
namespace qnn {

class QnnDlcManager;
class HtpContext : public QnnContext {
 public:
  HtpContext(
      QnnImplementation* implementation,
      QnnSystemImplementation* system_implementation,
      QnnBackend* backend,
      QnnDevice* device,
      QnnBackendCache* cache,
      const QnnExecuTorchHtpBackendOptions* htp_options,
      const QnnExecuTorchTargetOptions* target_options,
      QnnDlcManager* qnn_dlc_manager,
      const QnnExecuTorchProfileLevel& profile_level,
      Qnn_ContextHandle_t spill_fill_handle = nullptr)
      : QnnContext(
            implementation,
            system_implementation,
            backend,
            device,
            cache,
            qnn_dlc_manager,
            profile_level),
        sf_handle_(spill_fill_handle) {
    htp_context_custom_config_ = std::make_unique<HtpContextCustomConfig>(
        this, htp_options, target_options, profile_level);
  }
  ~HtpContext() {}

  Qnn_ContextHandle_t GetSpillFillHandle() const {
    return sf_handle_;
  }

 protected:
  executorch::runtime::Error MakeConfig(
      std::vector<const QnnContext_Config_t*>& config) override;

 private:
  std::vector<QnnContext_Config_t> context_config_;
  std::unique_ptr<HtpContextCustomConfig> htp_context_custom_config_;
  Qnn_ContextHandle_t sf_handle_;
};

} // namespace qnn
} // namespace backends
} // namespace executorch
