/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnImplementation.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnLogger.h>

#include <vector>

#include "HTP/QnnHtpCommon.h"
#include "QnnBackend.h"
#include "QnnCommon.h"
#include "QnnTypes.h"
namespace torch {
namespace executor {
namespace qnn {
// qnn backend
class QnnBackend {
 public:
  explicit QnnBackend(
      const QnnImplementation& implementation,
      QnnLogger* logger)
      : handle_(nullptr), implementation_(implementation), logger_(logger) {}

  virtual ~QnnBackend();
  virtual bool IsProfileEventTypeParentOfNodeTime(
      QnnProfile_EventType_t /*event_type*/) {
    return false;
  }

  Error Configure();

  Qnn_ErrorHandle_t BackendValidateOpConfig(const Qnn_OpConfig_t& op_config) {
    return implementation_.GetQnnInterface().qnn_backend_validate_op_config(
        handle_, op_config);
  };

  Qnn_BackendHandle_t GetHandle() {
    return handle_;
  }

  Error VerifyQNNSDKVersion(const QnnExecuTorchBackendType backend_id);

 protected:
  virtual Qnn_Version_t GetExpectedBackendVersion() const = 0;
  virtual Error MakeConfig(std::vector<const QnnBackend_Config_t*>& config) {
    return Error::Ok;
  };

 private:
  Qnn_BackendHandle_t handle_;
  const QnnImplementation& implementation_;
  QnnLogger* logger_;
  Error VersionChecker(
      const Qnn_Version_t& qnn_version,
      const Qnn_Version_t& expected,
      const std::string& prefix);
};
} // namespace qnn
} // namespace executor
} // namespace torch
