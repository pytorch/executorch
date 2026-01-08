/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/qc_compiler_spec_generated.h>
#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnImplementation.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnLogger.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnOpPackageManager.h>
#include <unordered_set>
#include <vector>

#include "HTP/QnnHtpCommon.h"
#include "QnnBackend.h"
#include "QnnCommon.h"
#include "QnnTypes.h"
#include "Saver/QnnSaverCommon.h"

namespace executorch {
namespace backends {
namespace qnn {
// qnn backend
class QnnBackend {
 public:
  explicit QnnBackend(QnnImplementation* implementation, QnnLogger* logger)
      : handle_(nullptr), implementation_(implementation), logger_(logger) {}
  QnnBackend(const QnnBackend&) = delete; // Delete copy constructor
  QnnBackend& operator=(const QnnBackend&) =
      delete; // Delete assignment operator

  virtual ~QnnBackend();
  virtual bool IsProfileEventTypeParentOfNodeTime(
      QnnProfile_EventType_t /*event_type*/) {
    return false;
  }

  executorch::runtime::Error Configure(
      const QnnExecuTorchOpPackageOptions* op_package_options);

  Qnn_ErrorHandle_t BackendValidateOpConfig(const Qnn_OpConfig_t& op_config) {
    return implementation_->GetQnnInterface().qnn_backend_validate_op_config(
        handle_, op_config);
  };

  Qnn_BackendHandle_t GetHandle() {
    return handle_;
  }

  executorch::runtime::Error VerifyQNNSDKVersion();

 protected:
  virtual Qnn_Version_t GetExpectedBackendVersion() const = 0;
  virtual executorch::runtime::Error MakeConfig(
      std::vector<const QnnBackend_Config_t*>& config) {
    return executorch::runtime::Error::Ok;
  };

 private:
  void BackendRegisterOpPackage(
      const flatbuffers::Vector<
          flatbuffers::Offset<qnn_delegate::QnnExecuTorchOpPackageInfo>>*
          op_packages_info);
  Qnn_BackendHandle_t handle_;
  QnnImplementation* implementation_;
  QnnOpPackageManager op_package_manager_;
  QnnLogger* logger_;
  executorch::runtime::Error VersionChecker(
      const Qnn_Version_t& qnn_version,
      const Qnn_Version_t& expected,
      const std::string& prefix);
};
} // namespace qnn
} // namespace backends
} // namespace executorch
