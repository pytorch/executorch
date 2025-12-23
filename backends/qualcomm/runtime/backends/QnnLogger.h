/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/qc_compiler_spec_generated.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnImplementation.h>
namespace executorch {
namespace backends {
namespace qnn {
void LoggingCallback(
    const char* fmt,
    QnnLog_Level_t level,
    std::uint64_t /*timestamp*/,
    va_list args);

class QnnLogger {
 public:
  explicit QnnLogger(
      QnnImplementation* implementation,
      QnnLog_Callback_t callback,
      QnnExecuTorchLogLevel log_level);
  QnnLogger(const QnnLogger&) = delete; // Delete copy constructor
  QnnLogger& operator=(const QnnLogger&) = delete; // Delete assignment operator
  ~QnnLogger();

  Qnn_LogHandle_t GetHandle() {
    return handle_;
  }

  QnnExecuTorchLogLevel GetLogLevel() {
    return log_level_;
  }

 private:
  Qnn_LogHandle_t handle_;
  QnnImplementation* implementation_;
  QnnExecuTorchLogLevel log_level_;
};
} // namespace qnn
} // namespace backends
} // namespace executorch
