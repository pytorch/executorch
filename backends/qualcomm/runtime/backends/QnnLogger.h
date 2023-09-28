/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef EXECUTORCH_QNN_EXECUTORCH_BACKENDS_QNN_LOGGER_H_
#define EXECUTORCH_QNN_EXECUTORCH_BACKENDS_QNN_LOGGER_H_
#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnImplementation.h>
namespace torch {
namespace executor {
namespace qnn {
void LoggingCallback(const char* fmt, QnnLog_Level_t level,
                     std::uint64_t /*timestamp*/, va_list args);

class QnnLogger {
 public:
  explicit QnnLogger(const QnnImplementation& implementation,
                     QnnLog_Callback_t callback,
                     QnnExecuTorchLogLevel log_level);
  ~QnnLogger();

  Qnn_LogHandle_t GetHandle() { return handle_; }

 private:
  Qnn_LogHandle_t handle_;
  const QnnImplementation& implementation_;
};
}  // namespace qnn
}  // namespace executor
}  // namespace torch

#endif  // EXECUTORCH_QNN_EXECUTORCH_BACKENDS_QNN_LOGGER_H_
