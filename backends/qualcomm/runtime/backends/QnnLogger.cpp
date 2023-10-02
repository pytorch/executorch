/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnImplementation.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnLogger.h>

#include <iostream>
#include <memory>

#include "QnnLog.h"
namespace torch {
namespace executor {
namespace qnn {
void LoggingCallback(
    const char* fmt,
    QnnLog_Level_t level,
    std::uint64_t /*timestamp*/,
    va_list args) {
  constexpr const int kLogBufferSize = 512;
  QnnExecuTorchLogLevel log_level;
  switch (level) {
    case QNN_LOG_LEVEL_ERROR:
      log_level = kLogLevelError;
      break;
    case QNN_LOG_LEVEL_WARN:
      log_level = kLogLevelWarn;
      break;
    default:
      log_level = kLogLevelInfo;
      break;
  }

  char buffer[kLogBufferSize];
  vsnprintf(buffer, kLogBufferSize, fmt, args);
  QNN_EXECUTORCH_LOG(log_level, buffer);
}
QnnLogger::QnnLogger(
    const QnnImplementation& implementation,
    QnnLog_Callback_t callback,
    QnnExecuTorchLogLevel log_level)
    : handle_(nullptr), implementation_(implementation) {
  const QnnInterface& qnn_interface = implementation.GetQnnInterface();

  QnnLog_Level_t qnn_log_level = QNN_LOG_LEVEL_ERROR;
  if (log_level > kLogOff) {
    switch (log_level) {
      case kLogLevelError:
        qnn_log_level = QNN_LOG_LEVEL_ERROR;
        break;
      case kLogLevelWarn:
        qnn_log_level = QNN_LOG_LEVEL_WARN;
        break;
      case kLogLevelInfo:
        qnn_log_level = QNN_LOG_LEVEL_INFO;
        break;
      case kLogLevelVerbose:
        qnn_log_level = QNN_LOG_LEVEL_VERBOSE;
        break;
      case kLogLevelDebug:
        qnn_log_level = QNN_LOG_LEVEL_DEBUG;
        break;
      default:
        QNN_EXECUTORCH_LOG(
            kLogLevelError,
            "[Qnn ExecuTorch] Unknown logging level %d",
            log_level);
    }
    QNN_EXECUTORCH_LOG(
        kLogLevelInfo,
        "[Qnn ExecuTorch] create QNN Logger with log_level %d",
        log_level);
    Qnn_ErrorHandle_t error =
        qnn_interface.qnn_log_create(callback, qnn_log_level, &handle_);

    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG(
          kLogLevelWarn,
          "[Qnn ExecuTorch] Failed to create log_handle for backend "
          " %u, qnn_log_level %d, error=%d",
          qnn_interface.GetBackendId(),
          qnn_log_level,
          QNN_GET_ERROR_CODE(error));

      // ignore error and continue to create backend handle...
      handle_ = nullptr;
    }
  }
}

QnnLogger::~QnnLogger() {
  const QnnInterface& qnn_interface = implementation_.GetQnnInterface();
  if (handle_ != nullptr) {
    Qnn_ErrorHandle_t error = qnn_interface.qnn_log_free(handle_);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG(
          kLogLevelError,
          "[Qnn ExecuTorch] Failed to free QNN log_handle. Backend "
          "ID %u, error %d",
          qnn_interface.GetBackendId(),
          QNN_GET_ERROR_CODE(error));
    }
    handle_ = nullptr;
  }
}
} // namespace qnn
} // namespace executor
} // namespace torch
