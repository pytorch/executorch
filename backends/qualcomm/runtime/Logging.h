/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/schema_generated.h>
#include <executorch/runtime/core/error.h>
namespace torch {
namespace executor {
namespace qnn {
using namespace qnn_delegate;

void Log(QnnExecuTorchLogLevel log_level, const char* format, ...);

#define QNN_EXECUTORCH_LOG(log_level, format, ...) \
  do {                                             \
    Log(log_level, format, ##__VA_ARGS__);         \
  } while (false);

#define QNN_EXECUTORCH_LOG_ERROR(fmt, ...) \
  QNN_EXECUTORCH_LOG(QnnExecuTorchLogLevel::kLogLevelError, fmt, ##__VA_ARGS__)
#define QNN_EXECUTORCH_LOG_WARN(fmt, ...) \
  QNN_EXECUTORCH_LOG(QnnExecuTorchLogLevel::kLogLevelWarn, fmt, ##__VA_ARGS__)
#define QNN_EXECUTORCH_LOG_INFO(fmt, ...) \
  QNN_EXECUTORCH_LOG(QnnExecuTorchLogLevel::kLogLevelInfo, fmt, ##__VA_ARGS__)
#define QNN_EXECUTORCH_LOG_VERBBOSE(fmt, ...) \
  QNN_EXECUTORCH_LOG(                         \
      QnnExecuTorchLogLevel::kLogLevelVerbose, fmt, ##__VA_ARGS__)
#define QNN_EXECUTORCH_LOG_DEBUG(fmt, ...) \
  QNN_EXECUTORCH_LOG(QnnExecuTorchLogLevel::kLogLevelDebug, fmt, ##__VA_ARGS__)
} // namespace qnn
} // namespace executor
} // namespace torch
