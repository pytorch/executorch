/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <cstdarg>
#include <cstdio>
#ifdef __ANDROID__
#include <android/log.h>
#endif
namespace torch {
namespace executor {
namespace qnn {
void Log(QnnExecuTorchLogLevel log_level, const char* format, ...) {
  va_list args;
  va_start(args, format);
  const char* serverity_name;
  switch (log_level) {
    case QnnExecuTorchLogLevel::kLogLevelVerbose:
      serverity_name = "VERBOSE";
      break;
    case QnnExecuTorchLogLevel::kLogLevelInfo:
      serverity_name = "INFO";
      break;
    case QnnExecuTorchLogLevel::kLogLevelWarn:
      serverity_name = "WARNING";
      break;
    case QnnExecuTorchLogLevel::kLogLevelError:
      serverity_name = "ERROR";
      break;
    case QnnExecuTorchLogLevel::kLogLevelDebug:
      serverity_name = "DEBUG";
      break;
    default:
      serverity_name = "Unknown severity";
      break;
  }
#ifdef __ANDROID__
  int android_severity = ANDROID_LOG_DEBUG;
  switch (log_level) {
    case QnnExecuTorchLogLevel::kLogLevelInfo:
      android_severity = ANDROID_LOG_INFO;
      break;
    case QnnExecuTorchLogLevel::kLogLevelWarn:
      android_severity = ANDROID_LOG_WARN;
      break;
    case QnnExecuTorchLogLevel::kLogLevelError:
      android_severity = ANDROID_LOG_ERROR;
      break;
    case QnnExecuTorchLogLevel::kLogLevelVerbose:
    case QnnExecuTorchLogLevel::kLogLevelDebug:
    default:
      android_severity = ANDROID_LOG_DEBUG;
      break;
  }
  __android_log_vprint(android_severity, "[Qnn ExecuTorch]", format, args);
#endif
  fprintf(stderr, "[%s] [Qnn ExecuTorch]: ", serverity_name);
  vfprintf(stderr, format, args);
  va_end(args);
  fputc('\n', stderr);
}
} // namespace qnn
} // namespace executor
} // namespace torch
