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
    case kLogLevelVerbose:
      serverity_name = "VERBOSE";
      break;
    case kLogLevelInfo:
      serverity_name = "INFO";
      break;
    case kLogLevelWarn:
      serverity_name = "WARNING";
      break;
    case kLogLevelError:
      serverity_name = "ERROR";
      break;
    case kLogLevelDebug:
      serverity_name = "DEBUG";
      break;
    default:
      serverity_name = "Unknown severity";
      break;
  }
#ifdef __ANDROID__
  int android_severity = ANDROID_LOG_DEBUG;
  switch (log_level) {
    case kLogLevelInfo:
      android_severity = ANDROID_LOG_INFO;
      break;
    case kLogLevelWarn:
      android_severity = ANDROID_LOG_WARN;
      break;
    case kLogLevelError:
      android_severity = ANDROID_LOG_ERROR;
      break;
    case kLogLevelVerbose:
    case kLogLevelDebug:
    default:
      android_severity = ANDROID_LOG_DEBUG;
      break;
  }
  __android_log_vprint(android_severity, "[Qnn Executorch]", format, args);
#endif
  fprintf(stderr, "[%s]", serverity_name);
  vfprintf(stderr, format, args);
  va_end(args);
  fputc('\n', stderr);
}
}  // namespace qnn
}  // namespace executor
}  // namespace torch
