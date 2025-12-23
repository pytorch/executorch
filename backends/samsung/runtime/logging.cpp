/*
 *  Copyright (c) 2025 Samsung Electronics Co. LTD
 *  All rights reserved
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 */
#include <cstdarg>
#include <cstdio>
#ifdef __ANDROID__
#include <android/log.h>
#endif
#include <executorch/backends/samsung/runtime/logging.h>

void EnnLogImpl(ENN_LOG_LEVEL log_level, const char* format, ...) {
  va_list args;
  va_start(args, format);
#ifdef __ANDROID__
  int android_severity = ANDROID_LOG_DEBUG;
  switch (log_level) {
    case ENN_LOG_LEVEL::DEBUG:
      android_severity = ANDROID_LOG_DEBUG;
      break;
    case ENN_LOG_LEVEL::INFO:
      android_severity = ANDROID_LOG_INFO;
      break;
    case ENN_LOG_LEVEL::WARNING:
      android_severity = ANDROID_LOG_WARN;
      break;
    case ENN_LOG_LEVEL::ERROR:
      android_severity = ANDROID_LOG_ERROR;
      break;
    default:
      android_severity = ANDROID_LOG_UNKNOWN;
      break;
  }
  __android_log_print(android_severity, "[Exynos ExecuTorch]", format, args);
#endif
  const char* serverity_name;
  switch (log_level) {
    case ENN_LOG_LEVEL::DEBUG:
      serverity_name = "DEBUG";
      break;
    case ENN_LOG_LEVEL::INFO:
      serverity_name = "INFO";
      break;
    case ENN_LOG_LEVEL::WARNING:
      serverity_name = "WARN";
      break;
    case ENN_LOG_LEVEL::ERROR:
      serverity_name = "ERROR";
      break;
    default:
      serverity_name = "UNKNOWN";
      break;
  }
  fprintf(stderr, "[%s][Exynos ExecuTorch]", serverity_name);
  vfprintf(stderr, format, args);
  fputc('\n', stderr);
  va_end(args);
}

#if defined(NDEBUG)
ENN_LOG_LEVEL EnnLogManager::output_log_level_ = ENN_LOG_LEVEL::INFO;
#else
ENN_LOG_LEVEL EnnLogManager::output_log_level_ = ENN_LOG_LEVEL::DEBUG;
#endif
std::mutex EnnLogManager::log_mutex_;

void EnnLogManager::setLogLevel(ENN_LOG_LEVEL log_level) {
  std::lock_guard<std::mutex> lgd(log_mutex_);
  output_log_level_ = log_level;
}

bool EnnLogManager::isLogOn(ENN_LOG_LEVEL log_level) {
  std::lock_guard<std::mutex> lgd(log_mutex_);
  return static_cast<int>(log_level) >= static_cast<int>(output_log_level_);
}
