/*
 *  Copyright (c) 2025 Samsung Electronics Co. LTD
 *  All rights reserved
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 */
#pragma once

#include <mutex>

#define FILENAME(fp) (strrchr(fp, '/') ? strrchr(fp, '/') + 1 : fp)

enum class ENN_LOG_LEVEL {
  DEBUG = 0,
  INFO = 1,
  WARNING = 2,
  ERROR = 3,
  MAX_LEVEL = ERROR,
};

class EnnLogManager {
 public:
  static void setLogLevel(ENN_LOG_LEVEL log_level);
  static bool isLogOn(ENN_LOG_LEVEL log_level);

 private:
  static ENN_LOG_LEVEL output_log_level_;
  static std::mutex log_mutex_;
};

void EnnLogImpl(ENN_LOG_LEVEL log_level, const char* format, ...);

#ifdef NDEBUG
#define __LOGX(log_level, fmt, ...)            \
  if (EnnLogManager::isLogOn(log_level)) {     \
    EnnLogImpl(log_level, fmt, ##__VA_ARGS__); \
  }
#else
#define __LOGX(log_level, fmt, ...)        \
  if (EnnLogManager::isLogOn(log_level)) { \
    EnnLogImpl(                            \
        log_level,                         \
        "[%s:%d]: " fmt,                   \
        FILENAME(__FILE__),                \
        __LINE__,                          \
        ##__VA_ARGS__);                    \
  }
#endif

#define ENN_LOG_DEBUG(fmt, ...) __LOGX(ENN_LOG_LEVEL::DEBUG, fmt, ##__VA_ARGS__)
#define ENN_LOG_INFO(fmt, ...) __LOGX(ENN_LOG_LEVEL::INFO, fmt, ##__VA_ARGS__)
#define ENN_LOG_WARN(fmt, ...) \
  __LOGX(ENN_LOG_LEVEL::WARNING, fmt, ##__VA_ARGS__)
#define ENN_LOG_ERROR(fmt, ...) __LOGX(ENN_LOG_LEVEL::ERROR, fmt, ##__VA_ARGS__)
