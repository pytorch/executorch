/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "log.h"

#ifdef __ANDROID__

#include <android/log.h>
#include <functional>
#include <mutex>
#include <sstream>

using executorch::extension::log_entry;

// Number of entries to store in the in-memory log buffer.
const size_t log_buffer_length = 16;

namespace {
std::vector<log_entry> log_buffer_;
std::mutex log_buffer_mutex_;
} // namespace

// For Android, write to logcat
void et_pal_emit_log_message(
    et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    const char* function,
    size_t line,
    const char* message,
    size_t length) {
  std::lock_guard<std::mutex> guard(log_buffer_mutex_);

  while (log_buffer_.size() >= log_buffer_length) {
    log_buffer_.erase(log_buffer_.begin());
  }

  log_buffer_.emplace_back(
      timestamp, level, filename, function, line, message, length);

  int android_log_level = ANDROID_LOG_UNKNOWN;
  if (level == 'D') {
    android_log_level = ANDROID_LOG_DEBUG;
  } else if (level == 'I') {
    android_log_level = ANDROID_LOG_INFO;
  } else if (level == 'E') {
    android_log_level = ANDROID_LOG_ERROR;
  } else if (level == 'F') {
    android_log_level = ANDROID_LOG_FATAL;
  }

  __android_log_print(android_log_level, "ExecuTorch", "%s", message);
}

namespace executorch::extension {

void access_log_buffer(std::function<void(std::vector<log_entry>&)> accessor) {
  std::lock_guard<std::mutex> guard(log_buffer_mutex_);
  accessor(log_buffer_);
}

} // namespace executorch::extension

#else

#include <cstdio>

void et_pal_emit_log_message(
    et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    const char* function,
    size_t line,
    const char* message,
    size_t length) {
  printf("%c executorch:%s:%zu] %s\n", level, filename, line, message);
  fflush(stdout);
}

#endif
