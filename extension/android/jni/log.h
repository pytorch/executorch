/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <functional>
#include <string>
#include <vector>

#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>

namespace executorch::extension {
struct log_entry {
  et_timestamp_t timestamp;
  et_pal_log_level_t level;
  std::string filename;
  std::string function;
  size_t line;
  std::string message;

  log_entry(
      et_timestamp_t timestamp,
      et_pal_log_level_t level,
      const char* filename,
      const char* function,
      size_t line,
      const char* message,
      size_t length)
      : timestamp(timestamp),
        level(level),
        filename(filename),
        function(function),
        line(line),
        message(message, length) {}
};

void access_log_buffer(std::function<void(std::vector<log_entry>&)> accessor);
} // namespace executorch::extension
