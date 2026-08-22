/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/runtime/platform/test/stub_platform.h>

#include <string>

class PalSpy : public PlatformIntercept {
 public:
  PalSpy() = default;

  void init() override {
    ++init_call_count;
  }

  static constexpr et_timestamp_t kTimestamp = 1234;

  et_timestamp_t current_ticks() override {
    ++current_ticks_call_count;
    return kTimestamp;
  }

  et_tick_ratio_t ticks_to_ns_multiplier() override {
    return tick_ns_multiplier;
  }

  void emit_log_message(
      et_timestamp_t timestamp,
      et_pal_log_level_t level,
      const char* filename,
      const char* function,
      size_t line,
      const char* message,
      size_t length) override {
    ++emit_log_message_call_count;
    last_log_message_args.timestamp = timestamp;
    last_log_message_args.level = level;
    last_log_message_args.filename = filename;
    last_log_message_args.function = function;
    last_log_message_args.line = line;
    last_log_message_args.message = message;
    last_log_message_args.length = length;
  }

  void* allocate(size_t size) override {
    ++allocate_call_count;
    last_allocated_size = size;
    last_allocated_ptr = (void*)0x1234;
    return nullptr;
  }

  void free(void* ptr) override {
    ++free_call_count;
    last_freed_ptr = ptr;
  }

  virtual ~PalSpy() = default;

  size_t init_call_count = 0;
  size_t current_ticks_call_count = 0;
  size_t emit_log_message_call_count = 0;
  et_tick_ratio_t tick_ns_multiplier = {1, 1};
  size_t allocate_call_count = 0;
  size_t free_call_count = 0;
  size_t last_allocated_size = 0;
  void* last_allocated_ptr = nullptr;
  void* last_freed_ptr = nullptr;

  /// The args that were passed to the most recent call to emit_log_message().
  struct {
    et_timestamp_t timestamp;
    et_pal_log_level_t level;
    std::string filename; // Copy of the char* to avoid lifetime issues.
    std::string function;
    size_t line;
    std::string message;
    size_t length;
  } last_log_message_args = {};
};
