/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>

#include <executorch/runtime/platform/compiler.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/types.h>

/**
 * An interface for intercepting calls to the PAL layer.
 */
class PlatformIntercept {
 public:
  PlatformIntercept() = default;

  /// Called when et_pal_init() is called.
  virtual void init() {}

  // We can't intercept et_pal_abort() since it's marked NORETURN.

  /// Called when et_pal_current_ticks() is called.
  virtual et_timestamp_t current_ticks() {
    return 0;
  }

  virtual et_tick_ratio_t ticks_to_ns_multiplier() {
    return {1, 1};
  }

  /// Called when et_pal_emit_log_message() is called.
  virtual void emit_log_message(
      ET_UNUSED et_timestamp_t timestamp,
      ET_UNUSED et_pal_log_level_t level,
      ET_UNUSED const char* filename,
      ET_UNUSED const char* function,
      ET_UNUSED size_t line,
      ET_UNUSED const char* message,
      ET_UNUSED size_t length) {}

  virtual void* allocate(ET_UNUSED size_t size) {
    return nullptr;
  }

  virtual void free(ET_UNUSED void* ptr) {}

  virtual ~PlatformIntercept() = default;
};

/**
 * RAII type to install a PlatformIntercept for the duration of a test case.
 */
class InterceptWith {
 public:
  explicit InterceptWith(PlatformIntercept& pi) {
    InterceptWith::install(&pi);
  }

  ~InterceptWith() {
    InterceptWith::install(nullptr);
  }

 private:
  /**
   * Installs the PlatformIntercept to forward to when et_pal_* functions are
   * called. To uninstall, pass in `nullptr`.
   */
  static void install(PlatformIntercept* pi);
};
