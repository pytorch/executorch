/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <cstdlib>

namespace executorch::runtime {

namespace {
/**
 * The singleton instance of the PAL function table.
 */
PalImpl pal_impl = {
    et_pal_init,
    et_pal_abort,
    et_pal_current_ticks,
    et_pal_ticks_to_ns_multiplier,
    et_pal_emit_log_message,
    et_pal_allocate,
    et_pal_free,
    __FILE__};

/**
 * Tracks whether the PAL has been overridden. This is used to warn when
 * multiple callers override the PAL.
 */
bool is_pal_overridden = false;
} // namespace

PalImpl PalImpl::create(
    pal_emit_log_message_method emit_log_message,
    const char* source_filename) {
  return PalImpl::create(
      nullptr, // init
      nullptr, // abort
      nullptr, // current_ticks
      nullptr, // ticks_to_ns_multiplier
      emit_log_message,
      nullptr, // allocate
      nullptr, // free
      source_filename);
}

PalImpl PalImpl::create(
    pal_init_method init,
    pal_abort_method abort,
    pal_current_ticks_method current_ticks,
    pal_ticks_to_ns_multiplier_method ticks_to_ns_multiplier,
    pal_emit_log_message_method emit_log_message,
    pal_allocate_method allocate,
    pal_free_method free,
    const char* source_filename) {
  return PalImpl{
      init,
      abort,
      current_ticks,
      ticks_to_ns_multiplier,
      emit_log_message,
      allocate,
      free,
      source_filename};
}

/**
 * Override the PAL functions with user implementations. Any null entries in the
 * table are unchanged and will keep the default implementation.
 */
bool register_pal(PalImpl impl) {
  if (is_pal_overridden) {
    ET_LOG(
        Error,
        "register_pal() called multiple times. Subsequent calls will override the previous implementation. Previous implementation was registered from %s.",
        impl.source_filename != nullptr ? impl.source_filename : "unknown");
  }
  is_pal_overridden = true;

  if (impl.abort != nullptr) {
    pal_impl.abort = impl.abort;
  }

  if (impl.current_ticks != nullptr) {
    pal_impl.current_ticks = impl.current_ticks;
  }

  if (impl.ticks_to_ns_multiplier != nullptr) {
    pal_impl.ticks_to_ns_multiplier = impl.ticks_to_ns_multiplier;
  }

  if (impl.emit_log_message != nullptr) {
    pal_impl.emit_log_message = impl.emit_log_message;
  }

  if (impl.allocate != nullptr) {
    pal_impl.allocate = impl.allocate;
  }

  if (impl.free != nullptr) {
    pal_impl.free = impl.free;
  }

  if (impl.init != nullptr) {
    pal_impl.init = impl.init;
    if (pal_impl.init != nullptr) {
      pal_impl.init();
    }
  }

  return true;
}

const PalImpl* get_pal_impl() {
  return &pal_impl;
}

void pal_init() {
  pal_impl.init();
}

ET_NORETURN void pal_abort() {
  pal_impl.abort();
  // This should be unreachable, but in case the PAL implementation doesn't
  // abort, force it here.
  std::abort();
}

et_timestamp_t pal_current_ticks() {
  return pal_impl.current_ticks();
}

et_tick_ratio_t pal_ticks_to_ns_multiplier() {
  return pal_impl.ticks_to_ns_multiplier();
}

void pal_emit_log_message(
    et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    const char* function,
    size_t line,
    const char* message,
    size_t length) {
  pal_impl.emit_log_message(
      timestamp, level, filename, function, line, message, length);
}

void* pal_allocate(size_t size) {
  return pal_impl.allocate(size);
}

void pal_free(void* ptr) {
  pal_impl.free(ptr);
}

} // namespace executorch::runtime
