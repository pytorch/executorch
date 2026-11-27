// Copyright 2026 Arm Limited and/or its affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Strong override of ExecuTorch's PAL log emitter. Without this, the weak
// implementation in runtime/platform/default/minimal.cpp drops every ET_LOG()
// message on the floor, so Program::load / load_method / execute() never tell
// us why they failed. With this override + -DET_MIN_LOG_LEVEL=Debug
// -DET_LOG_ENABLED=1 every ET_LOG() reaches our retargeted printf ->
// semihosting.

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <executorch/runtime/platform/platform.h>

extern "C" void et_pal_emit_log_message(
    et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    const char* function,
    size_t line,
    const char* message,
    size_t /*length*/) {
  // The enum values are already printable ASCII (D/I/E/F/?).
  printf(
      "ET %c %s:%u %s: %s\n",
      static_cast<char>(level),
      filename ? filename : "?",
      static_cast<unsigned>(line),
      function ? function : "?",
      message);
  (void)timestamp;
}
