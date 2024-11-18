/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/runtime/platform/compiler.h>
#include <stdio.h>
#include <time.h>
#include <cctype>
#if defined(__linux__) || defined(__ANDROID__) || defined(__unix__)
#include <sys/resource.h>
#endif

namespace executorch {
namespace extension {
namespace llm {

ET_EXPERIMENTAL void inline safe_printf(const char* piece) {
  // piece might be a raw byte token, and we only want to print printable chars
  // or whitespace because some of the other bytes can be various control codes,
  // backspace, etc.
  if (piece == nullptr) {
    return;
  }
  if (piece[0] == '\0') {
    return;
  }
  if (piece[1] == '\0') {
    unsigned char byte_val = piece[0];
    if (!(isprint(byte_val) || isspace(byte_val))) {
      return; // bad byte, don't print it
    }
  }
  printf("%s", piece);
}

// ----------------------------------------------------------------------------
// utilities: time

ET_EXPERIMENTAL long inline time_in_ms() {
  // return time in milliseconds, for benchmarking the model speed
  struct timespec time;
  clock_gettime(CLOCK_REALTIME, &time);
  return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// utilities: memory usage

// Returns the current RSS in bytes. Returns 0 if not supported.
// RSS: Resident Set Size, the amount of memory currently in the RAM for this
// process. These values are approximate, and are only used for logging
// purposes.
ET_EXPERIMENTAL size_t inline get_rss_bytes() {
#if defined(__linux__) || defined(__ANDROID__) || defined(__unix__)
  struct rusage r_usage;
  if (getrusage(RUSAGE_SELF, &r_usage) == 0) {
    return r_usage.ru_maxrss * 1024;
  }
#endif // __linux__ || __ANDROID__ || __unix__
  // Unsupported platform like Windows, or getrusage() failed.
  // __APPLE__ and __MACH__ are not supported because r_usage.ru_maxrss does not
  // consistently return kbytes on macOS. On older versions of macOS, it
  // returns bytes, but on newer versions it returns kbytes. Need to figure out
  // when this changed.
  return 0;
}
} // namespace llm
} // namespace extension
} // namespace executorch

namespace torch {
namespace executor {
namespace util {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::extension::llm::get_rss_bytes;
using ::executorch::extension::llm::safe_printf;
using ::executorch::extension::llm::time_in_ms;
} // namespace util
} // namespace executor
} // namespace torch
