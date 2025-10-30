/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if defined(__XTENSA__)

#include <stdio.h>
#include <sys/times.h>

#include <xtensa/sim.h>

#include <executorch/runtime/platform/platform.h>

#define ET_LOG_OUTPUT_FILE stdout

void et_pal_emit_log_message(
    et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    ET_UNUSED const char* function,
    size_t line,
    const char* message,
    ET_UNUSED size_t length) {
  // Not all platforms have ticks == nanoseconds, but this one does.
  timestamp /= 1000; // To microseconds
  int us = timestamp % 1000000;
  timestamp /= 1000000; // To seconds
  int sec = timestamp % 60;
  timestamp /= 60; // To minutes
  int min = timestamp % 60;
  timestamp /= 60; // To hours
  int hour = timestamp;

  fprintf(
      ET_LOG_OUTPUT_FILE,
      "%c %02d:%02d:%02d.%06d executorch:%s:%d] %s\n",
      static_cast<char>(level),
      hour,
      min,
      sec,
      us,
      filename,
      static_cast<int>(line),
      message);
  fflush(ET_LOG_OUTPUT_FILE);
}

et_timestamp_t et_pal_current_ticks(void) {
  struct tms curr_time;
  times(&curr_time);
  return curr_time.tms_utime;
}

void et_pal_init(void) {
  xt_iss_client_command("all", "enable");
}

#else

#include <time.h>

#include <cstdio>
#include <cstdlib>

#include <executorch/runtime/platform/platform.h>

#define ET_LOG_OUTPUT_FILE stderr

#define NSEC_PER_USEC 1000UL
#define USEC_IN_SEC 1000000UL
#define NSEC_IN_USEC 1000UL
#define NSEC_IN_SEC (NSEC_IN_USEC * USEC_IN_SEC)

et_timestamp_t et_pal_current_ticks(void) {
  struct timespec ts;
  auto ret = clock_gettime(CLOCK_REALTIME, &ts);
  if (ret != 0) {
    fprintf(ET_LOG_OUTPUT_FILE, "Could not get time\n");
    fflush(ET_LOG_OUTPUT_FILE);
    std::abort();
  }

  return ((ts.tv_sec * NSEC_IN_SEC) + (ts.tv_nsec));
}

#endif
