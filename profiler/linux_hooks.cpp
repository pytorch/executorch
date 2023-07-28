/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdint.h>
#include <time.h>
#include <cstdio>
#include <cstdlib>

#include <executorch/runtime/platform/hooks.h>

namespace torch {
namespace executor {

#define ET_LOG_OUTPUT_FILE stderr

#define NSEC_PER_USEC 1000UL
#define USEC_IN_SEC 1000000UL
#define NSEC_IN_USEC 1000UL
#define NSEC_IN_SEC (NSEC_IN_USEC * USEC_IN_SEC)

uint64_t get_curr_time(void) {
  struct timespec ts;
  auto ret = clock_gettime(CLOCK_REALTIME, &ts);
  if (ret != 0) {
    fprintf(ET_LOG_OUTPUT_FILE, "Could not get time\n");
    fflush(ET_LOG_OUTPUT_FILE);
    std::abort();
  }

  return ((ts.tv_sec * NSEC_IN_SEC) + (ts.tv_nsec));
}

} // namespace executor
} // namespace torch
