/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <sys/times.h>
#include <xtensa/sim.h>

#include <executorch/runtime/platform/hooks.h>

namespace torch {
namespace executor {

static bool init = false;

uint64_t get_curr_time(void) {
  if (!init) {
    xt_iss_client_command("all", "enable");
    init = true;
  }

  struct tms curr_time;

  times(&curr_time);
  return curr_time.tms_utime;
}

} // namespace executor
} // namespace torch
