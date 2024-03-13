/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdint.h>

namespace torch {
namespace executor {

// The stubs defined in this file are expected to be implemented/provided on
// a per platform basis. e.g. we'll have one for Linux running on x86 and
// another one maybe for a system running a RTOS on an ARM SoC.

// This is expected to return a 64 bit value that contains the most
// granular time representation available on the system. It could be
// ticks, cycle count or time in microseconds etc.
// TODO(T157580075): delete this file and merge functionality into Platform.hå
uint64_t get_curr_time(void);

} // namespace executor
} // namespace torch
