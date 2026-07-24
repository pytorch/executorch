/*
 * Copyright (c) 2026 iote.ai
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Nordic AXON NPU delegate — public profiling API.
 */
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Aggregate cycle counters across all AXON delegate handles. */
extern uint64_t axon_delegate_total_cycles;
extern uint32_t axon_delegate_total_calls;

/* Reset all per-handle and global cycle counters. */
void axon_delegate_reset_profile(void);

/* Dump per-handle cycle counts to the ExecuTorch log. */
void axon_delegate_dump_profile(void);

#ifdef __cplusplus
}
#endif
