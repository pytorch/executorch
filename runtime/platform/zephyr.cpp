/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 * PAL implementations for Arm Zephyr RTOS.
 */

#include <executorch/runtime/platform/platform.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

void et_pal_init(void) {}

ET_NORETURN void et_pal_abort(void) {
	k_panic();
	// k_panic() should never return, but ensure compiler knows this
	while (1) {
		/* Never reached */
	}
}

et_timestamp_t et_pal_current_ticks(void) {
	return k_uptime_ticks();
}

et_tick_ratio_t et_pal_ticks_to_ns_multiplier(void) {
	return { NSEC_PER_SEC, sys_clock_hw_cycles_per_sec() };
}

void et_pal_emit_log_message(
    ET_UNUSED et_timestamp_t timestamp,
    ET_UNUSED et_pal_log_level_t level,
    ET_UNUSED const char* filename,
    ET_UNUSED const char* function,
    ET_UNUSED size_t line,
    const char* message,
    ET_UNUSED size_t length) {
	printk("%s\n", message);
}

void* et_pal_allocate(size_t size) {
  return k_malloc(size);
}

void et_pal_free(void* ptr) {
  k_free(ptr);
} 
