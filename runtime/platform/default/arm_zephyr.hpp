#ifndef _ARM_ZEPHYR_PAL_HPP
#define _ARM_ZEPHYR_PAL_HPP

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

void et_pal_init(void) {}

ET_NORETURN void et_pal_abort(void) {
    _exit(-1);
}

et_timestamp_t et_pal_current_ticks(void) {
    return k_uptime_ticks();
}

et_tick_ratio_t et_pal_ticks_to_ns_multiplier(void) {
  // Since we don't know the CPU freq for your target and just cycles in the
  // FVP for et_pal_current_ticks() we return a conversion ratio of 1
  return {1, 1};
}

/**
 * Emit a log message via platform output (serial port, console, etc).
 */
void et_pal_emit_log_message(
    ET_UNUSED et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    ET_UNUSED const char* function,
    size_t line,
    const char* message,
    ET_UNUSED size_t length) {
  fprintf(
      stderr,
      "%c [executorch:%s:%zu %s()] %s\n",
      level,
      filename,
      line,
      function,
      message);
}

void* et_pal_allocate(ET_UNUSED size_t size) {
    return k_malloc(size);
}

void et_pal_free(ET_UNUSED void* ptr) {
    k_free(ptr);
} 

#endif // _ARM_ZEPHYR_PAL_HPP
