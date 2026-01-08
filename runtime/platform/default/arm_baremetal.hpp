#ifndef __ARM_BARE_METAL_HPP__
#define __ARM_BARE_METAL_HPP__

void et_pal_init(void) {
  // Enable ARM PMU Clock
  ARM_PMU_Enable();
  DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk; // Trace enable
  ARM_PMU_CYCCNT_Reset();
  ARM_PMU_CNTR_Enable(PMU_CNTENSET_CCNTR_ENABLE_Msk);
}

/**
 * Implementation of the et_pal_<funcs>()
 *
 * This functions are hardware adaption type of functions for things like
 * time/logging/memory allocation that could call your RTOS or need to to
 * be implemnted in some way.
 */

ET_NORETURN void et_pal_abort(void) {
#if !defined(SEMIHOSTING)
  __builtin_trap();
#else
  _exit(-1);
#endif
}

et_timestamp_t et_pal_current_ticks(void) {
  return ARM_PMU_Get_CCNTR();
}

et_tick_ratio_t et_pal_ticks_to_ns_multiplier(void) {
  // Since we don't know the CPU freq for your target and justs cycles in the
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

/**
 * Dynamic memory allocators intended to be used by temp_allocator
 * to implement malloc()/free() type of allocations.
 * Currenyly not used.
 */

void* et_pal_allocate(ET_UNUSED size_t size) {
  return nullptr;
}

void et_pal_free(ET_UNUSED void* ptr) {}

#endif // __ARM_BARE_METAL_HPP__
