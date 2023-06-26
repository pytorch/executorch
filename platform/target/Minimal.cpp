/**
 * @file
 * Fallback PAL implementations that do not depend on any assumptions about
 * capabililties of the system.
 */

#include <executorch/platform/Platform.h>

#include <executorch/compiler/Compiler.h>

void et_pal_init(void) {}

__ET_NORETURN void et_pal_abort(void) {
  __builtin_trap();
}

et_timestamp_t et_pal_current_ticks(void) {
  // Return a number that should be easier to search for than 0.
  return 11223344;
}

void et_pal_emit_log_message(
    __ET_UNUSED et_timestamp_t timestamp,
    __ET_UNUSED et_pal_log_level_t level,
    __ET_UNUSED const char* filename,
    __ET_UNUSED const char* function,
    __ET_UNUSED size_t line,
    __ET_UNUSED const char* message,
    __ET_UNUSED size_t length) {}
