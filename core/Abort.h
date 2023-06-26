/**
 * @file
 * Executorch global abort wrapper function.
 */

#pragma once

#include <executorch/compiler/Compiler.h>

namespace torch {
namespace executor {

/**
 * Trigger the Executorch global runtime to immediately exit without cleaning
 * up, and set an abnormal exit status (platform-defined).
 */
__ET_NORETURN void runtime_abort();

} // namespace executor
} // namespace torch
