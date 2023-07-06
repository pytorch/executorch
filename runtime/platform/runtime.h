/**
 * @file
 * Executorch global runtime wrapper functions.
 */

#pragma once

#include <executorch/runtime/platform/compiler.h>

namespace torch {
namespace executor {

/**
 * Initialize the Executorch global runtime.
 */
void runtime_init();

} // namespace executor
} // namespace torch
