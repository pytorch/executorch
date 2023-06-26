/**
 * @file
 * Executorch utility constants.
 */

#pragma once

#include <cstddef>

namespace torch {
namespace executor {

/**
 * Number of bytes in one kibibyte (2^10 bytes).
 */
constexpr size_t kKB = 1024U;

/**
 * Number of bytes in one mebibyte (2^20 bytes).
 */
constexpr size_t kMB = 1024U * kKB;

/**
 * Number of bytes in one gibibyte (2^30 bytes).
 */
constexpr size_t kGB = 1024U * kMB;

/**
 * Number of bytes in one tebibyte (2^40 bytes).
 */
constexpr size_t kTB = 1024U * kGB;

} // namespace executor
} // namespace torch
