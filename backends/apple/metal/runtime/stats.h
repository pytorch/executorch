/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>

#include <executorch/runtime/platform/log.h>

namespace executorch {
namespace backends {
namespace metal {

// =======================
// Metal backend timing statistics
// =======================

#ifdef EXECUTORCH_METAL_COLLECT_STATS

// Execute timing
double get_metal_backend_execute_total_ms();
int64_t get_metal_backend_execute_call_count();
// Returns map of method_name -> (total_ms, call_count)
std::unordered_map<std::string, std::pair<double, int64_t>>
get_metal_backend_per_method_stats();

// Init timing
double get_metal_backend_init_total_ms();
int64_t get_metal_backend_init_call_count();
// Returns map of method_name -> (total_ms, call_count) for init
std::unordered_map<std::string, std::pair<double, int64_t>>
get_metal_backend_init_per_method_stats();

// Reset all timing stats
void reset_metal_backend_stats();

// Print all timing stats to stdout
void print_metal_backend_stats();

#else // !EXECUTORCH_METAL_COLLECT_STATS

// No-op stubs when stats collection is disabled
inline double get_metal_backend_execute_total_ms() {
  return 0.0;
}
inline int64_t get_metal_backend_execute_call_count() {
  return 0;
}
inline std::unordered_map<std::string, std::pair<double, int64_t>>
get_metal_backend_per_method_stats() {
  return {};
}
inline double get_metal_backend_init_total_ms() {
  return 0.0;
}
inline int64_t get_metal_backend_init_call_count() {
  return 0;
}
inline std::unordered_map<std::string, std::pair<double, int64_t>>
get_metal_backend_init_per_method_stats() {
  return {};
}
inline void reset_metal_backend_stats() {}
inline void print_metal_backend_stats() {
  ET_LOG(
      Info,
      "Metal backend stats collection is disabled. "
      "Set EXECUTORCH_METAL_COLLECT_STATS=ON to collect stats.");
}

#endif // EXECUTORCH_METAL_COLLECT_STATS

} // namespace metal
} // namespace backends
} // namespace executorch
