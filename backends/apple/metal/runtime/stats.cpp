/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/apple/metal/runtime/stats.h>

#ifdef EXECUTORCH_METAL_COLLECT_STATS

#include <iostream>

namespace executorch {
namespace backends {
namespace metal {

void print_metal_backend_stats() {
  std::cout << "\n--- Metal Backend Performance Statistics ---" << std::endl;

  // Init stats
  double metal_init_total_ms = get_metal_backend_init_total_ms();
  int64_t metal_init_call_count = get_metal_backend_init_call_count();
  std::cout << "Metal init() total: " << metal_init_total_ms << " ms ("
            << metal_init_call_count << " calls)";
  if (metal_init_call_count > 0) {
    std::cout << " (avg: " << metal_init_total_ms / metal_init_call_count
              << " ms/call)";
  }
  std::cout << std::endl;

  // Per-method init breakdown
  auto init_per_method_stats = get_metal_backend_init_per_method_stats();
  if (!init_per_method_stats.empty()) {
    std::cout << "  Per-method init breakdown:" << std::endl;
    for (const auto& entry : init_per_method_stats) {
      const std::string& method_name = entry.first;
      double method_total_ms = entry.second.first;
      int64_t method_call_count = entry.second.second;
      std::cout << "    " << method_name << ": " << method_total_ms << " ms ("
                << method_call_count << " calls)";
      if (method_call_count > 0) {
        std::cout << " (avg: " << method_total_ms / method_call_count
                  << " ms/call)";
      }
      std::cout << std::endl;
    }
  }

  // Execute stats
  double metal_total_ms = get_metal_backend_execute_total_ms();
  int64_t metal_call_count = get_metal_backend_execute_call_count();
  std::cout << "\nMetal execute() total: " << metal_total_ms << " ms ("
            << metal_call_count << " calls)";
  if (metal_call_count > 0) {
    std::cout << " (avg: " << metal_total_ms / metal_call_count << " ms/call)";
  }
  std::cout << std::endl;

  // Per-method execute breakdown
  auto per_method_stats = get_metal_backend_per_method_stats();
  if (!per_method_stats.empty()) {
    std::cout << "  Per-method execute breakdown:" << std::endl;
    for (const auto& entry : per_method_stats) {
      const std::string& method_name = entry.first;
      double method_total_ms = entry.second.first;
      int64_t method_call_count = entry.second.second;
      std::cout << "    " << method_name << ": " << method_total_ms << " ms ("
                << method_call_count << " calls)";
      if (method_call_count > 0) {
        std::cout << " (avg: " << method_total_ms / method_call_count
                  << " ms/call)";
      }
      std::cout << std::endl;
    }
  }

  std::cout << "--------------------------------------------\n" << std::endl;
}

} // namespace metal
} // namespace backends
} // namespace executorch

#endif // EXECUTORCH_METAL_COLLECT_STATS
