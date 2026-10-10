/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/arm/runtime/VGFExecutionStats.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#define CHECK(expr)                                                           \
  do {                                                                        \
    if (!(expr)) {                                                            \
      std::fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #expr); \
      std::abort();                                                           \
    }                                                                         \
  } while (false)

#if defined(EXECUTORCH_VGF_IO_STATS) && EXECUTORCH_VGF_IO_STATS
using namespace executorch::backends::vgf;

void execute_for_test(size_t in_bytes, size_t out_bytes, bool success = true) {
  VGF_STATS_EXECUTION(nullptr);
  char input[128] = {};
  char output[128] = {};
  VGF_STATS_MEMCPY_IN(output, input, in_bytes);
  VGF_STATS_MEMCPY_IN(output, input, 7);
  {
    // Two timers in the same scope test expansion of __LINE__.
    VGF_STATS_TIME(submit_wait_ns);
    VGF_STATS_TIME(binding_refresh_ns);
  }
  VGF_STATS_MEMCPY_OUT(output, input, out_bytes);
  if (success)
    VGF_STATS_SUCCESS();
}

int main() {
  VgfExecutionRecord records[8];
  VgfStatsBuffer buffer{records, 8, 0, false};
  CHECK(set_vgf_stats_buffer(&buffer) == nullptr);
  execute_for_test(11, 23);
  execute_for_test(0, 5);
  CHECK(buffer.size == 2);
  CHECK(records[0].stats.in_copy_bytes == 18);
  CHECK(records[0].stats.out_copy_bytes == 23);
  CHECK(records[1].stats.in_copy_bytes == 7);
  CHECK(records[1].stats.out_copy_bytes == 5);
  CHECK(records[0].success && records[1].success);
  CHECK(current_vgf_execution_stats() == nullptr);
  // No nonzero timer assertion: a valid monotonic clock can have coarse ticks.
  CHECK(records[0].stats.execute_ns >= records[0].stats.in_memcpy_ns);
  execute_for_test(1, 2, false);
  CHECK(!records[2].success);
  records[0].stats.imports = 3;
  records[0].stats.fallback_count = 4;
  records[0].stats.reset();
  CHECK(records[0].stats.in_copy_bytes == 0);
  CHECK(records[0].stats.imports == 0);
  CHECK(records[0].stats.fallback_count == 0);
  std::thread other([] { execute_for_test(100, 101); });
  other.join();
  CHECK(buffer.size == 3); // Capture does not cross threads.
  for (int i = 0; i < 6; ++i)
    execute_for_test(1, 1);
  CHECK(buffer.size == 8 && buffer.overflow);
  CHECK(set_vgf_stats_buffer(nullptr) == &buffer);
  execute_for_test(1, 1); // No sink is also valid.
  std::puts("VGF stats enabled: PASS");
}
#else
int main() {
  char src[] = "unchanged", dst[sizeof(src)] = {};
  int side_effect = 0;
  VGF_STATS_EXECUTION(++side_effect);
  VGF_STATS_TIME(no_such_field);
  VGF_STATS_SUCCESS();
  VGF_STATS_MEMCPY_IN(dst, src, sizeof(src));
  CHECK(std::memcmp(dst, src, sizeof(src)) == 0);
  VGF_STATS_MEMCPY_OUT(dst, src, sizeof(src));
  CHECK(side_effect == 0);
  std::puts("VGF stats disabled: PASS");
}
#endif
