/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/arm/runtime/EthosUExecutionStats.h>

#include <cstdio>
#include <cstdlib>
#include <thread>

#define CHECK(expr)                                                           \
  do {                                                                        \
    if (!(expr)) {                                                            \
      std::fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #expr); \
      std::abort();                                                           \
    }                                                                         \
  } while (false)

using namespace executorch::backends::arm;

int main() {
  EthosUExecutionRecord records[2];
  EthosUStatsBuffer buffer{records, 2};
  EthosUExecutionRecord u55{};
  u55.delegate_handle = 10;
  u55.success = true;
  u55.stats.device_identifier = 0x40000000;
  u55.stats.product = 0;
  u55.stats.cycle_count = 100;
  u55.stats.cycle_count_valid = true;
  u55.stats.counter_count = 4;
  u55.stats.counters[0] = {17, 42, false};

  buffer.append(u55);
  CHECK(buffer.size == 1);

  EthosUExecutionRecord u85{};
  u85.delegate_handle = 20;
  u85.stats.device_identifier = 0x50000000;
  u85.stats.product = 2;
  u85.stats.counter_count = 8;
  u85.stats.counters[7] = {22, 99, false};

  buffer.append(u85);
  buffer.append(u55);
  CHECK(buffer.size == 2 && buffer.overflow);
  CHECK(records[0].stats.device_identifier == 0x40000000);
  CHECK(records[0].stats.counter_count == 4);
  CHECK(records[0].stats.counters[0].event_id == 17);
  CHECK(records[1].stats.device_identifier == 0x50000000);
  CHECK(records[1].stats.platform == EthosUPlatform::CoreDriver);
  CHECK(records[1].stats.counter_count == 8);
  CHECK(records[1].stats.counters[7].value == 99);
  CHECK(!records[1].success);
  CHECK(buffer.reset());
  CHECK(buffer.size == 0 && !buffer.overflow);
  buffer.append(u55);
  CHECK(buffer.size == 1);
  CHECK(!buffer.reset());

  constexpr size_t kThreads = 4;
  constexpr size_t kRecordsPerThread = 8;
  EthosUExecutionRecord concurrent_records[kThreads * kRecordsPerThread];
  EthosUStatsBuffer concurrent_buffer{
      concurrent_records, kThreads * kRecordsPerThread};
  std::thread publishers[kThreads];
  for (size_t thread = 0; thread < kThreads; ++thread) {
    publishers[thread] = std::thread([&, thread] {
      for (size_t index = 0; index < kRecordsPerThread; ++index) {
        auto record = u55;
        record.delegate_handle = thread * kRecordsPerThread + index;
        concurrent_buffer.append(record);
      }
    });
  }
  for (auto& publisher : publishers) {
    publisher.join();
  }
  CHECK(concurrent_buffer.size == kThreads * kRecordsPerThread);
  CHECK(!concurrent_buffer.overflow);
  bool seen[kThreads * kRecordsPerThread] = {};
  for (size_t index = 0; index < concurrent_buffer.size; ++index) {
    const auto handle = concurrent_records[index].delegate_handle;
    CHECK(handle < kThreads * kRecordsPerThread);
    CHECK(!seen[handle]);
    seen[handle] = true;
  }
  for (bool found : seen) {
    CHECK(found);
  }
  std::thread overflow_publishers[kThreads];
  for (auto& publisher : overflow_publishers) {
    publisher = std::thread([&] { concurrent_buffer.append(u85); });
  }
  for (auto& publisher : overflow_publishers) {
    publisher.join();
  }
  CHECK(concurrent_buffer.size == kThreads * kRecordsPerThread);
  CHECK(concurrent_buffer.reset());
  std::puts("Ethos-U execution stats: PASS");
}
