/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ethosu_driver.h>
#include <pmu_ethosu.h>

#include <executorch/backends/arm/runtime/EthosUBackend_Pmu.h>
#include <executorch/backends/arm/runtime/EthosUExecutionStats.h>

#include "ethosu_pmu_test_stubs.h"

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

extern "C" {
void ethosu_inference_begin(ethosu_driver*, void*);
void ethosu_inference_end(ethosu_driver*, void*);
}

int main() {
  EthosUExecutionRecord records[2];
  EthosUStatsBuffer buffer{records, 2};
  CHECK(set_ethosu_stats_buffer(&buffer) == nullptr);
  CHECK(executorch::backends::arm::ethosu_execution_metrics_enabled());

  MockPmu u55_pmu;
  ethosu_device_config u55_config{&u55_pmu};
  ethosu_driver u55{};
  u55.dev.config = &u55_config;
  u55.dev.reg = reinterpret_cast<volatile NPU_REG*>(0x40000000);
  u55.dev.caps = {0, 7, 0};
  u55.job.result = ETHOSU_JOB_RESULT_OK;

  MockPmu u85_pmu;
  ethosu_device_config u85_config{&u85_pmu};
  ethosu_driver u85{};
  u85.dev.config = &u85_config;
  u85.dev.reg = reinterpret_cast<volatile NPU_REG*>(0x50000000);
  u85.dev.caps = {2, 8, 0};
  u85.job.result = ETHOSU_JOB_RESULT_ERROR;

  auto invoke = [&buffer](ethosu_driver* driver, uint64_t handle) {
    EthosUInvocationContext context{};
    context.delegate_handle = handle;
    ethosu_inference_begin(driver, &context);
    CHECK(context.buffer == &buffer);
    CHECK(pmu(driver).enabled);
    pmu(driver).cycles = 100 + driver->dev.caps.product;
    pmu(driver).counts[0] = 42 + driver->dev.caps.product;
    pmu(driver).overflow = 1;
    ethosu_inference_end(driver, &context);
    CHECK(!pmu(driver).enabled);
  };

  std::thread first([&] { invoke(&u55, 10); });
  std::thread second([&] { invoke(&u85, 20); });
  first.join();
  second.join();

  CHECK(buffer.size == 2);
  CHECK(!buffer.overflow);
  CHECK(observer_begins == 2);
  CHECK(observer_ends == 2);
  for (size_t index = 0; index < buffer.size; ++index) {
    const auto& record = records[index];
    CHECK(record.stats.platform == EthosUPlatform::CoreDriver);
    CHECK(record.stats.counter_count == (record.stats.product == 2 ? 7u : 4u));
    CHECK(record.stats.counters[0].overflow);
    CHECK(record.stats.counters[0].value == 42 + record.stats.product);
    if (record.stats.product == 2) {
      CHECK(record.delegate_handle == 20);
      CHECK(record.stats.device_identifier == 0x50000000);
      CHECK(record.stats.counters[5].event_id == ETHOSU_PMU_MAC_ACTIVE);
      CHECK(!record.success);
    } else {
      CHECK(record.delegate_handle == 10);
      CHECK(record.stats.device_identifier == 0x40000000);
      CHECK(record.stats.counters[3].event_id == ETHOSU_PMU_NPU_IDLE);
      CHECK(record.success);
    }
  }

  MockPmu unknown_pmu;
  ethosu_device_config unknown_config{&unknown_pmu};
  ethosu_driver unknown{};
  unknown.dev.config = &unknown_config;
  unknown.dev.caps = {99, 8, 0};
  EthosUInvocationContext reused_context{};
  reused_context.counter_count = 32;
  ethosu_inference_begin(&unknown, &reused_context);
  CHECK(reused_context.buffer == &buffer);
  CHECK(reused_context.counter_count == 0);
  CHECK(!unknown_pmu.enabled);
  ethosu_inference_end(&unknown, &reused_context);
  CHECK(buffer.size == 2);
  CHECK(observer_begins == 3);
  CHECK(observer_ends == 3);

  CHECK(set_ethosu_stats_buffer(nullptr) == &buffer);
  std::puts("Ethos-U PMU callbacks: PASS");
}
