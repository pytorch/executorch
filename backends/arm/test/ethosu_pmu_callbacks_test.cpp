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

#include <array>
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

using namespace executorch::backends::arm;
using namespace executorch::runtime;

class CapturingTracer final : public EventTracer {
 public:
  uint8_t metadata[kEthosUCycleMetadataMaxSize]{};
  size_t metadata_size = 0;
  bool started = false;
  bool ended = false;

  void create_event_block(const char*) override {}
  EventTracerEntry start_profiling(const char*, ChainID, DebugHandle) override {
    return {};
  }
  EventTracerEntry start_profiling_delegate(
      const char* name,
      DelegateDebugIntId index) override {
    CHECK(std::strcmp(name, kEthosUCycleEventName) == 0);
    CHECK(index == kUnsetDelegateDebugIntId);
    started = true;
    EventTracerEntry entry{};
    entry.event_id = 17;
    return entry;
  }
  void end_profiling_delegate(
      EventTracerEntry entry,
      const void* data,
      size_t size) override {
    CHECK(started);
    CHECK(entry.event_id == 17);
    CHECK(size <= sizeof(metadata));
    std::memcpy(metadata, data, size);
    metadata_size = size;
    ended = true;
  }
  void log_profiling_delegate(
      const char*,
      DelegateDebugIntId,
      et_timestamp_t,
      et_timestamp_t,
      const void*,
      size_t) override {}
  void end_profiling(EventTracerEntry) override {}
  void track_allocation(AllocatorID, size_t) override {}
  AllocatorID track_allocator(const char*) override {
    return 0;
  }
  Result<bool> log_evalue(const EValue&, LoggedEValueType) override {
    return true;
  }
  Result<bool> log_intermediate_output_delegate(
      const char*,
      DelegateDebugIntId,
      const executorch::aten::Tensor&) override {
    return true;
  }
  Result<bool> log_intermediate_output_delegate(
      const char*,
      DelegateDebugIntId,
      const ArrayRef<executorch::aten::Tensor>) override {
    return true;
  }
  Result<bool> log_intermediate_output_delegate(
      const char*,
      DelegateDebugIntId,
      const int&) override {
    return true;
  }
  Result<bool> log_intermediate_output_delegate(
      const char*,
      DelegateDebugIntId,
      const bool&) override {
    return true;
  }
  Result<bool> log_intermediate_output_delegate(
      const char*,
      DelegateDebugIntId,
      const double&) override {
    return true;
  }
  void set_delegation_intermediate_output_filter(
      EventTracerFilterBase*) override {}
};

uint64_t read_le(const uint8_t* bytes, size_t size) {
  uint64_t value = 0;
  for (size_t index = 0; index < size; ++index) {
    value |= static_cast<uint64_t>(bytes[index]) << (8 * index);
  }
  return value;
}

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

  CapturingTracer tracer;
  EthosUInvocationContext traced_invocation{};
  traced_invocation.event_tracer = &tracer;
  ethosu_inference_begin(&u85, &traced_invocation);
  CHECK(pmu(&u85).enabled);
  pmu(&u85).cycles = 123456;
  pmu(&u85).counts[0] = 11;
  pmu(&u85).counts[5] = 67;
  pmu(&u85).overflow = ETHOSU_PMU_CCNT_Msk | (1u << 5);
  ethosu_inference_end(&u85, &traced_invocation);
  CHECK(!pmu(&u85).enabled);
  CHECK(tracer.started && tracer.ended);
  CHECK(
      tracer.metadata_size ==
      kEthosUCycleMetadataHeaderSize + 7 * kEthosUCycleCounterMetadataSize);
  CHECK(std::memcmp(tracer.metadata, "EUC1", 4) == 0);
  CHECK(read_le(tracer.metadata + 4, 4) == 2);
  CHECK(read_le(tracer.metadata + 8, 4) == 256);
  CHECK(
      read_le(tracer.metadata + 12, 4) ==
      static_cast<uint32_t>(EthosUPlatform::CoreDriver));
  CHECK(read_le(tracer.metadata + 16, 4) == 1);
  CHECK(read_le(tracer.metadata + 20, 8) == 0x50000000);
  CHECK(read_le(tracer.metadata + 28, 8) == 123456);
  CHECK(read_le(tracer.metadata + 36, 4) == 7);
  for (size_t slot = 0; slot < 7; ++slot) {
    const size_t offset =
        kEthosUCycleMetadataHeaderSize + slot * kEthosUCycleCounterMetadataSize;
    CHECK(read_le(tracer.metadata + offset, 4) == pmu(&u85).events[slot]);
    CHECK(read_le(tracer.metadata + offset + 4, 8) == pmu(&u85).counts[slot]);
    CHECK(read_le(tracer.metadata + offset + 12, 4) == (slot == 5));
  }

  CapturingTracer other_tracer;
  EthosUInvocationContext other_invocation{};
  other_invocation.event_tracer = &other_tracer;
  ethosu_inference_begin(&u55, &other_invocation);
  pmu(&u55).cycles = 45;
  pmu(&u55).counts[3] = 9;
  pmu(&u55).overflow = 1u << 3;
  ethosu_inference_end(&u55, &other_invocation);
  CHECK(other_tracer.started && other_tracer.ended);
  CHECK(
      other_tracer.metadata_size ==
      kEthosUCycleMetadataHeaderSize + 4 * kEthosUCycleCounterMetadataSize);
  CHECK(read_le(other_tracer.metadata + 8, 4) == 128);
  CHECK(
      read_le(other_tracer.metadata + 12, 4) ==
      static_cast<uint32_t>(EthosUPlatform::CoreDriver));
  CHECK(read_le(other_tracer.metadata + 16, 4) == 2);
  CHECK(read_le(other_tracer.metadata + 20, 8) == 0x40000000);
  CHECK(read_le(other_tracer.metadata + 28, 8) == 45);
  CHECK(read_le(other_tracer.metadata + 36, 4) == 4);
  for (size_t slot = 0; slot < 4; ++slot) {
    const size_t offset =
        kEthosUCycleMetadataHeaderSize + slot * kEthosUCycleCounterMetadataSize;
    CHECK(read_le(other_tracer.metadata + offset, 4) == pmu(&u55).events[slot]);
    CHECK(
        read_le(other_tracer.metadata + offset + 4, 8) ==
        pmu(&u55).counts[slot]);
    CHECK(read_le(other_tracer.metadata + offset + 12, 4) == (slot == 3));
  }
  CHECK(buffer.size == 2);

  CHECK(pmu(&u85).events[7] == ETHOSU_PMU_NO_EVENT);
  CHECK((pmu(&u85).enabled_mask & 0xff) == 0x7f);

  ethosu_pmu_event_type u55_events[] = {ETHOSU_PMU_MAC_ACTIVE};
  CHECK(set_ethosu55_65_pmu_events(u55_events) == Error::Ok);
  u55_events[0] = ETHOSU_PMU_NPU_IDLE;
  std::array<ethosu_pmu_event_type, 9> too_many_events{};
  std::array<ethosu_pmu_event_type, 5> too_many_u55_events{};
  CHECK(set_ethosu85_pmu_events(too_many_events) == Error::InvalidArgument);
  CHECK(
      set_ethosu55_65_pmu_events(too_many_u55_events) ==
      Error::InvalidArgument);

  buffer.reset();
  CHECK(set_ethosu_stats_buffer(&buffer) == nullptr);
  EthosUInvocationContext u55_invocation{};
  ethosu_inference_begin(&u55, &u55_invocation);
  CHECK((pmu(&u55).enabled_mask & 0xf) == 1);
  CHECK(pmu(&u55).events[0] == ETHOSU_PMU_MAC_ACTIVE);
  CHECK(pmu(&u55).events[3] == ETHOSU_PMU_NO_EVENT);
  pmu(&u55).counts[0] = 91;
  CHECK(set_ethosu55_65_pmu_events({}) == Error::Ok);
  ethosu_inference_end(&u55, &u55_invocation);
  CHECK(buffer.size == 1);
  CHECK(records[0].stats.counter_count == 1);
  CHECK(records[0].stats.counters[0].event_id == ETHOSU_PMU_MAC_ACTIVE);
  CHECK(records[0].stats.counters[0].value == 91);
  CHECK(set_ethosu_stats_buffer(nullptr) == &buffer);

  ethosu_pmu_event_type u55_configured_events[] = {
      ETHOSU_PMU_NPU_IDLE,
      ETHOSU_PMU_NO_EVENT,
      ETHOSU_PMU_MAC_ACTIVE,
      ETHOSU_PMU_WD_ACTIVE,
  };
  ethosu_pmu_event_type u85_configured_events[] = {
      ETHOSU_PMU_NPU_IDLE,
      ETHOSU_PMU_NO_EVENT,
      ETHOSU_PMU_MAC_ACTIVE,
      ETHOSU_PMU_WD_ACTIVE,
      ETHOSU_PMU_NPU_ACTIVE,
  };
  CHECK(set_ethosu55_65_pmu_events(u55_configured_events) == Error::Ok);
  CHECK(set_ethosu85_pmu_events(u85_configured_events) == Error::Ok);
  u55_configured_events[0] = ETHOSU_PMU_NO_EVENT;
  u85_configured_events[0] = ETHOSU_PMU_NO_EVENT;
  CHECK(set_ethosu85_pmu_events(too_many_events) == Error::InvalidArgument);
  buffer.reset();
  CHECK(set_ethosu_stats_buffer(&buffer) == nullptr);
  EthosUInvocationContext configured_u55{};
  ethosu_inference_begin(&u55, &configured_u55);
  CHECK((pmu(&u55).enabled_mask & 0xf) == 0xd);
  CHECK(pmu(&u55).events[0] == ETHOSU_PMU_NPU_IDLE);
  CHECK(pmu(&u55).events[1] == ETHOSU_PMU_NO_EVENT);
  CHECK(pmu(&u55).events[3] == ETHOSU_PMU_WD_ACTIVE);
  ethosu_inference_end(&u55, &configured_u55);
  CapturingTracer sparse_tracer;
  EthosUInvocationContext configured_u85{};
  configured_u85.event_tracer = &sparse_tracer;
  ethosu_inference_begin(&u85, &configured_u85);
  CHECK((pmu(&u85).enabled_mask & 0xff) == 0x1d);
  CHECK(pmu(&u85).events[0] == ETHOSU_PMU_NPU_IDLE);
  CHECK(pmu(&u85).events[4] == ETHOSU_PMU_NPU_ACTIVE);
  CHECK(pmu(&u85).events[7] == ETHOSU_PMU_NO_EVENT);
  pmu(&u85).counts[2] = 31;
  pmu(&u85).counts[4] = 59;
  ethosu_inference_end(&u85, &configured_u85);
  CHECK(buffer.size == 2);
  CHECK(records[0].stats.counter_count == 3);
  CHECK(records[0].stats.counters[1].event_id == ETHOSU_PMU_MAC_ACTIVE);
  CHECK(records[1].stats.counter_count == 4);
  CHECK(records[1].stats.counters[3].event_id == ETHOSU_PMU_NPU_ACTIVE);
  CHECK(records[1].stats.counters[1].value == 31);
  CHECK(records[1].stats.counters[3].value == 59);
  CHECK(sparse_tracer.started && sparse_tracer.ended);
  CHECK(
      sparse_tracer.metadata_size ==
      kEthosUCycleMetadataHeaderSize + 4 * kEthosUCycleCounterMetadataSize);
  CHECK(read_le(sparse_tracer.metadata + 36, 4) == 4);
  CHECK(
      read_le(
          sparse_tracer.metadata + kEthosUCycleMetadataHeaderSize +
              kEthosUCycleCounterMetadataSize,
          4) == ETHOSU_PMU_MAC_ACTIVE);
  CHECK(
      read_le(
          sparse_tracer.metadata + kEthosUCycleMetadataHeaderSize +
              3 * kEthosUCycleCounterMetadataSize + 4,
          8) == 59);
  CHECK(set_ethosu_stats_buffer(nullptr) == &buffer);

  CHECK(
      set_ethosu85_pmu_events(
          std::array{ETHOSU_PMU_MAC_ACTIVE, ETHOSU_PMU_NO_EVENT}) == Error::Ok);
  buffer.reset();
  CHECK(set_ethosu_stats_buffer(&buffer) == nullptr);
  EthosUInvocationContext short_u85{};
  ethosu_inference_begin(&u85, &short_u85);
  CHECK((pmu(&u85).enabled_mask & 0xff) == 1);
  CHECK(pmu(&u85).events[0] == ETHOSU_PMU_MAC_ACTIVE);
  CHECK(pmu(&u85).events[7] == ETHOSU_PMU_NO_EVENT);
  ethosu_inference_end(&u85, &short_u85);
  CHECK(buffer.size == 1);
  CHECK(records[0].stats.counter_count == 1);
  CHECK(set_ethosu_stats_buffer(nullptr) == &buffer);

  CHECK(set_ethosu85_pmu_events({}) == Error::Ok);
  CHECK(set_ethosu55_65_pmu_events({}) == Error::Ok);
  buffer.reset();
  CHECK(set_ethosu_stats_buffer(&buffer) == nullptr);
  EthosUInvocationContext cycles_only{};
  ethosu_inference_begin(&u55, &cycles_only);
  CHECK((pmu(&u55).enabled_mask & 0xf) == 0);
  pmu(&u55).cycles = 77;
  ethosu_inference_end(&u55, &cycles_only);
  CHECK(buffer.size == 1);
  CHECK(records[0].stats.counter_count == 0);
  CHECK(records[0].stats.cycle_count == 77);
  CHECK(set_ethosu_stats_buffer(nullptr) == &buffer);
  std::puts("Ethos-U PMU callbacks: PASS");
}
