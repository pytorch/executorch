/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ethosu.hpp>
#include <executorch/backends/arm/runtime/EthosUBackend_Internal.h>
#include <executorch/backends/arm/runtime/EthosUBackend_Pmu_LinuxDriver.h>
#include <executorch/backends/arm/runtime/EthosUExecutionStats_Internal.h>
#include <executorch/runtime/core/event_tracer.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>
#include <cstdlib>
#include <new>
#include <vector>

namespace {
thread_local bool fail_next_allocation = false;
}

void* operator new(std::size_t size) {
  if (fail_next_allocation) {
    fail_next_allocation = false;
    throw std::bad_alloc();
  }
  if (void* memory = std::malloc(size == 0 ? 1 : size)) {
    return memory;
  }
  throw std::bad_alloc();
}

void operator delete(void* memory) noexcept {
  std::free(memory);
}

void operator delete(void* memory, std::size_t) noexcept {
  std::free(memory);
}

using namespace executorch::backends::arm;
using namespace executorch::runtime;

namespace {
class CapturingTracer final : public executorch::runtime::EventTracer {
 public:
  void create_event_block(const char*) override {}
  EventTracerEntry start_profiling(const char*, ChainID, DebugHandle) override {
    return {};
  }
  int starts = 0;
  int ends = 0;
  std::vector<uint8_t> metadata;
  executorch::runtime::EventTracerEntry start_profiling_delegate(
      const char*,
      DelegateDebugIntId) override {
    ++starts;
    return {};
  }
  void end_profiling_delegate(
      executorch::runtime::EventTracerEntry,
      const void* data,
      size_t size) override {
    ++ends;
    const auto* bytes = static_cast<const uint8_t*>(data);
    if (size != 0) {
      metadata.assign(bytes, bytes + size);
    }
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

class LinuxExecutionStatsTest : public ::testing::Test {
 protected:
  EthosUExecutionRecord records[4]{};
  EthosUStatsBuffer buffer{records, 4};
  CapturingTracer tracer;
  EthosUInvocationContext invocation{};

  void SetUp() override {
    runtime_init();
    set_ethosu_stats_buffer(&buffer);
    EthosU::Device::caps = {};
    invocation.device_path = "/dev/ethosu0";
    invocation.delegate_handle = 123;
    invocation.event_tracer = &tracer;
  }
  void TearDown() override {
    EthosU::Device::fail_capabilities = false;
    EthosU::Inference::throw_on_submit = false;
    EthosU::Inference::throw_on_wait = false;
    EthosU::Inference::timeout = false;
    EthosU::Inference::result = EthosU::InferenceStatus::OK;
    set_ethosu_stats_buffer(nullptr);
    const ethosu_pmu_event_type u65[] = {
        ETHOSU_PMU_AXI0_RD_DATA_BEAT_RECEIVED,
        ETHOSU_PMU_AXI1_RD_DATA_BEAT_RECEIVED,
        ETHOSU_PMU_AXI0_WR_DATA_BEAT_WRITTEN,
        ETHOSU_PMU_NPU_IDLE};
    const ethosu_pmu_event_type u85[] = {
        ETHOSU_PMU_SRAM_RD_DATA_BEAT_RECEIVED,
        ETHOSU_PMU_SRAM_WR_DATA_BEAT_WRITTEN,
        ETHOSU_PMU_EXT_RD_DATA_BEAT_RECEIVED,
        ETHOSU_PMU_EXT_WR_DATA_BEAT_WRITTEN,
        ETHOSU_PMU_NPU_IDLE,
        ETHOSU_PMU_MAC_ACTIVE,
        ETHOSU_PMU_WD_ACTIVE};
    set_ethosu55_65_pmu_events(u65);
    set_ethosu85_pmu_events(u85);
  }
};

// Cppcheck cannot expand GoogleTest registration macros correctly.
// cppcheck-suppress-begin syntaxError
TEST_F(LinuxExecutionStatsTest, CapturesSymbolicEventsAndOwnedPath) {
  const ethosu_pmu_event_type events[] = {ETHOSU_PMU_NPU_IDLE};
  ASSERT_EQ(set_ethosu55_65_pmu_events(events), Error::Ok);
  std::string path = "/dev/ethosu0";
  invocation.device_path = path.c_str();
  ASSERT_EQ(ethosu_linux_inference_begin(invocation), Error::Ok);
  EXPECT_TRUE(invocation.pmu_enabled);
  EXPECT_EQ(invocation.pmu_events[0], 2);
  EXPECT_EQ(invocation.pmu_events[1], 0);
  EthosU::Inference inference;
  inference.counters[0] = 42;
  ethosu_linux_inference_end(&inference, invocation, true);
  path[5] = 'X';
  ASSERT_EQ(buffer.size, 1);
  const auto& record = records[0];
  EXPECT_EQ(record.delegate_handle, 123);
  EXPECT_TRUE(record.success);
  EXPECT_EQ(record.stats.device_identifier, "/dev/ethosu0");
  EXPECT_EQ(record.stats.platform, EthosUPlatform::LinuxDriver);
  EXPECT_EQ(record.stats.product, 1);
  EXPECT_EQ(record.stats.macs_per_cycle, 256);
  EXPECT_TRUE(record.stats.cycle_count_valid);
  EXPECT_EQ(record.stats.cycle_count, 0);
  EXPECT_FALSE(record.stats.cycle_overflow);
  ASSERT_EQ(record.stats.counter_count, 1);
  EXPECT_EQ(record.stats.counters[0].event_id, ETHOSU_PMU_NPU_IDLE);
  EXPECT_EQ(record.stats.counters[0].value, 42);
  EXPECT_FALSE(record.stats.counters[0].overflow);
  ASSERT_EQ(tracer.metadata.size(), 40 + 16 + 4 + 12);
  EXPECT_EQ(tracer.metadata[12], 1);
  EXPECT_EQ(tracer.metadata[20], 0);
  EXPECT_EQ(tracer.metadata[56], 12);
  EXPECT_EQ(
      std::string(tracer.metadata.begin() + 60, tracer.metadata.end()),
      "/dev/ethosu0");
  EXPECT_EQ(tracer.starts, tracer.ends);
}

TEST_F(LinuxExecutionStatsTest, UsesProductSpecificMappingAndSlotLimit) {
  EthosU::Device::caps.hwId.product.major = 7;
  ASSERT_EQ(ethosu_linux_inference_begin(invocation), Error::Ok);
  EXPECT_TRUE(invocation.pmu_enabled);
  EXPECT_NE(invocation.pmu_events[0], ETHOSU_PMU_SRAM_RD_DATA_BEAT_RECEIVED);
  ethosu_linux_inference_end(nullptr, invocation, false);
  EthosU::Device::caps.hwCfg.type =
      EthosU::HardwareConfiguration::DeviceType::SUBSYSTEM;
  EXPECT_EQ(ethosu_linux_inference_begin(invocation), Error::InvalidArgument);
  const ethosu_pmu_event_type events[] = {ETHOSU_PMU_NPU_IDLE};
  ASSERT_EQ(set_ethosu85_pmu_events(events), Error::Ok);
  EXPECT_EQ(ethosu_linux_inference_begin(invocation), Error::Ok);
  EXPECT_EQ(invocation.pmu_events[0], 2);
  ethosu_linux_inference_end(nullptr, invocation, false);
}

TEST_F(LinuxExecutionStatsTest, KeepsSubmittedEventsAfterReconfiguration) {
  const ethosu_pmu_event_type events[] = {ETHOSU_PMU_NPU_IDLE};
  ASSERT_EQ(set_ethosu55_65_pmu_events(events), Error::Ok);
  ASSERT_EQ(ethosu_linux_inference_begin(invocation), Error::Ok);
  ASSERT_EQ(set_ethosu55_65_pmu_events({}), Error::Ok);
  EthosU::Inference inference;
  inference.counters[0] = 42;
  ethosu_linux_inference_end(&inference, invocation, true);
  ASSERT_EQ(buffer.size, 1);
  ASSERT_EQ(records[0].stats.counter_count, 1);
  EXPECT_EQ(records[0].stats.counters[0].event_id, ETHOSU_PMU_NPU_IDLE);
  EXPECT_EQ(records[0].stats.counters[0].value, 42);
}

TEST_F(LinuxExecutionStatsTest, HandlesCapabilityReadFailureAtEnd) {
  ASSERT_EQ(ethosu_linux_inference_begin(invocation), Error::Ok);
  EthosU::Inference inference;
  EthosU::Device::fail_capabilities = true;
  ethosu_linux_inference_end(&inference, invocation, true);
  EXPECT_EQ(buffer.size, 0);
  EXPECT_EQ(tracer.starts, tracer.ends);
}

TEST_F(LinuxExecutionStatsTest, RejectsUnsupportedEvents) {
  const ethosu_pmu_event_type events[] = {
      ETHOSU_PMU_SRAM_RD_DATA_BEAT_RECEIVED};
  EXPECT_EQ(set_ethosu55_65_pmu_events(events), Error::InvalidArgument);
}

TEST_F(LinuxExecutionStatsTest, BalancesFailedOrUnavailableMeasurements) {
  EthosU::Inference inference;
  for (int failure = 0; failure < 3; ++failure) {
    ASSERT_EQ(ethosu_linux_inference_begin(invocation), Error::Ok);
    inference.fail_read = failure == 2;
    ethosu_linux_inference_end(
        failure == 0 ? nullptr : &inference, invocation, failure == 2);
  }
  EXPECT_EQ(buffer.size, 0);
  EXPECT_EQ(tracer.starts, 3);
  EXPECT_EQ(tracer.ends, 3);
}

TEST_F(LinuxExecutionStatsTest, SkipsCollectionWithoutConsumers) {
  set_ethosu_stats_buffer(nullptr);
  invocation.event_tracer = nullptr;
  EXPECT_EQ(ethosu_linux_inference_begin(invocation), Error::Ok);
  EXPECT_FALSE(invocation.pmu_enabled);
  ethosu_linux_inference_end(nullptr, invocation, true);
}

TEST_F(LinuxExecutionStatsTest, EmptyEventsStillCollectsCycles) {
  ASSERT_EQ(set_ethosu55_65_pmu_events({}), Error::Ok);
  ASSERT_EQ(ethosu_linux_inference_begin(invocation), Error::Ok);
  EXPECT_TRUE(invocation.pmu_enabled);
  EthosU::Inference inference;
  inference.cycles = 100;
  ethosu_linux_inference_end(&inference, invocation, true);
  ASSERT_EQ(buffer.size, 1);
  EXPECT_EQ(records[0].stats.counter_count, 0);
  EXPECT_EQ(records[0].stats.cycle_count, 100);
}
TEST_F(LinuxExecutionStatsTest, MetadataAllocationFailureClosesTrace) {
  ASSERT_EQ(ethosu_linux_inference_begin(invocation), Error::Ok);
  EthosUExecutionRecord record{};
  fail_next_allocation = true;
  ethosu_execution_end(invocation, &record);
  EXPECT_FALSE(fail_next_allocation);
  EXPECT_EQ(tracer.starts, 1);
  EXPECT_EQ(tracer.ends, 1);
  EXPECT_TRUE(tracer.metadata.empty());
}

TEST_F(LinuxExecutionStatsTest, CaptureAllocationFailureDoesNotEscape) {
  EthosUExecutionRecord record{};
  record.stats.device_identifier.assign(256, 'x');
  invocation.event_tracer = nullptr;
  ASSERT_EQ(ethosu_linux_inference_begin(invocation), Error::Ok);
  fail_next_allocation = true;
  ethosu_execution_end(invocation, &record);
  EXPECT_FALSE(fail_next_allocation);
  EXPECT_EQ(buffer.size, 0);
  ethosu_execution_end(invocation, &record);
  EXPECT_EQ(buffer.size, 1);
}

TEST_F(LinuxExecutionStatsTest, RecordCopyFailureLeavesBufferUsable) {
  EthosUExecutionRecord record{};
  record.stats.device_identifier.assign(256, 'x');
  bool failed = false;
  fail_next_allocation = true;
  try {
    buffer.append(record);
  } catch (const std::bad_alloc&) {
    failed = true;
  }
  EXPECT_TRUE(failed);
  EXPECT_EQ(buffer.size, 0);
  buffer.append(record);
  ASSERT_EQ(buffer.size, 1);
  EXPECT_EQ(records[0].stats.device_identifier, record.stats.device_identifier);
}

TEST_F(
    LinuxExecutionStatsTest,
    BackendBalancesSubmissionAndCompletionFailures) {
  VelaIOs outputs{0};
  VelaHandles handles{};
  handles.outputs = &outputs;
  ExecutionHandle handle{};
  handle.handles = handles;
  ASSERT_EQ(platform_init({}, nullptr, &handle), Error::Ok);
  MemoryAllocator allocator(0, nullptr);
  BackendExecutionContext context(&tracer, &allocator, nullptr);
  for (int failure = 0; failure < 4; ++failure) {
    EthosU::Inference::throw_on_submit = failure == 0;
    EthosU::Inference::throw_on_wait = failure == 1;
    EthosU::Inference::timeout = failure == 2;
    EthosU::Inference::result = EthosU::InferenceStatus::ERROR;
    EXPECT_EQ(
        platform_execute(context, &handle, handles, 0, 0, {}, nullptr),
        Error::InvalidState);
    EXPECT_EQ(tracer.starts, failure + 1);
    EXPECT_EQ(tracer.ends, failure + 1);
  }
  EXPECT_EQ(buffer.size, 0);
  EthosU::Inference::throw_on_submit = false;
  EthosU::Inference::throw_on_wait = false;
  EthosU::Inference::timeout = false;
  EthosU::Inference::result = EthosU::InferenceStatus::OK;
  EXPECT_EQ(
      platform_execute(context, &handle, handles, 0, 0, {}, nullptr),
      Error::Ok);
  EXPECT_TRUE(EthosU::Inference::cycle_counter_enabled);
  EXPECT_EQ(buffer.size, 1);
  EXPECT_EQ(tracer.starts, 5);
  EXPECT_EQ(tracer.ends, 5);
  platform_destroy(handle.platform_state);
}

// cppcheck-suppress-end syntaxError

} // namespace
