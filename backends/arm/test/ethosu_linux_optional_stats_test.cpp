/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if defined(ET_ARM_ETHOSU_TEST_STATS_LINKED)
#include <executorch/backends/arm/runtime/EthosUExecutionStats.h>
#endif
#include <ethosu.hpp>
#include <executorch/backends/arm/runtime/EthosUBackend_Internal.h>
#include <executorch/backends/arm/runtime/EthosUBackend_Pmu.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>

namespace executorch::backends::arm {
bool __attribute__((weak)) ethosu_execution_metrics_enabled() noexcept {
  return false;
}
} // namespace executorch::backends::arm

using namespace executorch::backends::arm;
using executorch::runtime::Error;

TEST(EthosULinuxOptionalStats, InvokesWithOrWithoutStatsLibrary) {
  executorch::runtime::runtime_init();
  VelaIOs outputs{0};
  VelaHandles handles{};
  handles.outputs = &outputs;
  ExecutionHandle handle{};
  handle.handles = handles;
  ASSERT_EQ(platform_init({}, nullptr, &handle), Error::Ok);
  executorch::runtime::MemoryAllocator allocator(0, nullptr);
  executorch::runtime::BackendExecutionContext context(
      nullptr, &allocator, nullptr);
#if defined(ET_ARM_ETHOSU_TEST_STATS_LINKED)
  EthosUExecutionRecord records[1]{};
  EthosUStatsBuffer buffer{records, 1};
  set_ethosu_stats_buffer(&buffer);
#endif
  EXPECT_EQ(
      platform_execute(context, &handle, handles, 0, 0, {}, nullptr),
      Error::Ok);
#if defined(ET_ARM_ETHOSU_TEST_STATS_LINKED)
  EXPECT_TRUE(ethosu_execution_metrics_enabled());
  EXPECT_TRUE(EthosU::Inference::cycle_counter_enabled);
  ASSERT_EQ(buffer.size, 1);
  EXPECT_TRUE(records[0].stats.cycle_count_valid);
  EXPECT_EQ(records[0].stats.device_identifier, "/dev/ethosu0");
  set_ethosu_stats_buffer(nullptr);
#else
  EXPECT_FALSE(ethosu_execution_metrics_enabled());
  EXPECT_FALSE(EthosU::Inference::cycle_counter_enabled);
#endif
  platform_destroy(handle.platform_state);
}
