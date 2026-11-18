/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <EthosULinuxPmuEvents.h>

#include <algorithm>
#include <exception>

#include <ethosu.hpp>
#include <executorch/backends/arm/runtime/EthosUBackend_Pmu_LinuxDriver.h>
#include <executorch/backends/arm/runtime/EthosUExecutionStats_Internal.h>
#include <executorch/runtime/platform/log.h>

namespace executorch::backends::arm {
using executorch::runtime::ArrayRef;
namespace {
uint32_t linux_product(uint32_t product_major) {
  switch (product_major) {
    case 4: // Ethos-U55 BASE.ID
      return 0;
    case 6: // Ethos-U65 BASE.ID
      return 1;
    case 7: // Ethos-U85 BASE.ID
      return 2;
    default:
      return UINT32_MAX;
  }
}

ArrayRef<ethosu_pmu_event_type> linux_event_table(uint32_t product) {
  if (product == 0 || product == 1) {
    return {kLinuxU65PmuEvents, std::size(kLinuxU65PmuEvents)};
  }
  if (product == 2) {
    return {kLinuxU85PmuEvents, std::size(kLinuxU85PmuEvents)};
  }
  return {};
}
} // namespace

bool ethosu_linux_pmu_event_supported(
    uint32_t product,
    ethosu_pmu_event_type event) {
  const auto table = linux_event_table(product);
  return std::find(table.begin(), table.end(), event) != table.end();
}

executorch::runtime::Error ethosu_linux_inference_begin(
    EthosUInvocationContext& invocation) {
  using executorch::runtime::Error;
  invocation.pmu_events = {};
  EthosU::Device device(invocation.device_path);
  const auto caps = device.capabilities();
  const auto product = linux_product(caps.hwId.product.major);
  const auto events = ethosu_prepare_invocation(invocation, product);
  const bool subsystem =
      caps.hwCfg.type == EthosU::HardwareConfiguration::DeviceType::SUBSYSTEM;
  const size_t counter_count = subsystem ? 4 : events.size();
  if (!events.empty()) {
    const auto table = linux_event_table(product);
    for (size_t slot = 0; slot < events.size(); ++slot) {
      const auto event = events[slot];
      if (slot >= counter_count) {
        if (event != ETHOSU_PMU_NO_EVENT) {
          ET_LOG(Error, "Ethos-U device has too few PMU counter slots");
          return Error::InvalidArgument;
        }
        continue;
      }
      const auto found = std::find(table.begin(), table.end(), event);
      if (found == table.end()) {
        return Error::InvalidArgument;
      }
      invocation.pmu_events[slot] = found - table.begin();
    }
    invocation.pmu_enabled = true;
  }
  ethosu_execution_begin(invocation);
  return Error::Ok;
}

void ethosu_linux_inference_end(
    const EthosU::Inference* inference,
    const EthosUInvocationContext& invocation,
    bool success) noexcept {
  if (!success || inference == nullptr || !invocation.pmu_enabled) {
    ethosu_execution_end(invocation, nullptr);
    return;
  }
  EthosUExecutionRecord record{};
  try {
    EthosU::Device device(invocation.device_path);
    const auto caps = device.capabilities();
    record.delegate_handle = invocation.delegate_handle;
    record.success = true;
    record.stats.device_identifier = invocation.device_path;
    record.stats.product = linux_product(caps.hwId.product.major);
    record.stats.macs_per_cycle = 1u << caps.hwCfg.macsPerClockCycle;
    record.stats.cycle_count = inference->getCycleCounter();
    record.stats.cycle_count_valid = true;
    const auto counts = inference->getPmuCounters();
    if (counts.size() < invocation.pmu_events.size()) {
      ethosu_execution_end(invocation, nullptr);
      return;
    }
    const auto table = linux_event_table(record.stats.product);
    for (uint32_t slot = 0; slot < invocation.pmu_events.size(); ++slot) {
      const auto event = table[invocation.pmu_events[slot]];
      if (event != ETHOSU_PMU_NO_EVENT) {
        record.stats.counters[record.stats.counter_count++] = {
            static_cast<uint32_t>(event), counts[slot], false};
      }
    }
  } catch (const std::exception& error) {
    ET_LOG(Debug, "Failed to read Ethos-U PMU counters: %s", error.what());
    ethosu_execution_end(invocation, nullptr);
    return;
  }
  ethosu_execution_end(invocation, &record);
}

} // namespace executorch::backends::arm
