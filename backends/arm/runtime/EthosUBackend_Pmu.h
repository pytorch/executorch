/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cstdint>

namespace executorch::backends::arm {

// EthosUStatsBuffer is defined in EthosUExecutionStats.h.
struct EthosUStatsBuffer;

struct EthosUInvocationContext {
  EthosUStatsBuffer* buffer = nullptr;
  uintptr_t delegate_handle = 0;
  uint32_t counter_count = 0;
};

bool ethosu_execution_metrics_enabled() noexcept;

} // namespace executorch::backends::arm
