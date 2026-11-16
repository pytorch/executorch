/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cstdint>

#if defined(ET_EVENT_TRACER_ENABLED)
#include <executorch/runtime/core/event_tracer.h>
#endif

namespace executorch::backends::arm {

// EthosUStatsBuffer is defined in EthosUExecutionStats.h.
struct EthosUStatsBuffer;

struct EthosUInvocationContext {
  EthosUStatsBuffer* buffer = nullptr;
  uintptr_t delegate_handle = 0;
  uint32_t counter_count = 0;
#if defined(ET_EVENT_TRACER_ENABLED)
  executorch::runtime::EventTracer* event_tracer = nullptr;
  executorch::runtime::EventTracerEntry cycle_event{};
#endif
};

bool ethosu_execution_metrics_enabled() noexcept;

} // namespace executorch::backends::arm
