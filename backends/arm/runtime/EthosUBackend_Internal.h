/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Workaround for runtime/core/portable_type/c10/c10/util/Float16-math.h
#if defined(__GNUC__) && defined(__ZEPHYR__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#endif

#include <executorch/backends/arm/runtime/VelaBinStream.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <cstddef>
#include <cstdint>

#if defined(__GNUC__) && defined(__ZEPHYR__)
#pragma GCC diagnostic pop
#endif

#if defined(ET_EVENT_TRACER_ENABLED)
#include <executorch/runtime/core/event_tracer.h>
#include <executorch/runtime/core/event_tracer_hooks.h>
using executorch::runtime::EventTracer;
using executorch::runtime::EventTracerEntry;

class EventTraceScope {
 public:
  EventTraceScope(EventTracer* event_tracer_, const char* name) {
    event_tracer = event_tracer_;
    event_tracer_entry_scope = event_tracer->start_profiling(name);
  }
  ~EventTraceScope() {
    event_tracer->end_profiling(event_tracer_entry_scope);
  }

 private:
  EventTracer* event_tracer;
  EventTracerEntry event_tracer_entry_scope;
};
#define EXECUTORCH_PROF_SCOPE(EVENTTRACER, NAME) \
  EventTraceScope event_tracer_scope = EventTraceScope(EVENTTRACER, NAME)
#define EXECUTORCH_PROF_START(EVENTTRACER, SCOPE, NAME) \
  SCOPE = EVENTTRACER->start_profiling(NAME)
#define EXECUTORCH_PROF_END(EVENTTRACER, SCOPE) \
  EVENTTRACER->end_profiling(SCOPE)
#else
#define EXECUTORCH_PROF_SCOPE(EVENTTRACER, NAME)
#define EXECUTORCH_PROF_START(EVENTTRACER, SCOPE, NAME)
#define EXECUTORCH_PROF_END(EVENTTRACER, SCOPE)
#endif

// Base address registers handed to the Ethos-U driver. Index maps to Vela
// region id: 0 weights, 1 scratch, 2 fast-scratch, 3 input (unused), 4 output
// (unused), 5 persistent (delegate-owned streaming state). The U55/U65/U85
// hardware and core-driver support up to NPU_REG_BASEP_ARRLEN (8) regions.
// Without ETHOSU_PERSISTENT_REGION only regions 0-2 are passed.
#ifdef ETHOSU_PERSISTENT_REGION
#define ETHOSU_NUM_BASE_ADDRS 6
#else
#define ETHOSU_NUM_BASE_ADDRS 3
#endif

namespace executorch {
namespace backends {
namespace arm {

struct PlatformState;

struct ExecutionHandle {
  PlatformState* platform_state;
  VelaHandles handles{};
  // Dedicated persistent (Vela region 5) buffer for delegate-owned streaming
  // state, base address 5. Kept separate from the scratch region (base 1) so
  // state is isolated from both intermediates and the read-only weights.
  // Allocated from the model-lifetime runtime allocator and zeroed once at
  // init(); null when the model has no persistent state.
  char* persistent_region{nullptr};
};

extern "C" {
void EthosUBackend_execute_begin();
void EthosUBackend_execute_end();
#if defined(ET_ARM_ETHOSU_PER_DELEGATE_PROFILING)
void EthosUBackend_delegate_begin(const void* handle);
void EthosUBackend_delegate_end();
#endif
#if defined(ET_ARM_ETHOSU_PROFILE_IO_COPIES)
void EthosUBackend_input_memcpy(size_t size);
void EthosUBackend_output_memcpy(size_t size);
#endif
extern unsigned char* ethosu_fast_scratch;
extern size_t ethosu_fast_scratch_size;
}

executorch::runtime::Error platform_init(
    executorch::runtime::ArrayRef<executorch::runtime::CompileSpec> specs,
    executorch::runtime::MemoryAllocator* allocator,
    ExecutionHandle* handle);

void platform_destroy(PlatformState* state);

bool needs_scratch_allocation();

executorch::runtime::Error platform_execute(
    executorch::runtime::BackendExecutionContext& context,
    const ExecutionHandle* execution_handle,
    const VelaHandles& handles,
    int input_count,
    int output_count,
    executorch::runtime::Span<executorch::runtime::EValue*> args,
    char* ethosu_scratch);

executorch::runtime::Error copy_with_layout_adjustment(
    const VelaIO& output_io,
    int output_index,
    const char* src,
    executorch::aten::Tensor& tensor_out,
    size_t tensor_bytes);

void calculate_dimensions(
    const executorch::aten::Tensor tensor,
    VelaIO* io,
    int* tensor_count,
    int* io_count);

} // namespace arm
} // namespace backends
} // namespace executorch
