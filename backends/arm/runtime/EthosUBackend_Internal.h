/*
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

#include <cstddef>
#include <cstdint>

#include <executorch/backends/arm/runtime/VelaBinStream.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>

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

#define ETHOSU_NUM_BASE_ADDRS 3

namespace executorch {
namespace backends {
namespace arm {

struct PlatformState;

struct ExecutionHandle {
  executorch::runtime::FreeableBuffer* processed;
  PlatformState* platform_state;
};

extern "C" {
void EthosUBackend_execute_begin();
void EthosUBackend_execute_end();
extern unsigned char* ethosu_fast_scratch;
extern size_t ethosu_fast_scratch_size;
}

PlatformState* platform_init(
    executorch::runtime::ArrayRef<executorch::runtime::CompileSpec> specs,
    executorch::runtime::MemoryAllocator* allocator);
void platform_destroy(PlatformState* state);
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
