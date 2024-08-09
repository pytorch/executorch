/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/event_tracer.h>

/**
 * @file
 *
 * This file contains the hooks that are inserted across various parts of the
 * core runtime code to call into the EventTracer class for logging of profiling
 * and debugging events. Any calls made to the EventTracer from the runtime must
 * be made via these hooks.
 * Users shouldn't directly add these hooks in their code and it's meant only
 * for usage in ExecuTorch internal code.
 *
 * The benefit of defining these hooks is that we can easily control whether or
 * not we want to compile in the EventTracer code based on the status of the
 * ET_EVENT_TRACER_ENABLED flag.
 *
 * TODO(dbort): Make this a private header of runtime/executor. It only contains
 * runtime-internal functions and should not be part of the public set of
 * headers.
 */

namespace executorch {
namespace runtime {
namespace internal {

/**
 * This class enables scope based profiling where needed using RAII.
 * Profiling will be started when the object is created and will end
 * when the object goes out of scope.
 */
class EventTracerProfileScope final {
 public:
  EventTracerProfileScope(EventTracer* event_tracer, const char* name) {
#ifdef ET_EVENT_TRACER_ENABLED
    event_tracer_ = event_tracer;
    if (event_tracer_ == nullptr) {
      return;
    }
    event_entry_ = event_tracer->start_profiling(name);
#else //! ET_EVENT_TRACER_ENABLED
    (void)event_tracer;
    (void)name;
#endif
  }

  ~EventTracerProfileScope() {
#ifdef ET_EVENT_TRACER_ENABLED
    if (event_tracer_ == nullptr) {
      return;
    }
    event_tracer_->end_profiling(event_entry_);
#endif
  }

 private:
#ifdef ET_EVENT_TRACER_ENABLED
  EventTracer* event_tracer_;
  EventTracerEntry event_entry_;
#endif
};

/**
 * This class helps us set and then clear out the chain id and debug handle
 * values stored in the event tracer class using RAII. This is typically called
 * in the executor loop before entering the codegen layer to configure the chain
 * id and debug handle of the current instruction being executed.
 * After we return from the kernel execution we can then reset the chain id and
 * debug handle to defaults when this object goes out of scope.
 */
class EventTracerProfileInstructionScope final {
 public:
  EventTracerProfileInstructionScope(
      EventTracer* event_tracer,
      ChainID chain_idx,
      DebugHandle debug_handle) {
#ifdef ET_EVENT_TRACER_ENABLED
    event_tracer_ = event_tracer;
    if (event_tracer_ == nullptr) {
      return;
    }
    event_tracer_->set_chain_debug_handle(chain_idx, debug_handle);
#else //! ET_EVENT_TRACER_ENABLED
    (void)event_tracer;
    (void)chain_idx;
    (void)debug_handle;
#endif
  }

  ~EventTracerProfileInstructionScope() {
#ifdef ET_EVENT_TRACER_ENABLED
    if (event_tracer_ == nullptr) {
      return;
    }
    event_tracer_->set_chain_debug_handle(kUnsetChainId, kUnsetDebugHandle);
#endif
  }

 private:
#ifdef ET_EVENT_TRACER_ENABLED
  EventTracer* event_tracer_;
#endif
};

/**
 * Create a new event block with the specified name. Any events logged
 * after this will be associated with this new event block.
 */
inline void event_tracer_create_event_block(
    EventTracer* event_tracer,
    char const* name) {
#ifdef ET_EVENT_TRACER_ENABLED
  if (event_tracer) {
    event_tracer->create_event_block(name);
  }
#else //! ET_EVENT_TRACER_ENABLED
  (void)event_tracer;
  (void)name;
#endif
}

/**
 * Explicitly mark the beginning of a new profiling event. This returns
 * an instance of an EventTracerEntry object that the user needs to keep
 * around and pass into the corresponding event_tracer_end_profiling_event
 * call.
 */
inline EventTracerEntry event_tracer_begin_profiling_event(
    EventTracer* event_tracer,
    char const* name) {
#ifdef ET_EVENT_TRACER_ENABLED
  if (event_tracer) {
    return event_tracer->start_profiling(name);
  }
#else //! ET_EVENT_TRACER_ENABLED
  (void)event_tracer;
  (void)name;
#endif
  // There is no active tracer; this value will be ignored.
  return EventTracerEntry();
}

/**
 * Mark the end of a profiling event passing in the entry token
 * returned by a previous call to ET_EVENT_TRACER_BEGIN_PROFILING_EVENT.
 */
inline void event_tracer_end_profiling_event(
    EventTracer* event_tracer,
    EventTracerEntry event) {
#ifdef ET_EVENT_TRACER_ENABLED
  if (event_tracer) {
    event_tracer->end_profiling(event);
  }
#else //! ET_EVENT_TRACER_ENABLED
  (void)event_tracer;
  (void)event;
#endif
}

/**
 * Start the tracking of the allocator represented by this name and returns
 * an AllocatorID that will be used to track all subsequent allocations done by
 * this allocator.
 */
inline AllocatorID event_tracer_track_allocator(
    EventTracer* event_tracer,
    const char* name) {
#ifdef ET_EVENT_TRACER_ENABLED
  if (event_tracer) {
    return event_tracer->track_allocator(name);
  }
#else //! ET_EVENT_TRACER_ENABLED
  (void)event_tracer;
  (void)name;
#endif
  // There is no active tracer; this value will be ignored.
  return 0;
}

/// Log the allocation event done via the allocator represented by id.
inline void event_tracer_track_allocation(
    EventTracer* event_tracer,
    AllocatorID id,
    size_t size) {
#ifdef ET_EVENT_TRACER_ENABLED
  if (event_tracer) {
    event_tracer->track_allocation(id, size);
  }
#else //! ET_EVENT_TRACER_ENABLED
  (void)event_tracer;
  (void)id;
  (void)size;
#endif
}

/// Log an intermediate value.
inline void event_tracer_log_evalue(EventTracer* event_tracer, EValue& evalue) {
#ifdef ET_EVENT_TRACER_ENABLED
  if (event_tracer) {
    if (event_tracer->event_tracer_debug_level() >=
        EventTracerDebugLogLevel::kIntermediateOutputs) {
      event_tracer->log_evalue(evalue, LoggedEValueType::kIntermediateOutput);
    }
  }
#else //! ET_EVENT_TRACER_ENABLED
  (void)event_tracer;
  (void)evalue;
#endif
}

/// Log a program output.
inline void event_tracer_log_evalue_output(
    EventTracer* event_tracer,
    const EValue& evalue) {
#ifdef ET_EVENT_TRACER_ENABLED
  /*
   * If debugging via event tracer is enabled but intermediate output logging is
   * disabled then we want to only log the outputs.
   */
  if (event_tracer) {
    if (event_tracer->event_tracer_debug_level() >=
        EventTracerDebugLogLevel::kProgramOutputs) {
      event_tracer->log_evalue(evalue, LoggedEValueType::kProgramOutput);
    }
  }
#else //! ET_EVENT_TRACER_ENABLED
  (void)event_tracer;
  (void)evalue;
#endif
}

// Set the bundled input index of the current bundled input being used by the
// method.
inline void event_tracer_set_bundled_input_index(
    EventTracer* event_tracer,
    int bundled_input_index) {
#ifdef ET_EVENT_TRACER_ENABLED
  if (event_tracer) {
    event_tracer->set_bundled_input_index(bundled_input_index);
  }
#else //! ET_EVENT_TRACER_ENABLED
  (void)event_tracer;
  (void)bundled_input_index;
#endif
}

} // namespace internal
} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
namespace internal {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::internal::event_tracer_begin_profiling_event;
using ::executorch::runtime::internal::event_tracer_create_event_block;
using ::executorch::runtime::internal::event_tracer_end_profiling_event;
using ::executorch::runtime::internal::event_tracer_log_evalue;
using ::executorch::runtime::internal::event_tracer_log_evalue_output;
using ::executorch::runtime::internal::event_tracer_set_bundled_input_index;
using ::executorch::runtime::internal::event_tracer_track_allocation;
using ::executorch::runtime::internal::event_tracer_track_allocator;
using ::executorch::runtime::internal::EventTracerProfileInstructionScope;
using ::executorch::runtime::internal::EventTracerProfileScope;
} // namespace internal
} // namespace executor
} // namespace torch
