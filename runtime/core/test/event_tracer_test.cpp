/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/runtime/core/event_tracer.h>
// Enable flag for test
#define ET_EVENT_TRACER_ENABLED
#include <executorch/runtime/core/event_tracer_hooks.h>
#include <executorch/runtime/core/event_tracer_hooks_delegate.h>

namespace torch {
namespace executor {

using namespace internal;

class DummyEventTracer : public EventTracer {
 public:
  DummyEventTracer() {}

  ~DummyEventTracer() override {}

  void create_event_block(const char* name) override {
    (void)name;
    return;
  }

  EventTracerEntry start_profiling(
      const char* name,
      ChainID chain_id = kUnsetChainId,
      DebugHandle debug_handle = kUnsetDebugHandle) override {
    (void)name;
    (void)chain_id;
    (void)debug_handle;
    return EventTracerEntry();
  }

  void end_profiling(EventTracerEntry prof_entry) override {
    (void)prof_entry;
    return;
  }

  void track_allocation(AllocatorID id, size_t size) override {
    (void)id;
    (void)size;
    return;
  }

  AllocatorID track_allocator(const char* name) override {
    (void)name;
    return 0;
  }

  EventTracerEntry start_profiling_delegate(
      const char* name,
      DebugHandle delegate_debug_id) override {
    (void)name;
    (void)delegate_debug_id;
    return EventTracerEntry();
  }

  void end_profiling_delegate(
      EventTracerEntry event_tracer_entry,
      const char* metadata) override {
    (void)event_tracer_entry;
    (void)metadata;
  }

  void log_profiling_delegate(
      const char* name,
      DebugHandle delegate_debug_id,
      et_timestamp_t start_time,
      et_timestamp_t end_time,
      const char* metadata = nullptr) override {
    (void)name;
    (void)delegate_debug_id;
    (void)start_time;
    (void)end_time;
    (void)metadata;
  }
};

/**
 * Exercise all the event_tracer API's for a basic sanity check.
 */
void RunSimpleTracerTest(EventTracer* event_tracer) {
  event_tracer_create_event_block(event_tracer, "ExampleEvent");
  event_tracer_create_event_block(event_tracer, "ExampleEvent");
  EventTracerEntry event_entry =
      event_tracer_begin_profiling_event(event_tracer, "ExampleEvent");
  event_tracer_end_profiling_event(event_tracer, event_entry);
  {
    EventTracerProfileScope event_tracer_profile_scope(
        event_tracer, "ExampleScope");
  }
  {
    EventTracerProfileInstructionScope event_tracer_profile_instruction_scope(
        event_tracer, 0, 1);
  }
  AllocatorID allocator_id =
      event_tracer_track_allocator(event_tracer, "AllocatorName");
  event_tracer_track_allocation(event_tracer, allocator_id, 64);
}

TEST(TestEventTracer, SimpleEventTracerTest) {
  // Call all the EventTracer macro's with a valid pointer to an event tracer
  // and also with a null pointer (to test that the null case works).
  DummyEventTracer dummy;
  std::vector<DummyEventTracer*> dummy_event_tracer_arr = {&dummy, nullptr};
  for (size_t i = 0; i < dummy_event_tracer_arr.size(); i++) {
    RunSimpleTracerTest(&dummy);
    RunSimpleTracerTest(nullptr);
  }
}

/**
 * Exercise all the event_tracer API's for delegates as a basic sanity check.
 */
void RunSimpleTracerTestDelegate(EventTracer* event_tracer) {
  EventTracerEntry event_tracer_entry = event_tracer_start_profiling_delegate(
      event_tracer, "test_event", kUnsetDebugHandle);
  event_tracer_end_profiling_delegate(
      event_tracer, event_tracer_entry, nullptr);
  event_tracer_start_profiling_delegate(event_tracer, nullptr, 1);
  event_tracer_end_profiling_delegate(
      event_tracer, event_tracer_entry, "test_metadata");
  event_tracer_log_profiling_delegate(
      event_tracer, "test_event", kUnsetDebugHandle, 0, 1, nullptr);
  event_tracer_log_profiling_delegate(event_tracer, nullptr, 1, 0, 1, nullptr);
}

TEST(TestEventTracer, SimpleEventTracerTestDelegate) {
  // Call all the EventTracer macro's with a valid pointer to an event tracer
  // and also with a null pointer (to test that the null case works).
  DummyEventTracer dummy;
  std::vector<DummyEventTracer*> dummy_event_tracer_arr = {&dummy, nullptr};
  for (size_t i = 0; i < dummy_event_tracer_arr.size(); i++) {
    RunSimpleTracerTestDelegate(&dummy);
    RunSimpleTracerTestDelegate(nullptr);
  }
}

} // namespace executor
} // namespace torch
// TODO : (T163645377) Add more test coverage to log and verify events passed
// into DummyTracer.
