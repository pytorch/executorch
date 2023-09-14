/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "executorch/sdk/etdump/etdump_flatcc.h"
#include <string.h>
#include "executorch/runtime/platform/assert.h"

namespace torch {
namespace executor {

// Constructor implementation
ETDumpGen::ETDumpGen(void* buffer, size_t buf_size) {
  // Initialize the flatcc builder using the buffer and buffer size
  flatcc_builder_init(&builder);
  flatbuffers_buffer_start(&builder, etdump_ETDump_file_identifier);
  etdump_ETDump_start(&builder);
  etdump_ETDump_version_add(&builder, ETDUMP_VERSION);
  etdump_ETDump_run_data_start(&builder);
  etdump_ETDump_run_data_push_start(&builder);
}

ETDumpGen::~ETDumpGen() {
  flatcc_builder_clear(&builder);
}

void ETDumpGen::clear_builder() {
  flatcc_builder_clear(&builder);
}

void ETDumpGen::create_event_block(const char* name) {
  if (etdump_gen_state == ETDumpGen_Adding_Events) {
    etdump_RunData_events_end(&builder);
  }
  if (num_blocks > 0) {
    etdump_ETDump_run_data_push_end(&builder);
    etdump_ETDump_run_data_push_start(&builder);
  }
  ++num_blocks;
  etdump_RunData_name_create_strn(&builder, name, strlen(name));
  etdump_gen_state = ETDumpGen_Block_Created;
}

int64_t ETDumpGen::create_string_entry(const char* name) {
  return flatbuffers_string_create_str(&builder, name);
}

void ETDumpGen::check_ready_to_add_events() {
  if (etdump_gen_state != ETDumpGen_Adding_Events) {
    ET_CHECK_MSG(
        (etdump_gen_state == ETDumpGen_Adding_Allocators ||
         etdump_gen_state == ETDumpGen_Block_Created),
        "ETDumpGen in an invalid state. Cannot add new events now.");
    if (etdump_gen_state == ETDumpGen_Adding_Allocators) {
      etdump_RunData_allocators_end(&builder);
    }
    etdump_RunData_events_start(&builder);
    etdump_gen_state = ETDumpGen_Adding_Events;
  }
}

EventTracerEntry ETDumpGen::start_profiling(
    const char* name,
    ChainID chain_id,
    DebugHandle debug_handle) {
  EventTracerEntry prof_entry;
  prof_entry.event_id = create_string_entry(name);

  if (chain_id == -1 && debug_handle == 0) {
    prof_entry.chain_id = chain_id_;
    prof_entry.debug_handle = debug_handle_;

  } else {
    prof_entry.chain_id = chain_id;
    prof_entry.debug_handle = debug_handle;
  }
  prof_entry.start_time = et_pal_current_ticks();
  return prof_entry;
}

// TODO: Update all occurrences of the ProfileEvent calls once the
// EventTracerEntry struct is updated.
void ETDumpGen::end_profiling(EventTracerEntry prof_entry) {
  et_timestamp_t end_time = et_pal_current_ticks();
  check_ready_to_add_events();
  etdump_RunData_events_push_start(&builder);

  etdump_Event_profile_event_create(
      &builder,
      prof_entry.event_id, // see todo - change to prof_entry.name
      prof_entry.chain_id,
      -1, // see todo - change to prof_entry.instruction_id
      prof_entry.debug_handle, // see todo - change to
                               // prof_entry.delegate_debug_id_int
      flatbuffers_string_create_str(
          &builder, ""), // see todo - change to prof_entry.dlegate_debug_id_str
      prof_entry.start_time,
      end_time);
  etdump_RunData_events_push_end(&builder);
}

AllocatorID ETDumpGen::track_allocator(const char* name) {
  ET_CHECK_MSG(
      (etdump_gen_state == ETDumpGen_Block_Created ||
       etdump_gen_state == ETDumpGen_Adding_Allocators),
      "Allocators can only be added immediately after a new block is created and before any events are added.");
  if (etdump_gen_state != ETDumpGen_Adding_Allocators) {
    etdump_RunData_allocators_start(&builder);
    etdump_gen_state = ETDumpGen_Adding_Allocators;
  }
  flatbuffers_string_ref_t ref = create_string_entry(name);
  etdump_RunData_allocators_push_create(&builder, ref);
  return etdump_RunData_allocators_reserved_len(&builder);
}

void ETDumpGen::track_allocation(
    AllocatorID allocator_id,
    size_t allocation_size) {
  check_ready_to_add_events();

  // etdump_AllocationEvent_ref_t alloc_event_ref =
  //     etdump_AllocationEvent_create(&builder, allocator_id, allocation_size);
  etdump_RunData_events_push_start(&builder);
  etdump_Event_allocation_event_create(&builder, allocator_id, allocation_size);
  etdump_RunData_events_push_end(&builder);
}

etdump_result ETDumpGen::get_etdump_data() {
  etdump_result result;
  if (etdump_gen_state == ETDumpGen_Adding_Events) {
    etdump_RunData_events_end(&builder);
  } else if (etdump_gen_state == ETDumpGen_Adding_Allocators) {
    etdump_RunData_allocators_end(&builder);
  }
  etdump_ETDump_run_data_push_end(&builder);
  etdump_ETDump_run_data_end(&builder);
  etdump_ETDump_ref_t root = etdump_ETDump_end(&builder);
  flatbuffers_buffer_end(&builder, root);
  if (num_blocks == 0) {
    result = {nullptr, 0};
  } else {
    result.buf = flatcc_builder_finalize_aligned_buffer(&builder, &result.size);
  }
  return result;
}

size_t ETDumpGen::get_num_blocks() {
  return num_blocks;
}
} // namespace executor
} // namespace torch
