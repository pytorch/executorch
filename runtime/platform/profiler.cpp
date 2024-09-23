/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <string.h>

#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/profiler.h>
#include <inttypes.h>

namespace executorch {
namespace runtime {

namespace {
static uint8_t prof_buf[prof_buf_size * MAX_PROFILE_BLOCKS];
// Base pointer for header
static prof_header_t* prof_header =
    (prof_header_t*)((uintptr_t)prof_buf + prof_header_offset);
// Base pointer for profiling entries
static prof_event_t* prof_arr =
    (prof_event_t*)((uintptr_t)prof_buf + prof_events_offset);
// Base pointer for memory allocator info array
static prof_allocator_t* mem_allocator_arr =
    (prof_allocator_t*)((uintptr_t)prof_buf + prof_mem_alloc_info_offset);
// Base pointer for memory profiling entries
static mem_prof_event_t* mem_prof_arr =
    (mem_prof_event_t*)((uintptr_t)prof_buf + prof_mem_alloc_events_offset);

static uint32_t num_blocks = 0;
static bool prof_stats_dumped = false;
prof_state_t profile_state_tls{-1, 0u};
} // namespace

const prof_state_t& get_profile_tls_state() {
  return profile_state_tls;
}

void set_profile_tls_state(const prof_state_t& state) {
  profile_state_tls = state;
}

ExecutorchProfilerInstructionScope::ExecutorchProfilerInstructionScope(
    const prof_state_t& state)
    : old_state_(get_profile_tls_state()) {
  set_profile_tls_state(state);
}

ExecutorchProfilerInstructionScope::~ExecutorchProfilerInstructionScope() {
  set_profile_tls_state(old_state_);
}

uint32_t begin_profiling(const char* name) {
  ET_CHECK_MSG(
      prof_header->prof_entries < MAX_PROFILE_EVENTS,
      "Out of profiling buffer space. Increase MAX_PROFILE_EVENTS and re-compile.");
  uint32_t curr_counter = prof_header->prof_entries;
  prof_header->prof_entries++;
  prof_arr[curr_counter].end_time = 0;
  prof_arr[curr_counter].name_str = name;
  prof_state_t state = get_profile_tls_state();
  prof_arr[curr_counter].chain_idx = state.chain_idx;
  prof_arr[curr_counter].instruction_idx = state.instruction_idx;
  // Set start time at the last to ensure that we're not capturing
  // any of the overhead in this function.
  prof_arr[curr_counter].start_time = et_pal_current_ticks();
  return curr_counter;
}

void end_profiling(uint32_t token_id) {
  ET_CHECK_MSG(token_id < MAX_PROFILE_EVENTS, "Invalid token id.");
  prof_arr[token_id].end_time = et_pal_current_ticks();
}

void dump_profile_stats(prof_result_t* prof_result) {
  prof_result->prof_data = (uint8_t*)prof_buf;
  prof_result->num_bytes = num_blocks * prof_buf_size;
  prof_result->num_blocks = num_blocks;

  if (!prof_stats_dumped) {
    for (size_t i = 0; i < num_blocks; i++) {
      prof_header_t* prof_header_local =
          (prof_header_t*)(prof_buf + prof_buf_size * i);
      prof_event_t* prof_event_local =
          (prof_event_t*)(prof_buf + prof_buf_size * i + prof_events_offset);
      // Copy over the string names into the space allocated in prof_event_t. We
      // avoided doing this earlier to keep the overhead in begin_profiling and
      // end_profiling as low as possible.
      for (size_t j = 0; j < prof_header_local->prof_entries; j++) {
        size_t str_len = strlen(prof_event_local[j].name_str);
        const char* str_ptr = prof_event_local[j].name_str;
        memset(prof_event_local[j].name, 0, PROF_NAME_MAX_LEN);
        if (str_len > PROF_NAME_MAX_LEN) {
          memcpy(prof_event_local[j].name, str_ptr, PROF_NAME_MAX_LEN);
        } else {
          memcpy(prof_event_local[j].name, str_ptr, str_len);
        }
      }
    }
  }

  prof_stats_dumped = true;
}

void reset_profile_stats() {
  prof_stats_dumped = false;
  prof_header->prof_entries = 0;
  prof_header->allocator_entries = 0;
  prof_header->mem_prof_entries = 0;
}

void track_allocation(int32_t id, uint32_t size) {
  if (id == -1)
    return;
  ET_CHECK_MSG(
      prof_header->mem_prof_entries < MAX_MEM_PROFILE_EVENTS,
      "Out of memory profiling buffer space. Increase MAX_MEM_PROFILE_EVENTS\
       to %" PRIu32 " and re-compile.",
      prof_header->mem_prof_entries);
  mem_prof_arr[prof_header->mem_prof_entries].allocator_id = id;
  mem_prof_arr[prof_header->mem_prof_entries].allocation_size = size;
  prof_header->mem_prof_entries++;
}

uint32_t track_allocator(const char* name) {
  ET_CHECK_MSG(
      prof_header->allocator_entries < MEM_PROFILE_MAX_ALLOCATORS,
      "Out of allocator tracking space, %d is needed. Increase MEM_PROFILE_MAX_ALLOCATORS and re-compile",
      prof_header->allocator_entries);
  size_t str_len = strlen(name);
  size_t num_allocators = prof_header->allocator_entries;
  memset(mem_allocator_arr[num_allocators].name, 0, PROF_NAME_MAX_LEN);
  if (str_len > PROF_NAME_MAX_LEN) {
    memcpy(mem_allocator_arr[num_allocators].name, name, PROF_NAME_MAX_LEN);
  } else {
    memcpy(mem_allocator_arr[num_allocators].name, name, str_len);
  }
  mem_allocator_arr[num_allocators].allocator_id = num_allocators;
  return prof_header->allocator_entries++;
}

void profiling_create_block(const char* name) {
  // If the current profiling block is not used then continue to use this, if
  // not move onto the next block.
  if (prof_header->prof_entries != 0 || prof_header->mem_prof_entries != 0 ||
      prof_header->allocator_entries != 0 || num_blocks == 0) {
    num_blocks += 1;
    ET_CHECK_MSG(
        num_blocks <= MAX_PROFILE_BLOCKS,
        "Only %d blocks are supported and they've all been used up but %d is used. Increment MAX_PROFILE_BLOCKS and re-run",
        MAX_PROFILE_BLOCKS,
        num_blocks);
  }

  // Copy over the name of this profiling block.
  size_t str_len =
      strlen(name) >= PROF_NAME_MAX_LEN ? PROF_NAME_MAX_LEN : strlen(name);
  uintptr_t base = (uintptr_t)prof_buf + (num_blocks - 1) * prof_buf_size;
  prof_header = (prof_header_t*)(base + prof_header_offset);
  memset(prof_header->name, 0, PROF_NAME_MAX_LEN);
  memcpy(prof_header->name, name, str_len);

  // Set profiler version for compatiblity checks in the post-processing
  // tool.
  prof_header->prof_ver = ET_PROF_VER;
  // Set the maximum number of entries that this block can support.
  prof_header->max_prof_entries = MAX_PROFILE_EVENTS;
  prof_header->max_allocator_entries = MEM_PROFILE_MAX_ALLOCATORS;
  prof_header->max_mem_prof_entries = MAX_MEM_PROFILE_EVENTS;
  reset_profile_stats();

  // Set the base addresses for the various profiling entries arrays.
  prof_arr = (prof_event_t*)(base + prof_events_offset);
  mem_allocator_arr = (prof_allocator_t*)(base + prof_mem_alloc_info_offset);
  mem_prof_arr = (mem_prof_event_t*)(base + prof_mem_alloc_events_offset);
}

void profiler_init(void) {
  profiling_create_block("default");
}

ExecutorchProfiler::ExecutorchProfiler(const char* name) {
  prof_tok = begin_profiling(name);
}

ExecutorchProfiler::~ExecutorchProfiler() {
  end_profiling(prof_tok);
}

} // namespace runtime
} // namespace executorch
