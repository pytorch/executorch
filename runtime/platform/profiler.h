/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

namespace torch {
namespace executor {

// Version string used to check for compatibility with post-processing
// tool
#define ET_PROF_VER 0x00000001

// By default we support profiling upto 1024 perf events. Build
// targets can override this to increase the profiling buffer size
// during compilation.
#ifndef MAX_PROFILE_EVENTS
#define MAX_PROFILE_EVENTS 1024
#endif
// By default we support profiling upto 1024 memory allocation events.
// Build targets can choose to override this, which will consequently have
// the effect of increasing/decreasing the profiling buffer size.
#ifndef MAX_MEM_PROFILE_EVENTS
#define MAX_MEM_PROFILE_EVENTS 1024
#endif
// By default we support profiling only upto 16 allocators. If users
// have more allocators than these then they can override this during
// compilation time. There will be an increase/decrease in the profiling
// buffer size based on the way this value is changed.
#ifndef MEM_PROFILE_MAX_ALLOCATORS
#define MEM_PROFILE_MAX_ALLOCATORS 32
#endif
// By default we support only one profiling block. If users want to profile
// something that will be iterated on multiple times then they will have to
// increment this to support their use case. In post-processing the stats for
// all these iterations will be consolidated.
#ifndef MAX_PROFILE_BLOCKS
#define MAX_PROFILE_BLOCKS 2
#endif

#define PROF_NAME_MAX_LEN 32

typedef struct alignas(8) {
  union {
    const char* name_str;
    char name[PROF_NAME_MAX_LEN];
  };
  // chain_idx == -1 is a null value, when profile event happens out of chain
  // execution
  int32_t chain_idx;
  uint32_t instruction_idx;
  uint64_t start_time;
  uint64_t end_time;
} prof_event_t;

typedef struct alignas(8) {
  uint32_t allocator_id;
  uint32_t allocation_size;
} mem_prof_event_t;

typedef struct alignas(8) {
  char name[PROF_NAME_MAX_LEN];
  uint64_t allocator_id;
} prof_allocator_t;

typedef struct alignas(8) {
  uint8_t* prof_data;
  uint32_t num_bytes;
  uint32_t num_blocks;
} prof_result_t;

typedef struct alignas(8) {
  char name[32];
  uint32_t prof_ver;
  uint32_t max_prof_entries;
  uint32_t prof_entries;
  uint32_t max_allocator_entries;
  uint32_t allocator_entries;
  uint32_t max_mem_prof_entries;
  uint32_t mem_prof_entries;
} prof_header_t;

/*
This is what the layout of the profiling buffer looks like.
---------------------------------------
| Profiling header                    |
---------------------------------------
| Profile events (Perf events)        |
---------------------------------------
| Memory allocators info              |
---------------------------------------
| Profile events (Memory allocations) |
---------------------------------------
*/

// offsets of the various sections in the profiling buffer
// Total size required for profiling buffer
constexpr uint32_t prof_buf_size = sizeof(prof_header_t) +
    sizeof(prof_event_t) * MAX_PROFILE_EVENTS +
    sizeof(mem_prof_event_t) * MAX_MEM_PROFILE_EVENTS +
    sizeof(prof_allocator_t) * MEM_PROFILE_MAX_ALLOCATORS;

constexpr size_t prof_header_offset = 0;
constexpr size_t prof_events_offset = sizeof(prof_header_t);
constexpr size_t prof_mem_alloc_info_offset =
    prof_events_offset + sizeof(prof_event_t) * MAX_PROFILE_EVENTS;
constexpr size_t prof_mem_alloc_events_offset = prof_mem_alloc_info_offset +
    sizeof(prof_allocator_t) * MEM_PROFILE_MAX_ALLOCATORS;

// Set the initial state for the profiler assuming we're using the
// statically allocated buffer declared in the profiler module.
void profiler_init(void);

// This starts the profiling of this event and returns a token
// by which this event can be referred to in the future.
uint32_t begin_profiling(const char* name);

// End profiling event represented by token_id
void end_profiling(uint32_t token_id);

// Dump profiler results, return pointer to prof event array and number of
// events in it.
void dump_profile_stats(prof_result_t* prof_result);

void reset_profile_stats();

void track_allocation(int32_t id, uint32_t size);

uint32_t track_allocator(const char* name);

void profiling_create_block(const char* name);

// This class enables scope based profiling where needed. Profiling
// will be started when the object is created and will end when the
// object goes out of scope.
class ExecutorchProfiler {
 public:
  explicit ExecutorchProfiler(const char* name);

  ~ExecutorchProfiler();

 private:
  uint32_t prof_tok;
};

typedef struct {
  int32_t chain_idx;
  uint32_t instruction_idx;
} prof_state_t;

const prof_state_t& get_profile_tls_state();

void set_profile_tls_state(const prof_state_t& state);

class ExecutorchProfilerInstructionScope {
 public:
  explicit ExecutorchProfilerInstructionScope(const prof_state_t& state);
  ~ExecutorchProfilerInstructionScope();

  // ScopeGuard: non-copyable, non-movable
  ExecutorchProfilerInstructionScope(
      const ExecutorchProfilerInstructionScope&) = delete;
  ExecutorchProfilerInstructionScope& operator=(
      const ExecutorchProfilerInstructionScope&) = delete;

  ExecutorchProfilerInstructionScope(ExecutorchProfilerInstructionScope&&) =
      delete;
  ExecutorchProfilerInstructionScope& operator=(
      ExecutorchProfilerInstructionScope&&) = delete;

 private:
  prof_state_t old_state_;
};

} // namespace executor
} // namespace torch

#ifdef PROFILING_ENABLED

#define EXECUTORCH_PROFILE_CREATE_BLOCK(name) \
  torch::executor::profiling_create_block(name);

// Convenience macros to begin and end profiling. These can be inserted
// anywhere as it'll be ensured that for the prod builds these will
// essentially be noops.
#define EXECUTORCH_BEGIN_PROF(name) torch::executor::begin_profiling(name);

#define EXECUTORCH_END_PROF(token_id) torch::executor::end_profiling(token_id);

#define EXECUTORCH_SCOPE_PROF(name) \
  torch::executor::ExecutorchProfiler profiler(name);

#define EXECUTORCH_PROFILE_INSTRUCTION_SCOPE(chain_idx, instruction_idx) \
  torch::executor::ExecutorchProfilerInstructionScope                    \
      __profiler_instruction_scope({chain_idx, instruction_idx});

#define EXECUTORCH_DUMP_PROFILE_RESULTS(prof_result) \
  torch::executor::dump_profile_stats(prof_result);

#define EXECUTORCH_RESET_PROFILE_RESULTS() \
  torch::executor::reset_profile_stats();

#define EXECUTORCH_TRACK_ALLOCATOR(name) torch::executor::track_allocator(name);

#define EXECUTORCH_TRACK_ALLOCATION(id, size) \
  torch::executor::track_allocation(id, size);

#else

#define EXECUTORCH_PROFILE_CREATE_BLOCK(name) ({ (void)(name); })

#define EXECUTORCH_BEGIN_PROF(name) \
  {}

#define EXECUTORCH_END_PROF(token_id) ({ (void)(token_id); })

#define EXECUTORCH_SCOPE_PROF(name) ({ (void)(name); })

#define EXECUTORCH_PROFILE_INSTRUCTION_SCOPE(chain_idx, instruction_idx) \
  ({                                                                     \
    (void)(chain_idx);                                                   \
    (void)(instruction_idx);                                             \
  })

#define EXECUTORCH_DUMP_PROFILE_RESULTS(prof_result_test) \
  memset(prof_result_test, 0, sizeof(torch::executor::prof_result_t));

#define EXECUTORCH_RESET_PROFILE_RESULTS() \
  {}

#define EXECUTORCH_TRACK_ALLOCATOR(name) \
  ({                                     \
    (void)(name);                        \
    -1;                                  \
  })

#define EXECUTORCH_TRACK_ALLOCATION(id, size) \
  ({                                          \
    (void)(id);                               \
    (void)(size);                             \
  })

#endif
