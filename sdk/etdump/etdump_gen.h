/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/memory_allocator.h>
#include "executorch/sdk/etdump/etdump_schema_generated.h"

typedef flatbuffers::FlatBufferBuilder* ETDBuilder;

// Custom FlatBuffers Allocator using ExecuTorch MemoryAllocator.
class CustomFlatbuffersAllocator : public flatbuffers::Allocator {
 public:
  explicit CustomFlatbuffersAllocator(
      torch::executor::MemoryAllocator& memory_allocator)
      : memory_allocator_(memory_allocator) {}

  uint8_t* allocate(size_t size) override;
  void deallocate(uint8_t* p, size_t size) override;

 private:
  torch::executor::MemoryAllocator memory_allocator_;
};

/*
 * ETDumpGen is the class that is responsible for generating the etdump
 * flatbuffer using the profiling results that are passed into it.
 */
class ETDumpGen {
 public:
  /*
   * Constructor expects a memory allocator to be passed in from which
   * all the allocations needed for etdump generation will be done.
   */
  explicit ETDumpGen(torch::executor::MemoryAllocator& memory_allocator)
      : custom_allocator_(memory_allocator),
        builder_(1024, &custom_allocator_) {}

  /*
   * Get the pointer to the etdump flatbuffer data.
   */
  const uint8_t* get_etdump_data() const;
  /*
   * Get size of etdump flatbuffer.
   */
  size_t get_etdump_size() const;
  /*
   * This triggers the generation of the etdump flatbuffer.
   */
  void generate_etdump();
  /*
   * Create a profile block entry in etdump using the profiling data
   * retrieved from the ExecuTorch profiler.
   */
  void CreateProfileBlockEntry(torch::executor::prof_header_t* prof_header);

 private:
  CustomFlatbuffersAllocator custom_allocator_;
  flatbuffers::FlatBufferBuilder builder_;
  std::vector<flatbuffers::Offset<etdump::ProfileBlock>> prof_blocks_offsets;
  std::vector<flatbuffers::Offset<etdump::DebugBlock>> debug_blocks_offsets;
  void convert_to_flatbuffer_allocators(
      std::vector<flatbuffers::Offset<etdump::Allocator>>& etdump_allocators,
      const torch::executor::prof_allocator_t* prof_allocators,
      size_t prof_allocators_count);
  void convert_to_flatbuffer_profile_events(
      std::vector<flatbuffers::Offset<etdump::ProfileEvent>>&
          etdump_prof_events,
      const torch::executor::prof_event_t* prof_events,
      size_t prof_events_count);
  void convert_to_flatbuffer_mem_events(
      std::vector<flatbuffers::Offset<etdump::AllocationEvent>>&
          etdump_mem_alloc_events,
      const torch::executor::mem_prof_event_t* mem_prof_events_offsets,
      size_t mem_prof_events_offsets_count);
};
