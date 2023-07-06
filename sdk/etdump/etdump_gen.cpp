#include <executorch/executor/MemoryAllocator.h>
#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/sdk/etdump/etdump_gen.h>

using namespace torch::executor;

uint8_t* CustomFlatbuffersAllocator::allocate(size_t size) {
  auto ptr = memory_allocator_.allocate(size);
  ET_CHECK_MSG(
      ptr != nullptr, "Failed to allocate memory in flatbuffers allocator");
  return reinterpret_cast<uint8_t*>(ptr);
}

void CustomFlatbuffersAllocator::deallocate(uint8_t* p, size_t size) {
  (void)p;
  (void)size;
}

// ETDumpGen with custom allocator support
const uint8_t* ETDumpGen::get_etdump_data() const {
  return builder_.GetBufferPointer();
}

size_t ETDumpGen::get_etdump_size() const {
  return builder_.GetSize();
}

void ETDumpGen::convert_to_flatbuffer_allocators(
    std::vector<flatbuffers::Offset<etdump::Allocator>>& etdump_allocators,
    const prof_allocator_t* prof_allocators,
    size_t prof_allocators_count) {
  etdump_allocators.reserve(prof_allocators_count);

  // Create representations of all the allocators in the flatbuffer.
  for (size_t i = 0; i < prof_allocators_count; ++i) {
    auto name = builder_.CreateString(prof_allocators[i].name);
    auto allocator = etdump::CreateAllocator(builder_, name);
    etdump_allocators.push_back(allocator);
  }
}

void ETDumpGen::convert_to_flatbuffer_profile_events(
    std::vector<flatbuffers::Offset<etdump::ProfileEvent>>& etdump_prof_events,
    const prof_event_t* prof_events,
    size_t prof_events_count) {
  std::vector<flatbuffers::Offset<etdump::ProfileEvent>> fb_profile_events;
  etdump_prof_events.reserve(prof_events_count);

  // Parse through all the profiling events and add them to the flatbuffer.
  for (size_t i = 0; i < prof_events_count; ++i) {
    auto name = builder_.CreateString(prof_events[i].name);
    auto profile_event = etdump::CreateProfileEvent(
        builder_,
        name,
        prof_events[i].instruction_idx,
        prof_events[i].start_time,
        prof_events[i].end_time);
    etdump_prof_events.push_back(profile_event);
  }
}

void ETDumpGen::convert_to_flatbuffer_mem_events(
    std::vector<flatbuffers::Offset<etdump::AllocationEvent>>&
        etdump_mem_alloc_events,
    const mem_prof_event_t* mem_prof_events_offsets,
    size_t mem_prof_events_offsets_count) {
  etdump_mem_alloc_events.reserve(mem_prof_events_offsets_count);

  // Parse through all the memory allocation events and add them to the
  // flatbuffer.
  for (size_t i = 0; i < mem_prof_events_offsets_count; ++i) {
    auto allocation_event = etdump::CreateAllocationEvent(
        builder_,
        mem_prof_events_offsets[i].allocator_id,
        mem_prof_events_offsets[i].allocation_size);
    etdump_mem_alloc_events.push_back(allocation_event);
  }
}

void ETDumpGen::CreateProfileBlockEntry(prof_header_t* prof_header) {
  // Create Allocator flatbuffers::Offset objects
  std::vector<flatbuffers::Offset<etdump::Allocator>> allocators_offsets;
  const prof_allocator_t* allocators_base =
      (prof_allocator_t*)((uintptr_t)prof_header + prof_mem_alloc_info_offset);
  convert_to_flatbuffer_allocators(
      allocators_offsets, allocators_base, prof_header->allocator_entries);

  std::vector<flatbuffers::Offset<etdump::ProfileEvent>> profile_events_offsets;
  const prof_event_t* prof_events_base =
      (prof_event_t*)((uintptr_t)prof_header + prof_events_offset);
  convert_to_flatbuffer_profile_events(
      profile_events_offsets, prof_events_base, prof_header->prof_entries);

  std::vector<flatbuffers::Offset<etdump::AllocationEvent>>
      mem_prof_events_offsets;
  const mem_prof_event_t* mem_prof_events_offsets_base =
      (mem_prof_event_t*)((uintptr_t)prof_header + prof_mem_alloc_events_offset);
  convert_to_flatbuffer_mem_events(
      mem_prof_events_offsets,
      mem_prof_events_offsets_base,
      prof_header->mem_prof_entries);

  // Create the ProfileBlock flatbuffers::Offset object
  auto name_offset = builder_.CreateString(prof_header->name);
  auto allocators_vector = builder_.CreateVector(allocators_offsets);
  auto profile_events_vector = builder_.CreateVector(profile_events_offsets);
  auto mem_prof_events_vector = builder_.CreateVector(mem_prof_events_offsets);

  prof_blocks_offsets.push_back(etdump::CreateProfileBlock(
      builder_,
      name_offset,
      allocators_vector,
      profile_events_vector,
      mem_prof_events_vector));
}

void ETDumpGen::generate_etdump() {
  // Create a RunData object using the profile blocks
  // and debug blocks data.
  auto run_data_offset = etdump::CreateRunData(
      builder_,
      builder_.CreateVector(debug_blocks_offsets),
      builder_.CreateVector(prof_blocks_offsets));

  std::vector<flatbuffers::Offset<etdump::RunData>> run_data_vector;
  run_data_vector.push_back(run_data_offset);
  auto run_data_fb_vector = builder_.CreateVector(run_data_vector);

  // Create ETDump object with version and run_data
  etdump::ETDumpBuilder et_dump_builder(builder_);
  et_dump_builder.add_version(1);
  et_dump_builder.add_run_data(run_data_fb_vector);
  auto et_dump = et_dump_builder.Finish();
  etdump::FinishETDumpBuffer(builder_, et_dump);
}
