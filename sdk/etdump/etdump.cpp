#include <executorch/sdk/etdump/etdump.h>
#include <executorch/sdk/etdump/etdump_gen.h>

using namespace torch::executor;

ETDump::ETDump(MemoryAllocator& memory_allocator)
    : allocator_(memory_allocator) {}

torch::executor::Error ETDump::serialize_prof_results_to_etdump(
    const char* path) {
  prof_result_t prof_result;
  EXECUTORCH_DUMP_PROFILE_RESULTS(&prof_result);

  if (!et_dump_gen) {
    et_dump_gen = allocator_.allocateInstance<ETDumpGen>();
    if (!et_dump_gen) {
      ET_LOG(Error, "Failed to allocate space for ETDumpGen");
      return torch::executor::Error::MemoryAllocationFailed;
    }
    new (et_dump_gen) ETDumpGen(allocator_);
  }
  for (size_t i = 0; i < prof_result.num_blocks; i++) {
    et_dump_gen->CreateProfileBlockEntry(
        (prof_header_t*)((uintptr_t)prof_result.prof_data + prof_buf_size * i));
  }

  et_dump_gen->generate_etdump();
  FILE* fp = fopen(path, "wb");
  if (!fp) {
    ET_LOG(Error, "Failed to open file for writing out etdump");
    return torch::executor::Error::AccessFailed;
  }

  auto ret = fwrite(
      et_dump_gen->get_etdump_data(), et_dump_gen->get_etdump_size(), 1, fp);
  if (ret != 1) {
    ET_LOG(Error, "Failed to write out etdump data");
    return torch::executor::Error::AccessFailed;
  }

  return torch::executor::Error::Ok;
}
