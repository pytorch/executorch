#pragma once

#include <executorch/executor/MemoryAllocator.h>
#include <executorch/runtime/core/error.h>

class ETDumpGen;

// ETDump class that enables users to generate an etdump from
// the profiling results currently stored in the profiling
// buffer.
// ETDump that will be written out is a flatbuffer defined by
// the schema present in etdump_schema.fbs.
class ETDump {
 public:
  /**
   * Constructor of the ETDump class expects a MemoryAllocator to
   * be passed in from which all the allocations needed for etdump
   * generation will be done.
   */
  explicit ETDump(torch::executor::MemoryAllocator& memory_allocator);

  /*
   * Serialize out all the profiling results stored in the
   * profiling buffer to an etdump at the specified path.
   */
  torch::executor::Error serialize_prof_results_to_etdump(const char* path);

 private:
  ETDumpGen* et_dump_gen = nullptr;
  torch::executor::MemoryAllocator allocator_;
};
