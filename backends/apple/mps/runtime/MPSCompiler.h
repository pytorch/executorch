//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#pragma once

#include <executorch/backends/apple/mps/runtime/MPSExecutor.h>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/platform/compiler.h>

#include <memory>
#include <vector>

namespace torch {
namespace executor {
namespace mps {
namespace delegate {

class MPSCompiler {
 public:
  // Takes Flatbuffer Serialized MPS Model and rebuilds the MPSGraphExecutable
  // returns an executor object that holds the MPS runtime object which we
  // can then use to set inputs and run inference using the MPSGraphExecutable.
  ET_NODISCARD static Error compileModel(
      const void* buffer_pointer,
      size_t num_bytes,
      MPSExecutor* executor,
      MemoryAllocator* runtime_allocator,
      ArrayRef<CompileSpec> compile_specs);
};

} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
