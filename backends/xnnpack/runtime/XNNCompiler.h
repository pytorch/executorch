// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

#include <executorch/backends/xnnpack/runtime/XNNExecutor.h>
#include <executorch/compiler/Compiler.h>
#include <xnnpack.h>
#include <memory>
#include <vector>

namespace torch {
namespace executor {
namespace xnnpack {
namespace delegate {

class XNNCompiler {
 public:
  // Takes Flatbuffer Serialized XNNPack Model and rebuilds the xnn-subgraph
  // returns an executor object that holds the xnn runtime object which we
  // can then use to set inputs and run inference using the xnn graph.
  __ET_NODISCARD static Error compileModel(
      const void* buffer_pointer,
      size_t num_bytes,
      XNNExecutor* executor,
      MemoryAllocator* runtime_allocator);
};

} // namespace delegate
} // namespace xnnpack
} // namespace executor
} // namespace torch
