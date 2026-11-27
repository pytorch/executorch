/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Puts LLM runner I/O buffers (KV cache, mask, logits) in rpcmem, so HTP reads
// and writes them in place. Linking this library is enough: it registers
// itself as the runner's I/O buffer allocator. Programs must be lowered with
// shared buffers enabled for QNN to use the memory without copying.

#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/extension/llm/runner/io_manager/io_manager.h>

namespace executorch {
namespace backends {
namespace qnn {
namespace {

using ::executorch::extension::llm::IOBufferAllocator;

class SharedBufferAllocator final : public IOBufferAllocator {
 public:
  void* allocate(size_t nbytes, size_t alignment) override {
    return QnnExecuTorchAllocCustomMem(nbytes, alignment);
  }
  void deallocate(void* block) override {
    QnnExecuTorchFreeCustomMem(block);
  }
  void bind(void* block, void* tensor_data) override {
    QnnExecuTorchAddCustomMemTensorAddr(tensor_data, block);
  }
};

const bool registered = ::executorch::extension::llm::set_io_buffer_allocator(
    [] { return std::make_unique<SharedBufferAllocator>(); });

} // namespace
} // namespace qnn
} // namespace backends
} // namespace executorch
