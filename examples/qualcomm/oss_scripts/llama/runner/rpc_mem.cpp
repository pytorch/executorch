/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/rpc_mem.h>
#include <executorch/runtime/core/memory_allocator.h>
using executorch::runtime::MemoryAllocator;
using executorch::runtime::TensorInfo;

namespace example {
RpcMem::RpcMem(
    const size_t total_cache_size,
    const size_t total_prompt_processor_io_size,
    const size_t total_token_generator_io_size)
    : calculated_offsets_(0) {
  size_t total_bytes = total_cache_size + total_prompt_processor_io_size +
      total_token_generator_io_size;
  shared_buffer_base_ptr_ = QnnExecuTorchAllocCustomMem(
      total_bytes, MemoryAllocator::kDefaultAlignment);
}
RpcMem::RpcMem(
    const size_t total_cache_size,
    const size_t total_prompt_processor_io_size,
    const size_t total_token_generator_io_size,
    const size_t total_embedding_processor_io_size,
    const size_t total_embedding_generator_io_size)
    : calculated_offsets_(0) {
  size_t total_bytes = total_cache_size + total_prompt_processor_io_size +
      total_token_generator_io_size + total_embedding_processor_io_size +
      total_embedding_generator_io_size;
  shared_buffer_base_ptr_ = QnnExecuTorchAllocCustomMem(
      total_bytes, MemoryAllocator::kDefaultAlignment);
}
RpcMem::~RpcMem() {
  QnnExecuTorchFreeCustomMem(shared_buffer_base_ptr_);
}

std::byte* RpcMem::allocate(size_t data_size) {
  std::byte* data_ptr = static_cast<std::byte*>(shared_buffer_base_ptr_);
  data_ptr += calculated_offsets_;
  // Record the position of the data pointer
  io_pos_map_[data_ptr] = calculated_offsets_;
  calculated_offsets_ += data_size;
  return data_ptr;
}

void RpcMem::add_memory_info(
    void* data_ptr,
    size_t data_size,
    TensorInfo tensor_info) {
  if (auto it = io_pos_map_.find(static_cast<std::byte*>(data_ptr));
      it == io_pos_map_.end()) {
    ET_LOG(Error, "Shared buffer pointer %p is not found", data_ptr);
  }
  if (binded_tensor_addr_set_.find(data_ptr) == binded_tensor_addr_set_.end()) {
    QnnExecuTorchAddCustomMemTensorAddr(data_ptr, shared_buffer_base_ptr_);
    binded_tensor_addr_set_.insert(data_ptr);
  }
};

} // namespace example
