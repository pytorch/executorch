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
  size_t pos = io_pos_map_[static_cast<std::byte*>(data_ptr)];
  uint32_t* shape = const_cast<uint32_t*>(
      reinterpret_cast<const uint32_t*>(tensor_info.sizes().data()));
  uint32_t rank = static_cast<uint32_t>(tensor_info.sizes().size());
  executorch::aten::ScalarType scalar_type = tensor_info.scalar_type();
  CustomMemTensorInfo info = {
      shared_buffer_base_ptr_,
      data_ptr,
      pos,
      data_size,
      shape,
      rank,
      scalar_type};
  QnnExecuTorchAddCustomMemTensorInfo(info);
};

} // namespace example
