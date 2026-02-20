/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/imem_alloc.h>
#include <unordered_map>
#include <unordered_set>

namespace example {
/**
 * @class RpcMem
 * @brief Final class for rpc memory allocation, implementing IMemAlloc
 * interface. Used for SMART_MASK mode.
 */
class RpcMem final : public IMemAlloc {
 public:
  RpcMem(
      const size_t total_cache_size,
      const size_t total_prompt_processor_io_size,
      const size_t total_token_generator_io_size);
  /**
   * @brief Constructor to allocate RpcMem with total sizes.
   * @param total_cache_size Total size of the cache.
   * @param total_prompt_processor_io_size Total size for prompt processor I/O.
   * @param total_token_generator_io_size Total size for token generator I/O.
   * @param total_embedding_processor_io_size Total size for embedding prompt
processor I/O.
   * @param total_embedding_generator_io_size Total size for embedding generator
I/O.    */
  RpcMem(
      const size_t total_cache_size,
      const size_t total_prompt_processor_io_size,
      const size_t total_token_generator_io_size,
      const size_t total_embedding_processor_io_size,
      const size_t total_embedding_generator_io_size);

  // Disable copy constructors, r-value referencing, etc
  RpcMem(const RpcMem&) = delete;
  RpcMem& operator=(const RpcMem&) = delete;
  RpcMem(RpcMem&&) = delete;
  RpcMem& operator=(RpcMem&&) = delete;
  virtual ~RpcMem();
  /**
   * @brief Allocate buffer of specified size with shared_buffer_base_ptr_.
   * @param data_size Size of the data to allocate.
   * @return Pointer to the allocated buffer.
   */
  std::byte* allocate(size_t size) override;

  /**
   * @brief Add memory information into QNN Backend to register RpcMem to the
tensor.
   * @param data_ptr Pointer to the data.
   * @param data_size Size of the data.
   * @param tensor_info Information about the tensor.
   */
  void add_memory_info(
      void* data_ptr,
      size_t data_size,
      executorch::runtime::TensorInfo tensor_info) override;

 private:
  // shared buffer
  void* shared_buffer_base_ptr_;
  size_t calculated_offsets_;
  std::unordered_map<std::byte*, size_t> io_pos_map_;
  std::unordered_set<void*> binded_tensor_addr_set_;
};

} // namespace example
