/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/imem_alloc.h>
#include <vector>

namespace example {
/**
 * @class ClientMem
 * @brief Final class for client buffer allocation, implementing IBufferAlloc
 * interface. This is specifically designed for use cases without shared buffer.
 */
class ClientMem final : public IMemAlloc {
 public:
  ClientMem(){};
  // Disable copy constructors, r-value referencing, etc
  ClientMem(const ClientMem&) = delete;
  ClientMem& operator=(const ClientMem&) = delete;
  ClientMem(ClientMem&&) = delete;
  ClientMem& operator=(ClientMem&&) = delete;
  virtual ~ClientMem(){};
  /**
   * @brief Allocate buffer of specified size with vector.
   * @param data_size Size of the data to allocate.
   * @return Pointer to the allocated buffer.
   */
  std::byte* allocate(size_t data_size) override {
    allocated_buffers_.push_back(std::vector<std::byte>(data_size));
    return allocated_buffers_.back().data();
  };
  // Only used for SMART_MASK mode
  void add_memory_info(
      void* data_ptr,
      size_t data_size,
      executorch::runtime::TensorInfo tensor_info) override {};

 private:
  std::vector<std::vector<std::byte>> allocated_buffers_;
};

} // namespace example
