/*
 *  Copyright (c) 2025 Samsung Electronics Co. LTD
 *  All rights reserved
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 */
#pragma once

#include <executorch/backends/samsung/runtime/enn_api_implementation.h>
#include <executorch/backends/samsung/runtime/enn_type.h>

#include <vector>

namespace torch {
namespace executor {
namespace enn {

struct DataBuffer {
  void* buf_ptr_ = nullptr;
  size_t size_ = 0;
};

class EnnExecutor {
 public:
  Error initialize(const char* binary_buf_addr, size_t buf_size);

  Error eval(
      const std::vector<DataBuffer>& inputs,
      const std::vector<DataBuffer>& outputs);

  int32_t getInputSize() const {
    return num_of_inputs_;
  }
  int32_t getOutputSize() const {
    return num_of_outputs_;
  }

  ~EnnExecutor();

 private:
  EnnModelId model_id_;
  EnnBufferPtr* alloc_buffer_ = nullptr;
  int32_t num_of_inputs_ = 0;
  int32_t num_of_outputs_ = 0;
};

} // namespace enn
} // namespace executor
} // namespace torch
