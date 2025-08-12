/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/runtime/executor/method_meta.h>
#include <cstddef>

namespace example {
/**
 * @class IMemAlloc
 * @brief Interface for buffer allocation.
 */
class IMemAlloc {
 public:
  IMemAlloc(){};
  virtual ~IMemAlloc(){};
  virtual std::byte* allocate(size_t data_size) = 0;
  virtual void add_memory_info(
      void* data_ptr,
      size_t data_size,
      executorch::runtime::TensorInfo tensor_info) = 0;
};

} // namespace example
