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
#include <executorch/runtime/core/error.h>

#include <vector>

using namespace torch::executor::enn;

namespace executorch {
namespace backends {
namespace enn {
namespace shared_memory_manager {

class SharedMemoryManager {
 public:
  static SharedMemoryManager* getInstance();

  SharedMemoryManager() = default;
  ~SharedMemoryManager() = default;
  SharedMemoryManager(const SharedMemoryManager&) = delete;
  SharedMemoryManager& operator=(const SharedMemoryManager&) = delete;
  SharedMemoryManager(SharedMemoryManager&&) = delete;
  SharedMemoryManager& operator=(SharedMemoryManager&&) = delete;

  void* alloc(const size_t size);
  void free(void* ptr);
  void free(void* ptr, std::align_val_t alignment);
  bool query(EnnBufferPtr* out, const void* ptr, const size_t size);

 private:
  std::vector<EnnBufferPtr> EnnBufferPtrList;
};

} // namespace shared_memory_manager
} // namespace enn
} // namespace backends
} // namespace executorch
