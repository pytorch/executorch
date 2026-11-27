/*
 *  Copyright (c) 2025 Samsung Electronics Co. LTD
 *  All rights reserved
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 */
#pragma once

#include <executorch/backends/samsung/runtime/enn_executor.h>
#include <executorch/runtime/core/memory_allocator.h>

#include <new>

namespace torch {
namespace executor {
namespace enn {

inline EnnExecutor* allocateEnnExecutor(MemoryAllocator& allocator) {
  auto* storage = allocator.allocateInstance<EnnExecutor>();
  if (storage == nullptr) {
    return nullptr;
  }
  return new (storage) EnnExecutor();
}

} // namespace enn
} // namespace executor
} // namespace torch
