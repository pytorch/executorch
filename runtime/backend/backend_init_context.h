/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/runtime/core/memory_allocator.h>

namespace torch {
namespace executor {

/**
 * BackendInitContext will be used to inject runtime info for to initialize
 * delegate.
 */
class BackendInitContext final {
 public:
  explicit BackendInitContext(MemoryAllocator* runtime_allocator)
      : runtime_allocator_(runtime_allocator) {}

  /** Get the runtime allocator passed from Method. It's the same runtime
   * executor used by the standard executor runtime and the life span is the
   * same as the model.
   */
  MemoryAllocator* get_runtime_allocator() {
    return runtime_allocator_;
  }

 private:
  MemoryAllocator* runtime_allocator_ = nullptr;
};

} // namespace executor
} // namespace torch
