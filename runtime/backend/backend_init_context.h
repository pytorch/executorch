/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/runtime/core/memory_allocator.h>

namespace executorch {
namespace runtime {

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

} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::BackendInitContext;
} // namespace executor
} // namespace torch
