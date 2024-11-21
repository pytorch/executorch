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
  explicit BackendInitContext(
      MemoryAllocator* runtime_allocator,
      const char* method_name = nullptr)
      : runtime_allocator_(runtime_allocator), method_name_(method_name) {}

  /** Get the runtime allocator passed from Method. It's the same runtime
   * executor used by the standard executor runtime and the life span is the
   * same as the model.
   */
  MemoryAllocator* get_runtime_allocator() {
    return runtime_allocator_;
  }

  /** Get the loaded method name from ExecuTorch runtime. Usually it's
   * "forward", however, if there are multiple methods in the .pte file, it can
   * be different. One example is that we may have prefill and decode methods in
   * the same .pte file. In this case, when client loads "prefill" method, the
   * `get_method_name` function will return "prefill", when client loads
   * "decode" method, the `get_method_name` function will return "decode".
   */
  const char* get_method_name() const {
    return method_name_;
  }

 private:
  MemoryAllocator* runtime_allocator_ = nullptr;
  const char* method_name_ = nullptr;
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
