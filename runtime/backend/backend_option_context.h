/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/runtime/core/event_tracer.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/core/named_data_map.h>

namespace executorch {
namespace ET_RUNTIME_NAMESPACE {
/**
 * BackendOptionContext will be used to inject runtime info for to initialize
 * delegate.
 */
class BackendOptionContext final {
 public:
  explicit BackendOptionContext() {}
};

} // namespace ET_RUNTIME_NAMESPACE
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::ET_RUNTIME_NAMESPACE::BackendOptionContext;
} // namespace executor
} // namespace torch
