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

#ifdef __GNUC__
// Disable -Wdeprecated-declarations, as some builds use 'Werror'.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

namespace executorch {
namespace ET_RUNTIME_NAMESPACE {
/**
 * BackendInitContext will be used to inject runtime info for to initialize
 * delegate.
 */
class BackendInitContext final {
 public:
  explicit BackendInitContext(
      MemoryAllocator* runtime_allocator,
      size_t method_id,
      EventTracer* event_tracer = nullptr,
      const char* method_name = nullptr,
      const NamedDataMap* named_data_map = nullptr)
      : runtime_allocator_(runtime_allocator),
#ifdef ET_EVENT_TRACER_ENABLED
        event_tracer_(event_tracer),
#else
        event_tracer_(nullptr),
#endif
        method_name_(method_name),
        method_id_(method_id),
        named_data_map_(named_data_map) {
  }

  /** Get the runtime allocator passed from Method. It's the same runtime
   * executor used by the standard executor runtime and the life span is the
   * same as the model.
   */
  MemoryAllocator* get_runtime_allocator() {
    return runtime_allocator_;
  }

  /**
   * Returns a pointer (null if not installed) to an instance of EventTracer to
   * do profiling/debugging logging inside the delegate backend. Users will need
   * access to this pointer to use any of the event tracer APIs.
   */
  EventTracer* event_tracer() {
    return event_tracer_;
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
  
  /** Get the method identifier of the loaded method. This corresponds to the
   * id() method on the Method object, and can be used to disambiguate between
   * different methods with the same name.
   */
  size_t method_id() const {
    return method_id_;
  }

  /** Get the named data map from ExecuTorch runtime.
   * This provides a way for backends to retrieve data blobs by key.
   */
  const NamedDataMap* get_named_data_map() const {
    return named_data_map_;
  }

 private:
  MemoryAllocator* runtime_allocator_ = nullptr;
  EventTracer* event_tracer_ = nullptr;
  const char* method_name_ = nullptr;
  size_t method_id_ = 0;
  const NamedDataMap* named_data_map_ = nullptr;
};

} // namespace ET_RUNTIME_NAMESPACE
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::ET_RUNTIME_NAMESPACE::BackendInitContext;
} // namespace executor
} // namespace torch
