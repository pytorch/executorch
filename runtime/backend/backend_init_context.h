/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/event_tracer.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/core/named_data_map.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>

#include <cstring>

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
      EventTracer* event_tracer = nullptr,
      const char* method_name = nullptr,
      const NamedDataMap* named_data_map = nullptr,
      Span<const BackendOption> runtime_specs = {})
      : runtime_allocator_(runtime_allocator),
#ifdef ET_EVENT_TRACER_ENABLED
        event_tracer_(event_tracer),
#else
        event_tracer_(nullptr),
#endif
        method_name_(method_name),
        named_data_map_(named_data_map),
        runtime_specs_(runtime_specs) {
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

  /** Get the named data map from ExecuTorch runtime.
   * This provides a way for backends to retrieve data blobs by key.
   */
  const NamedDataMap* get_named_data_map() const {
    return named_data_map_;
  }

  /**
   * Get the runtime specs (load-time options) for this backend.
   * These are per-delegate options passed at Module::load() time.
   *
   * @return Span of BackendOption containing the runtime specs, or empty span
   *         if no runtime specs were provided.
   */
  Span<const BackendOption> runtime_specs() const {
    return runtime_specs_;
  }

  /**
   * Get a runtime spec value by key and type.
   *
   * @tparam T The expected type (bool, int, or const char*)
   * @param key The option key to look up.
   * @return Result containing the value if found and type matches,
   *         Error::NotFound if key doesn't exist,
   *         Error::InvalidArgument if key exists but type doesn't match.
   */
  template <typename T>
  Result<T> get_runtime_spec(const char* key) const {
    static_assert(
        std::is_same_v<T, bool> || std::is_same_v<T, int> ||
            std::is_same_v<T, const char*>,
        "get_runtime_spec<T> only supports bool, int, and const char*");

    for (size_t i = 0; i < runtime_specs_.size(); ++i) {
      const auto& opt = runtime_specs_[i];
      if (std::strcmp(opt.key, key) == 0) {
        if constexpr (std::is_same_v<T, const char*>) {
          if (auto* arr = std::get_if<std::array<char, kMaxOptionValueLength>>(
                  &opt.value)) {
            return arr->data();
          }
        } else {
          if (auto* val = std::get_if<T>(&opt.value)) {
            return *val;
          }
        }
        return Error::InvalidArgument;
      }
    }
    return Error::NotFound;
  }

 private:
  MemoryAllocator* runtime_allocator_ = nullptr;
  EventTracer* event_tracer_ = nullptr;
  const char* method_name_ = nullptr;
  const NamedDataMap* named_data_map_ = nullptr;
  Span<const BackendOption> runtime_specs_;
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
