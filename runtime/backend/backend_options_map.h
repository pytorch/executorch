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
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/platform/assert.h>

#include <cstring>

namespace executorch {
namespace runtime {

/**
 * Maps backend IDs to their load-time options.
 *
 * This class is used to provide per-delegate configuration at Module::load()
 * time. Users can set options for multiple backends, and the runtime will
 * route the appropriate options to each backend during initialization.
 *
 * Example usage:
 * @code
 *   BackendOptions<4> coreml_opts;
 *   coreml_opts.set_option("compute_unit", "cpu_and_gpu");
 *
 *   LoadBackendOptionsMap map;
 *   map.set_options("CoreMLBackend", coreml_opts.view());
 *
 *   // Later, during backend init:
 *   auto opts = map.get_options("CoreMLBackend");
 * @endcode
 *
 * Note: This class does NOT take ownership of the option spans. The caller
 * must ensure that the BackendOptions objects outlive the LoadBackendOptionsMap
 * and any loaded models that use it.
 */
class LoadBackendOptionsMap final {
 public:
  /**
   * Default constructor - creates an empty map.
   */
  LoadBackendOptionsMap() : size_(0) {
    for (size_t i = 0; i < kMaxBackends; ++i) {
      entries_[i].backend_id[0] = '\0';
    }
  }

  /**
   * Sets options for a specific backend.
   *
   * If options for the given backend_id already exist, they will be replaced.
   *
   * @param backend_id The backend identifier (e.g., "CoreMLBackend",
   * "XNNPACKBackend"). Must not be null or empty.
   * @param options Span of BackendOption to associate with this backend.
   *                The span's underlying data must outlive this map and any
   *                models loaded with it.
   * @return Error::Ok on success.
   *         Error::InvalidArgument if backend_id is null/empty or max backends
   * exceeded.
   */
  Error set_options(const char* backend_id, Span<BackendOption> options) {
    if (backend_id == nullptr || backend_id[0] == '\0') {
      return Error::InvalidArgument;
    }

    return set_options_impl(backend_id, options);
  }

  /**
   * Sets options from a backend options builder.
   *
   * This convenience overload accepts any builder type that provides
   * backend_id() and view() methods, allowing simpler usage:
   *
   * @code
   *   ExampleBackendOptions opts;
   *   opts.setNumThreads(4).setEnableOptimization(true);
   *   map.set_options(opts);
   * @endcode
   *
   * @param builder A backend options builder with backend_id() and view()
   * methods.
   * @return Error::Ok on success, Error::InvalidArgument on failure.
   */
  template <typename Builder>
  Error set_options(Builder& builder) {
    return set_options_impl(builder.backend_id(), builder.view());
  }

 private:
  Error set_options_impl(const char* backend_id, Span<BackendOption> options) {
    // Check if backend already exists and update it
    for (size_t i = 0; i < size_; ++i) {
      if (std::strcmp(entries_[i].backend_id, backend_id) == 0) {
        entries_[i].options = options;
        return Error::Ok;
      }
    }

    // Add new entry if space available
    if (size_ >= kMaxBackends) {
      return Error::InvalidArgument;
    }

    const size_t id_len = std::strlen(backend_id);
    if (id_len >= kMaxBackendIdLength) {
      return Error::InvalidArgument;
    }
    std::memcpy(entries_[size_].backend_id, backend_id, id_len);
    entries_[size_].backend_id[id_len] = '\0';
    entries_[size_].options = options;
    ++size_;

    return Error::Ok;
  }

 public:
  /**
   * Gets options for a specific backend.
   *
   * @param backend_id The backend identifier to look up.
   * @return Span of options for this backend, or an empty span if the backend
   *         has no options configured or backend_id is null.
   */
  Span<const BackendOption> get_options(const char* backend_id) const {
    if (backend_id == nullptr) {
      return Span<const BackendOption>(nullptr, static_cast<size_t>(0));
    }

    for (size_t i = 0; i < size_; ++i) {
      if (std::strcmp(entries_[i].backend_id, backend_id) == 0) {
        return Span<const BackendOption>(
            entries_[i].options.data(), entries_[i].options.size());
      }
    }

    return Span<const BackendOption>(nullptr, static_cast<size_t>(0));
  }

  /**
   * Checks if options have been configured for a specific backend.
   *
   * @param backend_id The backend identifier to check.
   * @return true if options are set for this backend, false otherwise.
   */
  bool has_options(const char* backend_id) const {
    if (backend_id == nullptr) {
      return false;
    }

    for (size_t i = 0; i < size_; ++i) {
      if (std::strcmp(entries_[i].backend_id, backend_id) == 0) {
        return true;
      }
    }

    return false;
  }

  /**
   * Returns the number of backends with configured options.
   */
  size_t size() const {
    return size_;
  }

  /**
   * Non-owning view of a single (backend_id, options) entry, returned by
   * entry_at(). The pointer / span are valid until the map is mutated or
   * destroyed.
   */
  struct EntryView {
    const char* backend_id = nullptr;
    Span<const BackendOption> options;
  };

  /**
   * Returns the (backend_id, options) entry at the given index for
   * enumeration over the map's contents.
   *
   * @param index The entry index. Must be < size(); behavior is undefined
   *     otherwise. Use this together with size() to walk every entry.
   * @return EntryView referencing the entry's backend_id and options. The
   *     view is valid until the next mutation of, or destruction of, this
   *     map.
   *
   * Example:
   * @code
   *   for (size_t i = 0; i < map.size(); ++i) {
   *     const auto entry = map.entry_at(i);
   *     // use entry.backend_id and entry.options ...
   *   }
   * @endcode
   */
  EntryView entry_at(size_t index) const {
    ET_DCHECK_MSG(
        index < size_,
        "entry_at index %zu out of bounds (size=%zu)",
        index,
        size_);
    return EntryView{
        entries_[index].backend_id,
        Span<const BackendOption>(
            entries_[index].options.data(), entries_[index].options.size())};
  }

 private:
  static constexpr size_t kMaxBackends = 8;
  static constexpr size_t kMaxBackendIdLength = 64;

  struct Entry {
    char backend_id[kMaxBackendIdLength];
    Span<BackendOption> options;
  };

  Entry entries_[kMaxBackends];
  size_t size_;
};

} // namespace runtime
} // namespace executorch
