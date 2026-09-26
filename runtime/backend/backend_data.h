/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include <executorch/runtime/core/array_ref.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/event_tracer.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/core/named_data_map.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/platform/compiler.h>

namespace executorch {
namespace ET_RUNTIME_NAMESPACE {

struct SizedBuffer {
  void* buffer;
  size_t nbytes;
};

struct CompileSpec {
  const char* key;
  SizedBuffer value;
};

/** Backend-produced bytes and their required file alignment. */
struct BackendData {
  /** Bytes to persist. They must be allocated by the context's allocator. */
  Span<const uint8_t> bytes;

  /** Required absolute file alignment, or nullopt to preserve the original. */
  std::optional<size_t> alignment;
};

/** One entry in a replacement named-data table. */
struct NamedBackendData {
  /** Existing named-data key whose value will be replaced. */
  const char* key;

  /** Replacement bytes and their alignment requirement. */
  BackendData data;
};

/** One lazily loaded processed-data input for backend initialization. */
struct BackendDataInput {
  /** One method that references this input. Valid through the next call. */
  const char* method_name;

  /** Original delegate bytes, borrowed until the next call or hook return. */
  Span<const uint8_t> processed;

  /** Compile specifications associated with the delegate bytes. */
  ArrayRef<CompileSpec> compile_specs;
};

/**
 * Model-scoped context for one-time backend-data initialization.
 *
 * The context supplies one processed delegate blob at a time so that a backend
 * never needs to materialize all model data in memory. It also provides the
 * ET-owned temporary allocator from which all emitted bytes and key strings
 * must be allocated. The extension may reset that allocator after output has
 * been consumed and before the next input is returned.
 */
class BackendDataInitContext {
 public:
  /**
   * Constructs the common state exposed to a backend.
   *
   * @param[in] temp_allocator Resettable allocator owned by the caller. It
   *     must outlive this context.
   * @param[in] event_tracer Optional event tracer for this operation.
   * @param[in] named_data_map Combined read-only named data visible to the
   *     model, or nullptr when the model has no named data.
   */
  BackendDataInitContext(
      MemoryAllocator* temp_allocator,
      EventTracer* event_tracer,
      const NamedDataMap* named_data_map)
      : temp_allocator_(temp_allocator),
        event_tracer_(event_tracer),
        named_data_map_(named_data_map) {}

  virtual ~BackendDataInitContext() = default;

  /**
   * Lazily loads the next unique backend blob, or nullopt at end.
   *
   * Every returned field remains valid until the next call or until
   * initialize_backend_data() returns.
   */
  ET_NODISCARD virtual Result<std::optional<BackendDataInput>>
  next_backend_data() = 0;

  /** Returns the resettable allocator that must own submitted output data. */
  MemoryAllocator* get_temp_allocator() const {
    return temp_allocator_;
  }

  /**
   * Allocates temporary output or scratch storage.
   *
   * @param[in] size Number of bytes to allocate.
   * @param[in] alignment Required power-of-two memory alignment.
   * @returns Allocated storage, or nullptr when the allocation cannot be met.
   */
  void* allocate(
      size_t size,
      size_t alignment = MemoryAllocator::kDefaultAlignment) {
    return temp_allocator_->allocate(size, alignment);
  }

  /** Returns the optional event tracer for this model-wide operation. */
  EventTracer* event_tracer() const {
    return event_tracer_;
  }

  /** Returns the read-only named-data map for the PTE. */
  const NamedDataMap* get_named_data_map() const {
    return named_data_map_;
  }

 private:
  MemoryAllocator* temp_allocator_;
  EventTracer* event_tracer_;
  const NamedDataMap* named_data_map_;
};

/**
 * Accepts backend outputs in logical data terms rather than file offsets.
 *
 * Calls consume their inputs synchronously. Every byte span and key string
 * must come from the BackendDataInitContext temporary allocator. The writer
 * copies accepted data into storage before returning.
 */
class BackendDataWriter {
 public:
  virtual ~BackendDataWriter() = default;

  /**
   * Replaces values for existing named-data keys.
   *
   * Entries may be submitted while any processed input is current or after
   * iteration ends. Keys cannot be added, removed, or renamed. An empty span
   * replaces the existing value with a zero-sized value.
   *
   * @param[in] data Named replacement values to consume synchronously.
   */
  ET_NODISCARD virtual Error write_named_data(
      Span<const NamedBackendData> data) = 0;
};

} // namespace ET_RUNTIME_NAMESPACE
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::ET_RUNTIME_NAMESPACE::BackendData;
using ::executorch::ET_RUNTIME_NAMESPACE::BackendDataInitContext;
using ::executorch::ET_RUNTIME_NAMESPACE::BackendDataInput;
using ::executorch::ET_RUNTIME_NAMESPACE::BackendDataWriter;
using ::executorch::ET_RUNTIME_NAMESPACE::CompileSpec;
using ::executorch::ET_RUNTIME_NAMESPACE::NamedBackendData;
using ::executorch::ET_RUNTIME_NAMESPACE::SizedBuffer;
} // namespace executor
} // namespace torch
