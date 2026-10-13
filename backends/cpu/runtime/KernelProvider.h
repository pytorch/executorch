// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/native/runtime/graph/Graph.h>
#include <executorch/runtime/backend/interface.h>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

struct pthreadpool;

namespace executorch::backends::cpu {

inline constexpr size_t kBufferAlignment = 64;
inline constexpr size_t kReadableTail = 64;

/// Extents start at the bound pointer. Alignment must be a power of two.
struct BufferRequirements {
  size_t readable_bytes = 0;
  size_t writable_bytes = 0;
  size_t alignment = kBufferAlignment;

  bool valid() const;
  void merge(const BufferRequirements& other);
};

runtime::Result<BufferRequirements> tensor_requirements(
    size_t tensor_bytes,
    bool writable,
    size_t alignment = kBufferAlignment,
    size_t readable_tail = kReadableTail);

enum class StorageOwner { Borrowed, ConstantCopy, Activation, Scratch };

/// Capacities are measured from data. Readable bytes are initialized; writes
/// require their own declared extent. The owner retains storage through run.
struct Buffer {
  void* data = nullptr;
  size_t readable_bytes = 0;
  size_t writable_bytes = 0;
  size_t alignment = 0;
  StorageOwner owner = StorageOwner::Borrowed;

  bool accepts(size_t tensor_bytes, bool writable) const;
  bool accepts(const BufferRequirements& requirements) const;
};

runtime::Result<Buffer> allocate_buffer(
    runtime::MemoryAllocator& allocator,
    size_t tensor_bytes);
runtime::Result<Buffer> allocate_buffer(
    runtime::MemoryAllocator& allocator,
    const BufferRequirements& requirements);

/// Captured by the host, never discovered through implicit provider state.
/// Threadpool lifetime covers preparation and every invocation of the plan.
struct ExecutionContext {
  size_t threads = 1;
  bool avx2_fma = false;
  pthreadpool* threadpool = nullptr;
  runtime::EventTracer* event_tracer = nullptr;
};

struct Support {
  bool supported;
  std::string_view reason;
};

using Kernel = ptn::Node;
using KernelId = ptn::NodeId;

/// Native IDs remain stable when runtime implementation selection changes.
struct KernelRegion {
  std::vector<KernelId> nodes;
  std::vector<ptn::ValueId> inputs;
  std::vector<ptn::ValueId> outputs;
};

struct ValueRequirement {
  ptn::ValueId id = ptn::kInvalid;
  BufferRequirements buffer;
};

/// Scratch is invocation-local and reusable immediately after run returns.
/// Provider-private library workspace remains separately owned and accounted.
struct StorageRequirements {
  std::vector<ValueRequirement> values;
  BufferRequirements scratch;
};

runtime::Result<size_t> tensor_bytes(const ptn::Value& value);

/// Borrowed model state; outlives every provider executable compiled from it.
struct PreparationContext {
  const ptn::Graph& graph;
  const std::vector<Buffer>& buffers;
  runtime::MemoryAllocator& allocator;
  size_t& private_constant_bytes;
  ExecutionContext execution;
};

/// Synchronous executable. reshape never executes numerical kernels. bind must
/// validate current storage, including when shapes have not changed.
class Executable {
 public:
  virtual ~Executable() = default;
  virtual runtime::Error reshape() = 0;
  virtual runtime::Error bind(const Buffer& scratch = {}) = 0;
  virtual runtime::Error run(const ExecutionContext& context) = 0;
};

class KernelImplementation {
 public:
  virtual ~KernelImplementation() = default;
  virtual std::string_view name() const = 0;
  /// Among eligible implementations matching the same preference entry, the
  /// highest priority wins. The ET registry is 0 and XNNPACK is 100.
  virtual int baseline_priority() const = 0;
  virtual Support supports(
      const Kernel& kernel,
      const ptn::Graph& graph,
      const ExecutionContext& context) const = 0;
  virtual bool accepts_regions() const {
    return false;
  }
  /// Pure metadata query, before compile can read constants or allocate
  /// storage. The default declares contiguous FP32 storage for every region
  /// boundary.
  virtual runtime::Result<StorageRequirements> requirements(
      const KernelRegion& region,
      const ptn::Graph& graph,
      const ExecutionContext& context) const;
  virtual runtime::Result<std::unique_ptr<Executable>> compile(
      const KernelRegion& region,
      PreparationContext& context) = 0;
};

/// Providers and their implementations are model-owned. The linked factory list
/// is immutable; no provider assignment is serialized into the Native graph.
class KernelProvider {
 public:
  virtual ~KernelProvider() = default;
  virtual std::string_view name() const = 0;
  virtual std::vector<KernelImplementation*> implementations() = 0;
  /// Called after all regions compile and before any executable is reshaped.
  virtual runtime::Error finish() {
    return runtime::Error::Ok;
  }
};

using ProviderFactory = std::unique_ptr<KernelProvider> (*)();

struct KernelPreference {
  std::string provider;
  std::string implementation;
};

struct RuntimeConfiguration {
  runtime::Span<const ProviderFactory> providers;
  /// First matching entry wins; an empty implementation matches the provider.
  std::vector<KernelPreference> preferences;
  bool force = false;

  runtime::Error apply_options(const runtime::BackendInitContext& context);
};

/// Per-model Module::load options for CpuBackend; omitted keys use build
/// defaults. preference_count replaces the list using preferred_provider_N and
/// optional preferred_implementation_N entries, starting at N = 0.
inline constexpr std::string_view kPreferenceCountOption = "preference_count";
inline constexpr std::string_view kPreferredProviderOption =
    "preferred_provider";
inline constexpr std::string_view kPreferredImplementationOption =
    "preferred_implementation";
inline constexpr std::string_view kForceOption = "force";

using RuntimeConfigurationFactory = RuntimeConfiguration (*)();

/// Called during static initialization by the application's cpu_backend()
/// build composition.
void register_runtime_configuration(RuntimeConfigurationFactory factory);

} // namespace executorch::backends::cpu
