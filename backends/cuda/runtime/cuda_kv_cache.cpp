/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cuda/runtime/cuda_kv_cache.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include <executorch/backends/aoti/slim/c10/core/Device.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/aoti/slim/c10/cuda/Exception.h>
#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/backends/aoti/slim/factory/from_blob.h>
#include <executorch/backends/cuda/runtime/cuda_allocator.h>
#include <executorch/runtime/platform/log.h>

namespace executorch::backends::cuda {
namespace {

namespace aoti = ::executorch::backends::aoti;
namespace slimc10 = ::executorch::backends::aoti::slim::c10;
using ::executorch::backends::aoti::slim::from_blob;
using ::executorch::backends::aoti::slim::SlimTensor;
using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

struct Allocation {
  void* k{nullptr};
  void* v{nullptr};
  void* capacity_device{nullptr};
  int64_t capacity{0};
};

struct Descriptor {
  enum class Kind { Key, Value, Capacity };

  std::string internal_name;
  int64_t layer_id{0};
  Kind kind{Kind::Key};
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  slimc10::ScalarType dtype{slimc10::ScalarType::BFloat16};
  slimc10::Device device{slimc10::DeviceType::CUDA, 0};
};

struct Bound {
  std::vector<std::unique_ptr<SlimTensor>> tensors;
  std::vector<aoti::AOTInductorConstantMapEntry> pairs;
};

struct SessionState {
  OffGraphKVMetrics metrics;
  std::unordered_map<int64_t, Allocation> allocations;
  std::unordered_map<CudaDelegateHandle*, Bound> bound;
};

struct Context {
  OffGraphKVConfig config;
  int device{0};
  bool device_known{false};
  bool handles_associated{false};
  Error error{Error::Ok};
  std::unordered_map<CudaDelegateHandle*, std::unordered_map<std::string, Descriptor>>
      descriptors;
  std::unordered_set<std::string> discovered_fqns;
  int next_session_token{0};
  std::unordered_map<int, SessionState> sessions;
  std::unordered_map<CudaDelegateHandle*, int> bound_session;
};

struct Manager {
  std::mutex mutex;
  std::unordered_map<OffGraphKVContext, Context> contexts;
  std::unordered_map<CudaDelegateHandle*, OffGraphKVContext> handle_contexts;
  OffGraphKVContext next_context{1};
};

Manager& manager() {
  static Manager instance;
  return instance;
}

thread_local OffGraphKVContext loading_context = kInvalidOffGraphKVContext;
thread_local OffGraphKVContext active_context = kInvalidOffGraphKVContext;
thread_local int active_session_token = kNoOffGraphKVSession;

std::string fqn(int64_t layer_id, const char* suffix) {
  return "__et_offgraph_kv_layer_" + std::to_string(layer_id) + "_" + suffix;
}

size_t storage_bytes(
    const OffGraphKVConfig& config,
    const OffGraphKVLayerConfig& layer,
    int64_t capacity) {
  return static_cast<size_t>(layer.num_kv_heads) *
      static_cast<size_t>(capacity) * static_cast<size_t>(layer.head_dim) *
      config.element_size();
}

Error allocate_layer(
    Context& context,
    SessionState& session,
    const OffGraphKVLayerConfig& layer,
    int64_t capacity,
    cudaStream_t stream) {
  Allocation allocation;
  const size_t bytes = storage_bytes(context.config, layer, capacity);
  auto k = CudaAllocator::allocate_async(bytes, context.device, stream);
  ET_CHECK_OK_OR_RETURN_ERROR(k.error());
  allocation.k = k.get();
  auto v = CudaAllocator::allocate_async(bytes, context.device, stream);
  if (!v.ok()) {
    CudaAllocator::deallocate_async(allocation.k, context.device, stream);
    return v.error();
  }
  allocation.v = v.get();
  auto cap = CudaAllocator::allocate_async(
      sizeof(int64_t), context.device, stream);
  if (!cap.ok()) {
    CudaAllocator::deallocate_async(allocation.k, context.device, stream);
    CudaAllocator::deallocate_async(allocation.v, context.device, stream);
    return cap.error();
  }
  allocation.capacity_device = cap.get();
  allocation.capacity = capacity;
  const cudaError_t copy_error = cudaMemcpyAsync(
      allocation.capacity_device,
      &allocation.capacity,
      sizeof(int64_t),
      cudaMemcpyHostToDevice,
      stream);
  if (copy_error != cudaSuccess) {
    CudaAllocator::deallocate_async(allocation.k, context.device, stream);
    CudaAllocator::deallocate_async(allocation.v, context.device, stream);
    CudaAllocator::deallocate_async(
        allocation.capacity_device, context.device, stream);
    ET_LOG(
        Error,
        "offgraph_kv: capacity initialization failed: %s",
        cudaGetErrorString(copy_error));
    return Error::Internal;
  }
  session.metrics.allocated_bytes +=
      static_cast<int64_t>(2 * bytes + sizeof(int64_t));
  session.allocations.emplace(layer.layer_id, allocation);
  return Error::Ok;
}

Error ensure_initial_allocations(
    Context& context,
    SessionState& session,
    cudaStream_t stream) {
  bool allocated = false;
  for (const auto& layer : context.config.layers) {
    if (session.allocations.find(layer.layer_id) != session.allocations.end()) {
      continue;
    }
    const int64_t capacity = layer.policy == OffGraphKVPolicy::Ring
        ? layer.window * 2
        : context.config.initial_capacity;
    ET_CHECK_OK_OR_RETURN_ERROR(
        allocate_layer(context, session, layer, capacity, stream));
    allocated = true;
  }
  if (allocated) {
    session.metrics.flat_capacity = context.config.initial_capacity;
    ET_LOG(
        Info,
        "offgraph_kv: initialized flat_capacity=%lld allocated_bytes=%lld",
        static_cast<long long>(session.metrics.flat_capacity),
        static_cast<long long>(session.metrics.allocated_bytes));
  }
  return Error::Ok;
}

void release_allocation(Context& context, Allocation& allocation) {
  if (!context.device_known) {
    return;
  }
  CudaAllocator::deallocate_async(
      allocation.k, context.device, cudaStreamPerThread);
  CudaAllocator::deallocate_async(
      allocation.v, context.device, cudaStreamPerThread);
  CudaAllocator::deallocate_async(
      allocation.capacity_device, context.device, cudaStreamPerThread);
}

void reset_cuda_graphs(Context& context) {
  for (const auto& item : context.descriptors) {
    if (item.first->cuda_graph_state.phase != CudaGraphPhase::Disabled) {
      item.first->cuda_graph_state.reset_for_recapture();
    }
  }
}

Error grow_flat(
    Context& context,
    SessionState& session,
    int64_t new_capacity,
    cudaStream_t stream) {
  const int64_t old_capacity = session.metrics.flat_capacity;
  reset_cuda_graphs(context);
  for (const auto& layer : context.config.layers) {
    if (layer.policy != OffGraphKVPolicy::Flat) {
      continue;
    }
    auto old_it = session.allocations.find(layer.layer_id);
    ET_CHECK_OR_RETURN_ERROR(
        old_it != session.allocations.end(),
        InvalidState,
        "offgraph_kv: layer allocation is missing");
    Allocation old = old_it->second;
    session.allocations.erase(old_it);
    const Error allocation_error =
        allocate_layer(context, session, layer, new_capacity, stream);
    if (allocation_error != Error::Ok) {
      session.allocations.emplace(layer.layer_id, old);
      return allocation_error;
    }
    Allocation& replacement = session.allocations.at(layer.layer_id);
    const size_t row_bytes = static_cast<size_t>(old.capacity) *
        static_cast<size_t>(layer.head_dim) * context.config.element_size();
    const size_t new_pitch = static_cast<size_t>(new_capacity) *
        static_cast<size_t>(layer.head_dim) * context.config.element_size();
    for (const auto pair : {std::pair{replacement.k, old.k},
                            std::pair{replacement.v, old.v}}) {
      ET_CUDA_CHECK_OR_RETURN_ERROR(cudaMemcpy2DAsync(
          pair.first,
          new_pitch,
          pair.second,
          row_bytes,
          row_bytes,
          static_cast<size_t>(layer.num_kv_heads),
          cudaMemcpyDeviceToDevice,
          stream));
    }
    session.metrics.allocated_bytes -= static_cast<int64_t>(
        2 * storage_bytes(context.config, layer, old.capacity) +
        sizeof(int64_t));
    CudaAllocator::deallocate_async(old.k, context.device, stream);
    CudaAllocator::deallocate_async(old.v, context.device, stream);
    CudaAllocator::deallocate_async(
        old.capacity_device, context.device, stream);
  }
  session.metrics.flat_capacity = new_capacity;
  session.metrics.growth_count++;
  session.bound.clear();
  context.bound_session.clear();
  ET_LOG(
      Info,
      "offgraph_kv: grew flat_capacity=%lld->%lld allocated_bytes=%lld "
      "growth_count=%lld",
      static_cast<long long>(old_capacity),
      static_cast<long long>(new_capacity),
      static_cast<long long>(session.metrics.allocated_bytes),
      static_cast<long long>(session.metrics.growth_count));
  return Error::Ok;
}

Error build_descriptors(Context& context, CudaDelegateHandle* handle) {
  ET_CHECK_OR_RETURN_ERROR(
      handle->get_num_constants && handle->get_constant_name &&
          handle->get_constant_original_fqn &&
          handle->update_user_managed_constant_buffer_pairs,
      NotSupported,
      "offgraph_kv: AOTI external-buffer APIs are unavailable");
  size_t count = 0;
  ET_CHECK_OK_OR_RETURN_ERROR(
      handle->get_num_constants(handle->container_handle, &count));
  std::unordered_map<std::string, std::string> internal_names;
  for (size_t index = 0; index < count; ++index) {
    const char* internal = nullptr;
    const char* original = nullptr;
    ET_CHECK_OK_OR_RETURN_ERROR(handle->get_constant_name(
        handle->container_handle, index, &internal));
    ET_CHECK_OK_OR_RETURN_ERROR(handle->get_constant_original_fqn(
        handle->container_handle, index, &original));
    if (internal && original && internal[0] && original[0]) {
      internal_names.emplace(original, internal);
    }
  }

  auto& descriptors = context.descriptors[handle];
  for (const auto& layer : context.config.layers) {
    const int64_t max_capacity = layer.policy == OffGraphKVPolicy::Ring
        ? layer.window * 2
        : context.config.maximum_capacity;
    for (const auto& [suffix, kind] : {
             std::pair{"k", Descriptor::Kind::Key},
             std::pair{"v", Descriptor::Kind::Value}}) {
      const std::string name = fqn(layer.layer_id, suffix);
      const auto found = internal_names.find(name);
      if (found == internal_names.end()) {
        continue;
      }
      descriptors.emplace(
          name,
          Descriptor{
              found->second,
              layer.layer_id,
              kind,
              {layer.num_kv_heads * max_capacity * layer.head_dim},
              {1},
              context.config.storage_dtype,
              slimc10::Device(slimc10::DeviceType::CUDA, context.device)});
      context.discovered_fqns.insert(name);
    }
    const std::string capacity_name = fqn(layer.layer_id, "capacity");
    const auto found = internal_names.find(capacity_name);
    if (found != internal_names.end()) {
      descriptors.emplace(
          capacity_name,
          Descriptor{
              found->second,
              layer.layer_id,
              Descriptor::Kind::Capacity,
              {1},
              {1},
              slimc10::ScalarType::Long,
              slimc10::Device(slimc10::DeviceType::CUDA, context.device)});
      context.discovered_fqns.insert(capacity_name);
    }
  }
  return Error::Ok;
}

Error bind(
    Context& context,
    SessionState& session,
    int token,
    CudaDelegateHandle* handle) {
  auto existing = session.bound.find(handle);
  if (existing == session.bound.end()) {
    Bound bound;
    for (const auto& item : context.descriptors[handle]) {
      const Descriptor& descriptor = item.second;
      auto allocation = session.allocations.find(descriptor.layer_id);
      ET_CHECK_OR_RETURN_ERROR(
          allocation != session.allocations.end(),
          InvalidState,
          "offgraph_kv: allocation for layer %lld is missing",
          static_cast<long long>(descriptor.layer_id));
      void* pointer = descriptor.kind == Descriptor::Kind::Capacity
          ? allocation->second.capacity_device
          : (descriptor.kind == Descriptor::Kind::Key ? allocation->second.k
                                                      : allocation->second.v);
      auto tensor = std::make_unique<SlimTensor>(from_blob(
          pointer,
          ::executorch::runtime::makeArrayRef(
              descriptor.sizes.data(), descriptor.sizes.size()),
          ::executorch::runtime::makeArrayRef(
              descriptor.strides.data(), descriptor.strides.size()),
          descriptor.dtype,
          descriptor.device));
      bound.pairs.push_back(
          {descriptor.internal_name.c_str(),
           reinterpret_cast<aoti::AtenTensorHandle>(tensor.get())});
      bound.tensors.push_back(std::move(tensor));
    }
    existing = session.bound.emplace(handle, std::move(bound)).first;
  }
  const auto active = context.bound_session.find(handle);
  if (active != context.bound_session.end() && active->second == token) {
    return Error::Ok;
  }
  if (!existing->second.pairs.empty()) {
    ET_CHECK_OK_OR_RETURN_ERROR(
        handle->update_user_managed_constant_buffer_pairs(
            handle->container_handle,
            existing->second.pairs.data(),
            existing->second.pairs.size(),
            false,
            false));
  }
  context.bound_session[handle] = token;
  return Error::Ok;
}

} // namespace

namespace detail {

OffGraphKVContext offgraph_kv_create_context(OffGraphKVConfig config) {
  auto& state = manager();
  std::lock_guard<std::mutex> guard(state.mutex);
  const OffGraphKVContext id = state.next_context++;
  Context context;
  context.config = std::move(config);
  state.contexts.emplace(id, std::move(context));
  return id;
}

void offgraph_kv_destroy_context(OffGraphKVContext id) {
  auto& state = manager();
  std::lock_guard<std::mutex> guard(state.mutex);
  auto found = state.contexts.find(id);
  if (found == state.contexts.end()) {
    return;
  }
  for (auto& session : found->second.sessions) {
    for (auto& allocation : session.second.allocations) {
      release_allocation(found->second, allocation.second);
    }
  }
  (void)cudaStreamSynchronize(cudaStreamPerThread);
  for (auto it = state.handle_contexts.begin();
       it != state.handle_contexts.end();) {
    it = it->second == id ? state.handle_contexts.erase(it) : std::next(it);
  }
  state.contexts.erase(found);
}

void offgraph_kv_begin_load(OffGraphKVContext context) {
  loading_context = context;
}

void offgraph_kv_end_load() {
  loading_context = kInvalidOffGraphKVContext;
}

Result<int> offgraph_kv_create_session(OffGraphKVContext id) {
  auto& state = manager();
  std::lock_guard<std::mutex> guard(state.mutex);
  auto found = state.contexts.find(id);
  ET_CHECK_OR_RETURN_ERROR(
      found != state.contexts.end(), InvalidArgument, "invalid offgraph KV context");
  const int token = found->second.next_session_token++;
  found->second.sessions.emplace(token, SessionState{});
  return token;
}

void offgraph_kv_destroy_session(OffGraphKVContext id, int token) {
  auto& state = manager();
  std::lock_guard<std::mutex> guard(state.mutex);
  auto found = state.contexts.find(id);
  if (found == state.contexts.end()) {
    return;
  }
  auto session = found->second.sessions.find(token);
  if (session == found->second.sessions.end()) {
    return;
  }
  reset_cuda_graphs(found->second);
  for (auto& allocation : session->second.allocations) {
    release_allocation(found->second, allocation.second);
  }
  (void)cudaStreamSynchronize(cudaStreamPerThread);
  for (auto it = found->second.bound_session.begin();
       it != found->second.bound_session.end();) {
    it = it->second == token ? found->second.bound_session.erase(it)
                             : std::next(it);
  }
  found->second.sessions.erase(session);
}

void offgraph_kv_set_active(OffGraphKVContext id, int token) {
  active_context = id;
  active_session_token = token;
}

Error offgraph_kv_validate(OffGraphKVContext id) {
  auto& state = manager();
  std::lock_guard<std::mutex> guard(state.mutex);
  auto found = state.contexts.find(id);
  if (found == state.contexts.end()) {
    return Error::InvalidArgument;
  }
  Context& context = found->second;
  if (context.error != Error::Ok) {
    return context.error;
  }
  ET_CHECK_OR_RETURN_ERROR(
      context.config.maximum_capacity > 0 &&
          context.config.initial_capacity > 0 &&
          context.config.initial_capacity <= context.config.maximum_capacity &&
          !context.config.layers.empty(),
      InvalidArgument,
      "offgraph_kv: invalid cache configuration");
  std::unordered_set<int64_t> layer_ids;
  for (const auto& layer : context.config.layers) {
    ET_CHECK_OR_RETURN_ERROR(
        layer.layer_id >= 0 && layer.num_kv_heads > 0 && layer.head_dim > 0 &&
            (layer.policy == OffGraphKVPolicy::Flat || layer.window > 0) &&
            layer_ids.insert(layer.layer_id).second,
        InvalidArgument,
        "offgraph_kv: invalid or duplicate layer configuration");
    for (const char* suffix : {"k", "v", "capacity"}) {
      if (context.discovered_fqns.find(fqn(layer.layer_id, suffix)) ==
          context.discovered_fqns.end()) {
        ET_LOG(
            Error,
            "offgraph_kv: missing AOTI storage for layer %lld (%s)",
            static_cast<long long>(layer.layer_id),
            suffix);
        return Error::InvalidProgram;
      }
    }
  }
  return context.handles_associated ? Error::Ok : Error::InvalidState;
}

Error offgraph_kv_prepare(
    OffGraphKVContext id,
    int token,
    int64_t write_length) {
  auto& state = manager();
  std::lock_guard<std::mutex> guard(state.mutex);
  auto found = state.contexts.find(id);
  ET_CHECK_OR_RETURN_ERROR(
      found != state.contexts.end(), InvalidArgument, "invalid offgraph KV context");
  Context& context = found->second;
  auto session = context.sessions.find(token);
  ET_CHECK_OR_RETURN_ERROR(
      session != context.sessions.end(), InvalidArgument, "invalid offgraph KV session");
  ET_CHECK_OR_RETURN_ERROR(write_length > 0, InvalidArgument, "write length must be positive");
  const int64_t required = session->second.metrics.logical_length + write_length;
  ET_CHECK_OR_RETURN_ERROR(
      required <= context.config.maximum_capacity,
      InvalidArgument,
      "offgraph_kv: required capacity %lld exceeds maximum %lld",
      static_cast<long long>(required),
      static_cast<long long>(context.config.maximum_capacity));
  ET_CHECK_OK_OR_RETURN_ERROR(
      ensure_initial_allocations(context, session->second, cudaStreamPerThread));
  if (required > session->second.metrics.flat_capacity) {
    const int64_t doubled = session->second.metrics.flat_capacity * 2;
    const int64_t next = std::min(
        context.config.maximum_capacity,
        std::max(required, std::max(context.config.initial_capacity, doubled)));
    ET_CHECK_OK_OR_RETURN_ERROR(
        grow_flat(context, session->second, next, cudaStreamPerThread));
  }
  return Error::Ok;
}

Error offgraph_kv_commit(
    OffGraphKVContext id,
    int token,
    int64_t write_length) {
  auto& state = manager();
  std::lock_guard<std::mutex> guard(state.mutex);
  auto found = state.contexts.find(id);
  ET_CHECK_OR_RETURN_ERROR(
      found != state.contexts.end(), InvalidArgument, "invalid offgraph KV context");
  auto session = found->second.sessions.find(token);
  ET_CHECK_OR_RETURN_ERROR(
      session != found->second.sessions.end(),
      InvalidArgument,
      "invalid offgraph KV session");
  ET_CHECK_OR_RETURN_ERROR(
      write_length > 0, InvalidArgument, "write length must be positive");
  session->second.metrics.logical_length += write_length;
  return Error::Ok;
}

Error offgraph_kv_reset(OffGraphKVContext id, int token) {
  auto& state = manager();
  std::lock_guard<std::mutex> guard(state.mutex);
  auto found = state.contexts.find(id);
  ET_CHECK_OR_RETURN_ERROR(
      found != state.contexts.end(), InvalidArgument, "invalid offgraph KV context");
  auto session = found->second.sessions.find(token);
  ET_CHECK_OR_RETURN_ERROR(
      session != found->second.sessions.end(),
      InvalidArgument,
      "invalid offgraph KV session");
  session->second.metrics.logical_length = 0;
  ET_LOG(
      Info,
      "offgraph_kv: reset flat_capacity=%lld allocated_bytes=%lld",
      static_cast<long long>(session->second.metrics.flat_capacity),
      static_cast<long long>(session->second.metrics.allocated_bytes));
  return Error::Ok;
}

OffGraphKVMetrics offgraph_kv_metrics(OffGraphKVContext id) {
  auto& state = manager();
  std::lock_guard<std::mutex> guard(state.mutex);
  auto found = state.contexts.find(id);
  OffGraphKVMetrics metrics;
  if (found == state.contexts.end()) {
    return metrics;
  }
  for (const auto& session : found->second.sessions) {
    metrics.logical_length =
        std::max(metrics.logical_length, session.second.metrics.logical_length);
    metrics.flat_capacity =
        std::max(metrics.flat_capacity, session.second.metrics.flat_capacity);
    metrics.growth_count += session.second.metrics.growth_count;
    metrics.allocated_bytes += session.second.metrics.allocated_bytes;
  }
  return metrics;
}

int64_t offgraph_kv_initial_bytes_per_session(OffGraphKVContext id) {
  auto& state = manager();
  std::lock_guard<std::mutex> guard(state.mutex);
  auto found = state.contexts.find(id);
  if (found == state.contexts.end()) {
    return 0;
  }
  int64_t bytes = 0;
  for (const auto& layer : found->second.config.layers) {
    const int64_t capacity = layer.policy == OffGraphKVPolicy::Ring
        ? layer.window * 2
        : found->second.config.initial_capacity;
    bytes += static_cast<int64_t>(
        2 * storage_bytes(found->second.config, layer, capacity) +
        sizeof(int64_t));
  }
  return bytes;
}

} // namespace detail

void offgraph_kv_note_handle(CudaDelegateHandle* handle) {
  if (loading_context == kInvalidOffGraphKVContext) {
    return;
  }
  auto& state = manager();
  std::lock_guard<std::mutex> guard(state.mutex);
  auto found = state.contexts.find(loading_context);
  if (found == state.contexts.end()) {
    return;
  }
  Context& context = found->second;
  context.handles_associated = true;
  state.handle_contexts[handle] = loading_context;
  if (!context.device_known) {
    if (cudaGetDevice(&context.device) != cudaSuccess) {
      context.error = Error::Internal;
      return;
    }
    context.device_known = true;
  }
  const Error error = build_descriptors(context, handle);
  if (error != Error::Ok) {
    context.error = error;
  }
}

void offgraph_kv_forget_handle(CudaDelegateHandle* handle) {
  auto& state = manager();
  std::lock_guard<std::mutex> guard(state.mutex);
  auto found = state.handle_contexts.find(handle);
  if (found == state.handle_contexts.end()) {
    return;
  }
  auto context = state.contexts.find(found->second);
  if (context != state.contexts.end()) {
    context->second.descriptors.erase(handle);
    context->second.bound_session.erase(handle);
    for (auto& session : context->second.sessions) {
      session.second.bound.erase(handle);
    }
  }
  state.handle_contexts.erase(found);
}

Error offgraph_kv_rebind_for_execute(CudaDelegateHandle* handle) {
  auto& state = manager();
  std::lock_guard<std::mutex> guard(state.mutex);
  auto found = state.handle_contexts.find(handle);
  if (found == state.handle_contexts.end()) {
    return Error::Ok;
  }
  Context& context = state.contexts.at(found->second);
  const auto descriptors = context.descriptors.find(handle);
  if (descriptors != context.descriptors.end() && descriptors->second.empty()) {
    return Error::Ok;
  }
  ET_CHECK_OR_RETURN_ERROR(
      found->second == active_context &&
          active_session_token != kNoOffGraphKVSession,
      InvalidState,
      "offgraph_kv: active session is required for execute");
  if (context.error != Error::Ok) {
    return context.error;
  }
  auto session = context.sessions.find(active_session_token);
  ET_CHECK_OR_RETURN_ERROR(
      session != context.sessions.end(),
      InvalidArgument,
      "offgraph_kv: active session was not created");
  ET_CHECK_OR_RETURN_ERROR(
      !session->second.allocations.empty(),
      InvalidState,
      "offgraph_kv: prepare must run before execute");
  return bind(context, session->second, active_session_token, handle);
}

} // namespace executorch::backends::cuda
