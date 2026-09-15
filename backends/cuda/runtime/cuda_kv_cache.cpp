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

struct Context {
  OffGraphKVConfig config;
  int device{0};
  bool device_known{false};
  bool handles_associated{false};
  Error error{Error::Ok};
  OffGraphKVMetrics metrics;
  std::unordered_map<int64_t, Allocation> allocations;
  std::unordered_map<CudaDelegateHandle*, std::unordered_map<std::string, Descriptor>>
      descriptors;
  std::unordered_map<CudaDelegateHandle*, Bound> bound;
  std::unordered_set<std::string> discovered_fqns;
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
  context.metrics.allocated_bytes += static_cast<int64_t>(2 * bytes + sizeof(int64_t));
  context.allocations.emplace(layer.layer_id, allocation);
  return Error::Ok;
}

Error ensure_initial_allocations(Context& context, cudaStream_t stream) {
  bool allocated = false;
  for (const auto& layer : context.config.layers) {
    if (context.allocations.find(layer.layer_id) != context.allocations.end()) {
      continue;
    }
    const int64_t capacity = layer.policy == OffGraphKVPolicy::Ring
        ? layer.window * 2
        : context.config.initial_capacity;
    ET_CHECK_OK_OR_RETURN_ERROR(
        allocate_layer(context, layer, capacity, stream));
    allocated = true;
  }
  if (allocated) {
    context.metrics.flat_capacity = context.config.initial_capacity;
    ET_LOG(
        Info,
        "offgraph_kv: initialized flat_capacity=%lld allocated_bytes=%lld",
        static_cast<long long>(context.metrics.flat_capacity),
        static_cast<long long>(context.metrics.allocated_bytes));
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

Error grow_flat(Context& context, int64_t new_capacity, cudaStream_t stream) {
  const int64_t old_capacity = context.metrics.flat_capacity;
  for (const auto& item : context.descriptors) {
    if (item.first->cuda_graph_state.phase != CudaGraphPhase::Disabled) {
      item.first->cuda_graph_state.reset_for_recapture();
    }
  }
  for (const auto& layer : context.config.layers) {
    if (layer.policy != OffGraphKVPolicy::Flat) {
      continue;
    }
    auto old_it = context.allocations.find(layer.layer_id);
    ET_CHECK_OR_RETURN_ERROR(
        old_it != context.allocations.end(),
        InvalidState,
        "offgraph_kv: layer allocation is missing");
    Allocation old = old_it->second;
    context.allocations.erase(old_it);
    const Error allocation_error =
        allocate_layer(context, layer, new_capacity, stream);
    if (allocation_error != Error::Ok) {
      context.allocations.emplace(layer.layer_id, old);
      return allocation_error;
    }
    Allocation& replacement = context.allocations.at(layer.layer_id);
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
    context.metrics.allocated_bytes -= static_cast<int64_t>(
        2 * storage_bytes(context.config, layer, old.capacity) +
        sizeof(int64_t));
    CudaAllocator::deallocate_async(old.k, context.device, stream);
    CudaAllocator::deallocate_async(old.v, context.device, stream);
    CudaAllocator::deallocate_async(
        old.capacity_device, context.device, stream);
  }
  context.metrics.flat_capacity = new_capacity;
  context.metrics.growth_count++;
  context.bound.clear();
  ET_LOG(
      Info,
      "offgraph_kv: grew flat_capacity=%lld->%lld allocated_bytes=%lld "
      "growth_count=%lld",
      static_cast<long long>(old_capacity),
      static_cast<long long>(new_capacity),
      static_cast<long long>(context.metrics.allocated_bytes),
      static_cast<long long>(context.metrics.growth_count));
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

Error bind(Context& context, CudaDelegateHandle* handle) {
  auto existing = context.bound.find(handle);
  if (existing != context.bound.end()) {
    return Error::Ok;
  }
  Bound bound;
  for (const auto& item : context.descriptors[handle]) {
    const Descriptor& descriptor = item.second;
    auto allocation = context.allocations.find(descriptor.layer_id);
    ET_CHECK_OR_RETURN_ERROR(
        allocation != context.allocations.end(),
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
  if (!bound.pairs.empty()) {
    ET_CHECK_OK_OR_RETURN_ERROR(
        handle->update_user_managed_constant_buffer_pairs(
            handle->container_handle,
            bound.pairs.data(),
            bound.pairs.size(),
            false,
            false));
  }
  context.bound.emplace(handle, std::move(bound));
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
  for (auto& allocation : found->second.allocations) {
    release_allocation(found->second, allocation.second);
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

Error offgraph_kv_prepare(OffGraphKVContext id, int64_t write_length) {
  auto& state = manager();
  std::lock_guard<std::mutex> guard(state.mutex);
  auto found = state.contexts.find(id);
  ET_CHECK_OR_RETURN_ERROR(
      found != state.contexts.end(), InvalidArgument, "invalid offgraph KV context");
  Context& context = found->second;
  ET_CHECK_OR_RETURN_ERROR(write_length > 0, InvalidArgument, "write length must be positive");
  const int64_t required = context.metrics.logical_length + write_length;
  ET_CHECK_OR_RETURN_ERROR(
      required <= context.config.maximum_capacity,
      InvalidArgument,
      "offgraph_kv: required capacity %lld exceeds maximum %lld",
      static_cast<long long>(required),
      static_cast<long long>(context.config.maximum_capacity));
  ET_CHECK_OK_OR_RETURN_ERROR(ensure_initial_allocations(context, cudaStreamPerThread));
  if (required > context.metrics.flat_capacity) {
    const int64_t doubled = context.metrics.flat_capacity * 2;
    const int64_t next = std::min(
        context.config.maximum_capacity,
        std::max(required, std::max(context.config.initial_capacity, doubled)));
    ET_CHECK_OK_OR_RETURN_ERROR(grow_flat(context, next, cudaStreamPerThread));
  }
  return Error::Ok;
}

Error offgraph_kv_commit(OffGraphKVContext id, int64_t write_length) {
  auto& state = manager();
  std::lock_guard<std::mutex> guard(state.mutex);
  auto found = state.contexts.find(id);
  ET_CHECK_OR_RETURN_ERROR(
      found != state.contexts.end(), InvalidArgument, "invalid offgraph KV context");
  ET_CHECK_OR_RETURN_ERROR(
      write_length > 0, InvalidArgument, "write length must be positive");
  found->second.metrics.logical_length += write_length;
  return Error::Ok;
}

Error offgraph_kv_reset(OffGraphKVContext id) {
  auto& state = manager();
  std::lock_guard<std::mutex> guard(state.mutex);
  auto found = state.contexts.find(id);
  ET_CHECK_OR_RETURN_ERROR(
      found != state.contexts.end(), InvalidArgument, "invalid offgraph KV context");
  found->second.metrics.logical_length = 0;
  ET_LOG(
      Info,
      "offgraph_kv: reset flat_capacity=%lld allocated_bytes=%lld",
      static_cast<long long>(found->second.metrics.flat_capacity),
      static_cast<long long>(found->second.metrics.allocated_bytes));
  return Error::Ok;
}

OffGraphKVMetrics offgraph_kv_metrics(OffGraphKVContext id) {
  auto& state = manager();
  std::lock_guard<std::mutex> guard(state.mutex);
  auto found = state.contexts.find(id);
  return found == state.contexts.end() ? OffGraphKVMetrics{} : found->second.metrics;
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
    context->second.bound.erase(handle);
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
  if (context.error != Error::Ok) {
    return context.error;
  }
  ET_CHECK_OR_RETURN_ERROR(
      !context.allocations.empty(),
      InvalidState,
      "offgraph_kv: prepare must run before execute");
  return bind(context, handle);
}

} // namespace executorch::backends::cuda
