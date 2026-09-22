/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cuda/runtime/cuda_kv_cache.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <executorch/backends/aoti/slim/c10/core/Device.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/aoti/slim/c10/cuda/Exception.h>
#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/backends/aoti/slim/factory/from_blob.h>
#include <executorch/backends/cuda/runtime/cuda_allocator.h>
#include <executorch/extension/llm/cache/sequence_cache.h>
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

std::string fqn(int64_t layer_id, const char* suffix) {
  return "__et_offgraph_kv_layer_" + std::to_string(layer_id) + "_" + suffix;
}

bool is_ring(const cache::LayerGeometry& layer) {
  return layer.policy.kind == cache::LayerPolicy::Kind::Ring;
}

// The slim enum shares ExecuTorch's ScalarType numbering. Any fixed-width
// scalar is storable; the kernels only ever ask for a float type, but the
// storage layer stays agnostic so it is the kernel, not this check, that
// decides what it can read.
bool storage_dtype_of(int kv_dtype, slimc10::ScalarType& out) {
  switch (static_cast<slimc10::ScalarType>(kv_dtype)) {
    case slimc10::ScalarType::Byte:
    case slimc10::ScalarType::Char:
    case slimc10::ScalarType::Short:
    case slimc10::ScalarType::Int:
    case slimc10::ScalarType::Long:
    case slimc10::ScalarType::Half:
    case slimc10::ScalarType::Float:
    case slimc10::ScalarType::Bool:
    case slimc10::ScalarType::BFloat16:
      out = static_cast<slimc10::ScalarType>(kv_dtype);
      return true;
    default:
      return false;
  }
}

// One sequence's KV storage on one CUDA device.
//
// The neutral base owns the logical length, admission and rewind. This class
// owns only bytes: the device allocations, their geometric growth, and the
// AOTI descriptors that point the compiled program at them. Physical slot math
// stays in the Triton kernels, because it has to run on device so a captured
// CUDA graph replays against the current positions rather than the ones that
// were live at capture.
class CudaSequenceKVCache final : public cache::SequenceCache,
                                  public CudaKVCache {
 public:
  CudaSequenceKVCache(
      const cache::CacheGeometry& geometry,
      const cache::CacheConfig& cfg,
      slimc10::ScalarType storage_dtype)
      : cache::SequenceCache(geometry, cfg),
        geometry_(geometry),
        config_(cfg),
        storage_dtype_(storage_dtype) {}

  CudaSequenceKVCache(const CudaSequenceKVCache&) = delete;
  CudaSequenceKVCache& operator=(const CudaSequenceKVCache&) = delete;
  CudaSequenceKVCache(CudaSequenceKVCache&&) = delete;
  CudaSequenceKVCache& operator=(CudaSequenceKVCache&&) = delete;

  ~CudaSequenceKVCache() override {
    std::lock_guard<std::mutex> guard(mutex_);
    for (auto& allocation : allocations_) {
      release_allocation(allocation.second);
    }
    (void)cudaStreamSynchronize(cudaStreamPerThread);
  }

  // cache::SequenceControl. Also rewinds the reported length so a reset
  // session does not keep reporting the old one.
  void clear() override {
    std::lock_guard<std::mutex> guard(mutex_);
    cache::SequenceCache::clear();
    metrics_.logical_length = 0;
    ET_LOG(
        Info,
        "offgraph_kv: reset flat_capacity=%lld allocated_bytes=%lld",
        static_cast<long long>(metrics_.flat_capacity),
        static_cast<long long>(metrics_.allocated_bytes));
  }

  // CudaKVCache.
  runtime::Result<bool> note_handle(CudaDelegateHandle* handle) override {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!device_known_) {
      if (cudaGetDevice(&device_) != cudaSuccess) {
        error_ = Error::Internal;
        return error_;
      }
      device_known_ = true;
    }
    const Error error = build_descriptors(handle);
    if (error != Error::Ok) {
      error_ = error;
      return error;
    }
    const bool serves = !descriptors_[handle].empty();
    if (!serves) {
      descriptors_.erase(handle);
      return false;
    }
    handles_associated_ = true;
    return true;
  }

  void forget_handle(CudaDelegateHandle* handle) override {
    std::lock_guard<std::mutex> guard(mutex_);
    descriptors_.erase(handle);
    bound_.erase(handle);
  }

  Error rebind_for_execute(CudaDelegateHandle* handle) override {
    std::lock_guard<std::mutex> guard(mutex_);
    if (descriptors_.find(handle) == descriptors_.end()) {
      return Error::Ok; // a handle with no off-graph storage
    }
    if (error_ != Error::Ok) {
      return error_;
    }
    ET_CHECK_OR_RETURN_ERROR(
        !allocations_.empty(),
        InvalidState,
        "offgraph_kv: prepare_step must run before execute");
    return bind(handle);
  }

  Error validate() const override {
    std::lock_guard<std::mutex> guard(mutex_);
    if (error_ != Error::Ok) {
      return error_;
    }
    for (size_t index = 0; index < geometry_.layers.size(); ++index) {
      for (const char* suffix : {"k", "v", "capacity"}) {
        if (discovered_fqns_.find(fqn(static_cast<int64_t>(index), suffix)) ==
            discovered_fqns_.end()) {
          ET_LOG(
              Error,
              "offgraph_kv: missing AOTI storage for layer %zu (%s)",
              index,
              suffix);
          return Error::InvalidProgram;
        }
      }
    }
    return handles_associated_ ? Error::Ok : Error::InvalidState;
  }

  Error prepare_step(int64_t write_length) override {
    std::lock_guard<std::mutex> guard(mutex_);
    ET_CHECK_OR_RETURN_ERROR(
        write_length > 0, InvalidArgument, "write length must be positive");
    ET_CHECK_OK_OR_RETURN_ERROR(error_);
    // plan() is the neutral admission check: it rejects a step that runs past
    // capacity, and one wider than a ring layer can serve. It runs on the host
    // between executes, so it never lands inside a CUDA graph capture.
    const int position = length();
    for (size_t index = 0; index < geometry_.layers.size(); ++index) {
      ET_CHECK_OR_RETURN_ERROR(
          plan(
              static_cast<int>(index), position, static_cast<int>(write_length))
              .has_value(),
          InvalidArgument,
          "offgraph_kv: a %lld-token step at position %d does not fit layer %zu",
          static_cast<long long>(write_length),
          position,
          index);
    }
    const int64_t required = position + write_length;
    ET_CHECK_OK_OR_RETURN_ERROR(
        ensure_initial_allocations(cudaStreamPerThread));
    if (required > metrics_.flat_capacity) {
      const int64_t doubled = metrics_.flat_capacity * 2;
      const int64_t next = std::min<int64_t>(
          config_.capacity,
          std::max(
              required, std::max<int64_t>(config_.initial_capacity, doubled)));
      ET_CHECK_OK_OR_RETURN_ERROR(grow_flat(next, cudaStreamPerThread));
    }
    return Error::Ok;
  }

  Error commit_step(int64_t write_length) override {
    std::lock_guard<std::mutex> guard(mutex_);
    ET_CHECK_OR_RETURN_ERROR(
        write_length > 0, InvalidArgument, "write length must be positive");
    ET_CHECK_OK_OR_RETURN_ERROR(error_);
    const auto step = plan(0, length(), static_cast<int>(write_length));
    ET_CHECK_OR_RETURN_ERROR(
        step.has_value(), InvalidArgument, "offgraph_kv: uncommittable step");
    cache::SequenceCache::commit(*step);
    metrics_.logical_length = length();
    return Error::Ok;
  }

  OffGraphKVMetrics metrics() const override {
    std::lock_guard<std::mutex> guard(mutex_);
    return metrics_;
  }

 protected:
  void* face(cache::FaceId id) override {
    if (void* p = cache::SequenceCache::face(id)) {
      return p;
    }
    return cache::expose<CudaKVCache>(this, id);
  }

 private:
  size_t element_size() const {
    return slimc10::elementSize(storage_dtype_);
  }

  size_t storage_bytes(const cache::LayerGeometry& layer, int64_t capacity)
      const {
    return static_cast<size_t>(layer.n_kv_heads) *
        static_cast<size_t>(capacity) * static_cast<size_t>(layer.head_dim) *
        element_size();
  }

  // Slots a ring layer needs to serve one step of up to max_write tokens: the
  // step writes all of them before attending, and its earliest query still
  // reads back window - 1 positions. Same formula as cache::RingPolicy and as
  // ring_physical_capacity() in triton/kernels/offgraph_kv.py.
  int64_t ring_capacity(const cache::LayerGeometry& layer) const {
    const int max_write =
        config_.max_write ? *config_.max_write : layer.policy.window;
    return static_cast<int64_t>(layer.policy.window) + max_write - 1;
  }

  Error allocate_layer(
      int64_t layer_id,
      const cache::LayerGeometry& layer,
      int64_t capacity,
      cudaStream_t stream) {
    Allocation allocation;
    const size_t bytes = storage_bytes(layer, capacity);
    auto k = CudaAllocator::allocate_async(bytes, device_, stream);
    ET_CHECK_OK_OR_RETURN_ERROR(k.error());
    allocation.k = k.get();
    auto v = CudaAllocator::allocate_async(bytes, device_, stream);
    if (!v.ok()) {
      CudaAllocator::deallocate_async(allocation.k, device_, stream);
      return v.error();
    }
    allocation.v = v.get();
    auto cap = CudaAllocator::allocate_async(sizeof(int64_t), device_, stream);
    if (!cap.ok()) {
      CudaAllocator::deallocate_async(allocation.k, device_, stream);
      CudaAllocator::deallocate_async(allocation.v, device_, stream);
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
      CudaAllocator::deallocate_async(allocation.k, device_, stream);
      CudaAllocator::deallocate_async(allocation.v, device_, stream);
      CudaAllocator::deallocate_async(
          allocation.capacity_device, device_, stream);
      ET_LOG(
          Error,
          "offgraph_kv: capacity initialization failed: %s",
          cudaGetErrorString(copy_error));
      return Error::Internal;
    }
    metrics_.allocated_bytes +=
        static_cast<int64_t>(2 * bytes + sizeof(int64_t));
    allocations_.emplace(layer_id, allocation);
    return Error::Ok;
  }

  Error ensure_initial_allocations(cudaStream_t stream) {
    bool allocated = false;
    for (size_t index = 0; index < geometry_.layers.size(); ++index) {
      const int64_t layer_id = static_cast<int64_t>(index);
      if (allocations_.find(layer_id) != allocations_.end()) {
        continue;
      }
      const cache::LayerGeometry& layer = geometry_.layers[index];
      const int64_t capacity =
          is_ring(layer) ? ring_capacity(layer) : config_.initial_capacity;
      ET_CHECK_OK_OR_RETURN_ERROR(
          allocate_layer(layer_id, layer, capacity, stream));
      allocated = true;
    }
    if (allocated) {
      metrics_.flat_capacity = config_.initial_capacity;
      ET_LOG(
          Info,
          "offgraph_kv: initialized flat_capacity=%lld allocated_bytes=%lld",
          static_cast<long long>(metrics_.flat_capacity),
          static_cast<long long>(metrics_.allocated_bytes));
    }
    return Error::Ok;
  }

  void release_allocation(Allocation& allocation) {
    if (!device_known_) {
      return;
    }
    CudaAllocator::deallocate_async(allocation.k, device_, cudaStreamPerThread);
    CudaAllocator::deallocate_async(allocation.v, device_, cudaStreamPerThread);
    CudaAllocator::deallocate_async(
        allocation.capacity_device, device_, cudaStreamPerThread);
  }

  Error grow_flat(int64_t new_capacity, cudaStream_t stream) {
    const int64_t old_capacity = metrics_.flat_capacity;
    for (size_t index = 0; index < geometry_.layers.size(); ++index) {
      const cache::LayerGeometry& layer = geometry_.layers[index];
      if (is_ring(layer)) {
        continue;
      }
      const int64_t layer_id = static_cast<int64_t>(index);
      auto old_it = allocations_.find(layer_id);
      ET_CHECK_OR_RETURN_ERROR(
          old_it != allocations_.end(),
          InvalidState,
          "offgraph_kv: layer allocation is missing");
      Allocation old = old_it->second;
      allocations_.erase(old_it);
      const Error allocation_error =
          allocate_layer(layer_id, layer, new_capacity, stream);
      if (allocation_error != Error::Ok) {
        allocations_.emplace(layer_id, old);
        return allocation_error;
      }
      Allocation& replacement = allocations_.at(layer_id);
      const size_t row_bytes = static_cast<size_t>(old.capacity) *
          static_cast<size_t>(layer.head_dim) * element_size();
      const size_t new_pitch = static_cast<size_t>(new_capacity) *
          static_cast<size_t>(layer.head_dim) * element_size();
      for (const auto pair :
           {std::pair{replacement.k, old.k}, std::pair{replacement.v, old.v}}) {
        const cudaError_t copy_error = cudaMemcpy2DAsync(
            pair.first,
            new_pitch,
            pair.second,
            row_bytes,
            row_bytes,
            static_cast<size_t>(layer.n_kv_heads),
            cudaMemcpyDeviceToDevice,
            stream);
        if (copy_error != cudaSuccess) {
          // allocations_ already points at the replacement, so returning here
          // would strand `old`. Free it, and refuse further use: this layer now
          // holds storage with unpopulated history while earlier layers have
          // already grown, and no partial state is safe to decode against.
          CudaAllocator::deallocate_async(old.k, device_, stream);
          CudaAllocator::deallocate_async(old.v, device_, stream);
          CudaAllocator::deallocate_async(old.capacity_device, device_, stream);
          metrics_.allocated_bytes -= static_cast<int64_t>(
              2 * storage_bytes(layer, old.capacity) + sizeof(int64_t));
          ET_LOG(
              Error,
              "offgraph_kv: growth copy failed: %s",
              cudaGetErrorString(copy_error));
          error_ = Error::Internal;
          return error_;
        }
      }
      metrics_.allocated_bytes -= static_cast<int64_t>(
          2 * storage_bytes(layer, old.capacity) + sizeof(int64_t));
      CudaAllocator::deallocate_async(old.k, device_, stream);
      CudaAllocator::deallocate_async(old.v, device_, stream);
      CudaAllocator::deallocate_async(old.capacity_device, device_, stream);
    }
    metrics_.flat_capacity = new_capacity;
    metrics_.growth_count++;
    bound_.clear();
    // Growth moved every pointer a captured graph baked in, so any handle
    // running a graph has to capture again. Done only once the growth has
    // committed: an earlier sweep would throw away a still-valid graph on
    // every path above that returns an error.
    for (const auto& item : descriptors_) {
      if (item.first->cuda_graph_state.phase != CudaGraphPhase::Disabled) {
        item.first->cuda_graph_state.reset_for_recapture();
      }
    }
    ET_LOG(
        Info,
        "offgraph_kv: grew flat_capacity=%lld->%lld allocated_bytes=%lld "
        "growth_count=%lld",
        static_cast<long long>(old_capacity),
        static_cast<long long>(new_capacity),
        static_cast<long long>(metrics_.allocated_bytes),
        static_cast<long long>(metrics_.growth_count));
    return Error::Ok;
  }

  Error build_descriptors(CudaDelegateHandle* handle) {
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

    auto& descriptors = descriptors_[handle];
    size_t found_layers = 0;
    for (size_t index = 0; index < geometry_.layers.size(); ++index) {
      const cache::LayerGeometry& layer = geometry_.layers[index];
      const int64_t layer_id = static_cast<int64_t>(index);
      const int64_t max_capacity =
          is_ring(layer) ? ring_capacity(layer) : config_.capacity;
      for (const auto& [suffix, kind] :
           {std::pair{"k", Descriptor::Kind::Key},
            std::pair{"v", Descriptor::Kind::Value}}) {
        const std::string name = fqn(layer_id, suffix);
        const auto found = internal_names.find(name);
        if (found == internal_names.end()) {
          continue;
        }
        descriptors.emplace(
            name,
            Descriptor{
                found->second,
                layer_id,
                kind,
                {layer.n_kv_heads * max_capacity * layer.head_dim},
                {1},
                storage_dtype_,
                slimc10::Device(slimc10::DeviceType::CUDA, device_)});
        discovered_fqns_.insert(name);
      }
      const std::string capacity_name = fqn(layer_id, "capacity");
      const auto found = internal_names.find(capacity_name);
      if (found != internal_names.end()) {
        ++found_layers;
        descriptors.emplace(
            capacity_name,
            Descriptor{
                found->second,
                layer_id,
                Descriptor::Kind::Capacity,
                {1},
                {1},
                slimc10::ScalarType::Long,
                slimc10::Device(slimc10::DeviceType::CUDA, device_)});
        discovered_fqns_.insert(capacity_name);
      }
    }
    // A method either has no off-graph storage (embeddings, vision) or has all
    // of it. Anything between means the lowering pass and this runtime disagree
    // about the geometry, which is worth failing on here -- while the offending
    // method is still named -- rather than at the first decode.
    ET_CHECK_OR_RETURN_ERROR(
        found_layers == 0 || found_layers == geometry_.layers.size(),
        InvalidProgram,
        "offgraph_kv: program carries %zu of %zu layers' storage",
        found_layers,
        geometry_.layers.size());
    return Error::Ok;
  }

  Error bind(CudaDelegateHandle* handle) {
    auto existing = bound_.find(handle);
    if (existing != bound_.end()) {
      return Error::Ok;
    }
    Bound bound;
    for (const auto& item : descriptors_[handle]) {
      const Descriptor& descriptor = item.second;
      auto allocation = allocations_.find(descriptor.layer_id);
      ET_CHECK_OR_RETURN_ERROR(
          allocation != allocations_.end(),
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
    bound_.emplace(handle, std::move(bound));
    return Error::Ok;
  }

  // Guards everything below. The engine serialises its own calls, but the
  // delegate reaches note_handle/rebind from whichever thread loads or runs a
  // method.
  mutable std::mutex mutex_;

  cache::CacheGeometry geometry_;
  cache::CacheConfig config_;
  slimc10::ScalarType storage_dtype_;

  int device_{0};
  bool device_known_{false};
  bool handles_associated_{false};
  Error error_{Error::Ok};
  OffGraphKVMetrics metrics_;
  std::unordered_map<int64_t, Allocation> allocations_;
  std::unordered_map<
      CudaDelegateHandle*,
      std::unordered_map<std::string, Descriptor>>
      descriptors_;
  std::unordered_map<CudaDelegateHandle*, Bound> bound_;
  std::unordered_set<std::string> discovered_fqns_;
};

} // namespace

std::shared_ptr<cache::Cache> make_cuda_sequence_kv_cache(
    const cache::CacheGeometry& geometry,
    const cache::CacheConfig& cfg) {
  slimc10::ScalarType storage_dtype = slimc10::ScalarType::BFloat16;
  // Checked before constructing: SequenceCache asserts on an invalid geometry,
  // and returning null lets CacheFactory report the failure instead.
  if (!cache::valid(geometry, cfg) || cfg.initial_capacity <= 0 ||
      cfg.initial_capacity > cfg.capacity ||
      !storage_dtype_of(cfg.kv_dtype, storage_dtype)) {
    ET_LOG(Error, "offgraph_kv: invalid cache geometry or config");
    return nullptr;
  }
  return std::make_shared<CudaSequenceKVCache>(geometry, cfg, storage_dtype);
}

} // namespace executorch::backends::cuda
