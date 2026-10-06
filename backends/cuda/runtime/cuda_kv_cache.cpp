/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cuda/runtime/cuda_kv_cache.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <executorch/backends/aoti/slim/c10/core/Device.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/backends/aoti/slim/factory/from_blob.h>
#include <executorch/extension/cuda/cuda_allocator.h>
#include <executorch/extension/llm/cache/cache_registry.h>
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
  int64_t rows{0};
};

struct Descriptor {
  std::string internal_name;
  int64_t layer_id{0};
  bool is_value{false};
};

// The tensors a handle's AOTI constants currently point at.
struct Bound {
  std::vector<std::unique_ptr<SlimTensor>> tensors;
};

// Makes the cache's device current for a scope. The backend runs on one
// device, so this is normally a no-op, but event and stream calls act on the
// current device and a caller may have switched it.
class DeviceGuard {
 public:
  explicit DeviceGuard(int device) {
    if (cudaGetDevice(&previous_) == cudaSuccess && previous_ != device) {
      restore_ = cudaSetDevice(device) == cudaSuccess;
    }
  }
  ~DeviceGuard() {
    if (restore_) {
      (void)cudaSetDevice(previous_);
    }
  }
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

 private:
  int previous_{0};
  bool restore_{false};
};

std::string fqn(int64_t layer_id, const char* suffix) {
  return "__et_offgraph_kv_layer_" + std::to_string(layer_id) + "_" + suffix;
}

bool is_ring(const cache::LayerGeometry& layer) {
  return layer.policy.kind == cache::LayerPolicy::Kind::Ring;
}

// The slim enum shares ExecuTorch's ScalarType numbering. The storage layer
// accepts these dense scalar types; which one a program can use is decided
// by its compiled constants, which note_handle() checks the dtype against.
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
    case slimc10::ScalarType::Undefined:
    default:
      return false;
  }
}

// One sequence's KV storage on one CUDA device.
//
// The neutral base owns the logical length, admission and rewind. This class
// owns only bytes: the device allocations, their geometric growth, and the
// AOTI bindings that point the compiled program at them. Physical slot math
// stays in the compiled program, on device, so a captured CUDA graph replays
// against the current positions rather than the ones live at capture.
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

  // Nothing else can reach the cache once it is being destroyed, so this takes
  // no lock. It also never touches stream_: that is the caller's stream, which
  // may already be gone. The device is drained instead, which covers any step
  // still in flight on whatever stream ran it.
  ~CudaSequenceKVCache() override {
    if (!device_known_) {
      return;
    }
    DeviceGuard device(device_);
    (void)cudaDeviceSynchronize();
    for (auto& allocation : allocations_) {
      release(allocation.second, cudaStreamPerThread);
    }
    (void)cudaStreamSynchronize(cudaStreamPerThread);
    if (last_step_done_ != nullptr) {
      (void)cudaEventDestroy(last_step_done_);
    }
  }

  // cache::SequenceControl. Keeps the storage: a reset session reuses the
  // grown allocations, so it neither regrows nor invalidates captured graphs.
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
    if (descriptors_[handle].empty()) {
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
    ET_CHECK_OK_OR_RETURN_ERROR(error_);
    ET_CHECK_OR_RETURN_ERROR(
        !allocations_.empty(),
        InvalidState,
        "offgraph_kv: prepare_step must run before execute");
    return bind(handle);
  }

  Error prepare_step(int64_t write_length, cudaStream_t stream) override {
    std::lock_guard<std::mutex> guard(mutex_);
    ET_CHECK_OK_OR_RETURN_ERROR(check_write_length(write_length));
    ET_CHECK_OK_OR_RETURN_ERROR(error_);
    if (!validated_) {
      const Error valid = validate_locked();
      if (valid != Error::Ok) {
        error_ = valid;
        return valid;
      }
      validated_ = true;
    }
    const int position = length();
    ET_CHECK_OK_OR_RETURN_ERROR(admit(position, write_length));
    DeviceGuard device(device_);
    ET_CHECK_OK_OR_RETURN_ERROR(follow_previous_step(stream));
    stream_ = stream;
    const int64_t required = position + write_length;
    ET_CHECK_OK_OR_RETURN_ERROR(ensure_initial_allocations(required, stream));
    if (required > metrics_.flat_capacity) {
      const int64_t next = std::min<int64_t>(
          config_.capacity,
          std::max<int64_t>(required, metrics_.flat_capacity * 2));
      ET_CHECK_OK_OR_RETURN_ERROR(grow_flat(next, stream));
    }
    return Error::Ok;
  }

  Error commit_step(int64_t write_length) override {
    std::lock_guard<std::mutex> guard(mutex_);
    ET_CHECK_OK_OR_RETURN_ERROR(check_write_length(write_length));
    ET_CHECK_OK_OR_RETURN_ERROR(error_);
    // Every layer, as prepare_step() admitted it: a ring layer can refuse a
    // step the flat layers accept.
    const int position = length();
    ET_CHECK_OK_OR_RETURN_ERROR(admit(position, write_length));
    const auto step = plan(0, position, static_cast<int>(write_length));
    ET_CHECK_OR_RETURN_ERROR(
        step.has_value(), InvalidArgument, "offgraph_kv: uncommittable step");
    cache::SequenceCache::commit(*step);
    metrics_.logical_length = length();
    DeviceGuard device(device_);
    return mark_step_done();
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
  // Bytes in one sequence row of a layer: every head's vector at one position.
  size_t row_bytes(const cache::LayerGeometry& layer) const {
    return static_cast<size_t>(layer.n_kv_heads) *
        static_cast<size_t>(layer.head_dim) *
        slimc10::elementSize(storage_dtype_);
  }

  // Slots a ring layer needs to serve one step of up to max_write tokens: the
  // step writes all of them before attending, and its earliest query still
  // reads back window - 1 positions. Same formula as cache::RingPolicy and as
  // ring_physical_capacity() in backends/cuda/passes/lower_offgraph_kv.py.
  // make_cuda_sequence_kv_cache() refuses ring layers without max_write.
  int64_t ring_capacity(const cache::LayerGeometry& layer) const {
    return static_cast<int64_t>(layer.policy.window) +
        config_.max_write.value_or(1) - 1;
  }

  static Error check_write_length(int64_t write_length) {
    ET_CHECK_OR_RETURN_ERROR(
        write_length > 0 && write_length <= std::numeric_limits<int>::max(),
        InvalidArgument,
        "offgraph_kv: write length %lld is out of range",
        static_cast<long long>(write_length));
    return Error::Ok;
  }

  // plan() is the neutral admission check: it rejects a step that runs past
  // capacity, and one wider than a ring layer can serve. It runs on the host
  // between executes, so it never lands inside a CUDA graph capture.
  Error admit(int position, int64_t write_length) const {
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
    return Error::Ok;
  }

  // Rows the compiled program declares for a layer's storage.
  int64_t declared_rows(const cache::LayerGeometry& layer) const {
    return is_ring(layer) ? ring_capacity(layer) : config_.capacity;
  }

  // Whole-model check, so it cannot run until every program has been loaded
  // and registered. The first step is the earliest moment that is guaranteed.
  // Caller holds mutex_.
  Error validate_locked() const {
    for (size_t index = 0; index < geometry_.layers.size(); ++index) {
      for (const char* suffix : {"k", "v"}) {
        if (discovered_fqns_.count(fqn(static_cast<int64_t>(index), suffix)) ==
            0) {
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

  Error allocate_layer(
      const cache::LayerGeometry& layer,
      int64_t rows,
      cudaStream_t stream,
      Allocation& out) {
    const size_t bytes = row_bytes(layer) * static_cast<size_t>(rows);
    auto k = CudaAllocator::allocate_async(bytes, device_, stream);
    ET_CHECK_OK_OR_RETURN_ERROR(k.error());
    auto v = CudaAllocator::allocate_async(bytes, device_, stream);
    if (!v.ok()) {
      CudaAllocator::deallocate_async(k.get(), device_, stream);
      return v.error();
    }
    out = Allocation{k.get(), v.get(), rows};
    metrics_.allocated_bytes += static_cast<int64_t>(2 * bytes);
    return Error::Ok;
  }

  void release(Allocation& allocation, cudaStream_t stream) {
    if (!device_known_) {
      return;
    }
    CudaAllocator::deallocate_async(allocation.k, device_, stream);
    CudaAllocator::deallocate_async(allocation.v, device_, stream);
  }

  // Releases an allocation made by allocate_layer and takes it off the books.
  void discard(
      const cache::LayerGeometry& layer,
      Allocation& allocation,
      cudaStream_t stream) {
    metrics_.allocated_bytes -=
        static_cast<int64_t>(2 * row_bytes(layer) * allocation.rows);
    release(allocation, stream);
  }

  // The previous step may have run on another stream (a caller-selected one),
  // and its kernels may still be writing the storage this step reads, or that
  // a growth below copies and frees. Order this stream behind it.
  Error follow_previous_step(cudaStream_t stream) {
    if (last_step_done_ == nullptr || stream == stream_) {
      return Error::Ok;
    }
    const cudaError_t error = cudaStreamWaitEvent(stream, last_step_done_, 0);
    ET_CHECK_OR_RETURN_ERROR(
        error == cudaSuccess,
        Internal,
        "offgraph_kv: cannot order the step stream behind the previous one: %s",
        cudaGetErrorString(error));
    return Error::Ok;
  }

  // Records where the committed step's work ends on its stream. Called once
  // the delegate has enqueued all of it, so the event covers every kernel that
  // touched the storage.
  Error mark_step_done() {
    if (last_step_done_ == nullptr) {
      const cudaError_t error =
          cudaEventCreateWithFlags(&last_step_done_, cudaEventDisableTiming);
      ET_CHECK_OR_RETURN_ERROR(
          error == cudaSuccess,
          Internal,
          "offgraph_kv: cannot create the step event: %s",
          cudaGetErrorString(error));
    }
    const cudaError_t error = cudaEventRecord(last_step_done_, stream_);
    ET_CHECK_OR_RETURN_ERROR(
        error == cudaSuccess,
        Internal,
        "offgraph_kv: cannot record the step event: %s",
        cudaGetErrorString(error));
    return Error::Ok;
  }

  // A first step wider than the initial capacity is allocated at its own
  // width rather than allocated and immediately grown.
  Error ensure_initial_allocations(int64_t required, cudaStream_t stream) {
    if (!allocations_.empty()) {
      return Error::Ok;
    }
    const int64_t flat_rows =
        std::max<int64_t>(config_.initial_capacity, required);
    // All or nothing: a failure frees what was allocated so far, and the next
    // step may try again.
    std::vector<Allocation> allocated;
    allocated.reserve(geometry_.layers.size());
    for (const cache::LayerGeometry& layer : geometry_.layers) {
      const int64_t rows = is_ring(layer) ? ring_capacity(layer) : flat_rows;
      Allocation allocation;
      const Error error = allocate_layer(layer, rows, stream, allocation);
      if (error != Error::Ok) {
        for (size_t index = 0; index < allocated.size(); ++index) {
          discard(geometry_.layers[index], allocated[index], stream);
        }
        return error;
      }
      allocated.push_back(allocation);
    }
    for (size_t index = 0; index < allocated.size(); ++index) {
      allocations_.emplace(static_cast<int64_t>(index), allocated[index]);
    }
    metrics_.flat_capacity = flat_rows;
    ET_LOG(
        Info,
        "offgraph_kv: initialized flat_capacity=%lld allocated_bytes=%lld",
        static_cast<long long>(metrics_.flat_capacity),
        static_cast<long long>(metrics_.allocated_bytes));
    return Error::Ok;
  }

  // Reallocates every flat layer at new_rows and carries the rows already
  // written across. BSHD storage makes that one contiguous prefix per buffer.
  // Everything is ordered on `stream`, so the copy runs after the last step
  // that wrote the old storage, the old storage is freed only after the copy,
  // and the next step's kernels see the copied rows.
  //
  // Transactional: every replacement is allocated and filled before any old
  // storage is released. A failure on any layer frees the replacements and
  // leaves the cache exactly as it was -- storage, bindings, capacity -- so the
  // step fails but the cache stays usable.
  Error grow_flat(int64_t new_rows, cudaStream_t stream) {
    const int64_t old_rows = metrics_.flat_capacity;
    const int64_t live_rows = length();
    std::vector<std::pair<size_t, Allocation>> replacements;
    auto roll_back = [&]() {
      for (auto& [index, replacement] : replacements) {
        discard(geometry_.layers[index], replacement, stream);
      }
    };
    for (size_t index = 0; index < geometry_.layers.size(); ++index) {
      const cache::LayerGeometry& layer = geometry_.layers[index];
      if (is_ring(layer)) {
        continue;
      }
      Allocation replacement;
      const Error error = allocate_layer(layer, new_rows, stream, replacement);
      if (error != Error::Ok) {
        roll_back();
        return error;
      }
      replacements.emplace_back(index, replacement);
    }
    for (const auto& [index, replacement] : replacements) {
      const Allocation& current = allocations_.at(static_cast<int64_t>(index));
      const size_t live_bytes =
          row_bytes(geometry_.layers[index]) * static_cast<size_t>(live_rows);
      if (live_bytes == 0) {
        continue;
      }
      for (const auto& [dst, src] :
           {std::pair{replacement.k, current.k},
            std::pair{replacement.v, current.v}}) {
        const cudaError_t copy_error = cudaMemcpyAsync(
            dst, src, live_bytes, cudaMemcpyDeviceToDevice, stream);
        if (copy_error != cudaSuccess) {
          ET_LOG(
              Error,
              "offgraph_kv: growth copy failed: %s",
              cudaGetErrorString(copy_error));
          roll_back();
          return Error::Internal;
        }
      }
    }
    for (auto& [index, replacement] : replacements) {
      Allocation& current = allocations_.at(static_cast<int64_t>(index));
      discard(geometry_.layers[index], current, stream);
      current = replacement;
    }
    metrics_.flat_capacity = new_rows;
    metrics_.growth_count++;
    // Every program sharing this cache now points at freed storage: drop the
    // bindings so each rebinds before its next run, and any captured CUDA
    // graph so it is captured again against the new storage. prefill usually
    // grows the cache while decode's graph sits idle, so this reaches every
    // handle, not only the one stepping now.
    //
    // Rebinding also resets AOTI's constant-fold state, which must be run
    // eagerly, so every graph-enabled handle gets at least one eager step
    // before it captures -- including one that was about to capture for the
    // first time, and without shortening a longer warmup still outstanding.
    bound_.clear();
    for (auto& entry : descriptors_) {
      CudaGraphState& graph = entry.first->cuda_graph_state;
      if (graph.phase == CudaGraphPhase::Replay) {
        graph.recapture();
      } else if (graph.phase == CudaGraphPhase::Warmup) {
        graph.warmup_remaining = std::max(graph.warmup_remaining, 1);
      }
    }
    ET_LOG(
        Info,
        "offgraph_kv: grew flat_capacity=%lld->%lld allocated_bytes=%lld "
        "growth_count=%lld",
        static_cast<long long>(old_rows),
        static_cast<long long>(new_rows),
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
    struct Compiled {
      std::string internal_name;
      size_t index;
    };
    std::unordered_map<std::string, Compiled> compiled;
    for (size_t index = 0; index < count; ++index) {
      const char* internal = nullptr;
      const char* original = nullptr;
      ET_CHECK_OK_OR_RETURN_ERROR(handle->get_constant_name(
          handle->container_handle, index, &internal));
      ET_CHECK_OK_OR_RETURN_ERROR(handle->get_constant_original_fqn(
          handle->container_handle, index, &original));
      if (internal && original && internal[0] && original[0]) {
        compiled.emplace(original, Compiled{internal, index});
      }
    }

    // A reload of the same handle replaces what it had rather than adding to
    // it, so its constants are never bound twice.
    auto& descriptors = descriptors_[handle];
    descriptors.clear();
    bound_.erase(handle);
    size_t found_layers = 0;
    for (size_t index = 0; index < geometry_.layers.size(); ++index) {
      const int64_t layer_id = static_cast<int64_t>(index);
      size_t found = 0;
      for (const auto& [suffix, is_value] :
           {std::pair{"k", false}, std::pair{"v", true}}) {
        const std::string name = fqn(layer_id, suffix);
        const auto it = compiled.find(name);
        if (it == compiled.end()) {
          continue;
        }
        ET_CHECK_OK_OR_RETURN_ERROR(check_compiled(
            handle, geometry_.layers[index], name, it->second.index));
        ++found;
        descriptors.push_back(
            Descriptor{it->second.internal_name, layer_id, is_value});
        discovered_fqns_.insert(name);
      }
      if (found == 2) {
        ++found_layers;
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

  // The program's kernels address its storage with the shape and dtype it was
  // compiled with, and AOTI binds external buffers without checking either.
  // So this cache's own idea of a layer -- dtype, heads, head dim, and rows
  // (capacity, or window + max_write - 1 for a ring) -- must match what the
  // program declared, or a step could write past the allocation.
  Error check_compiled(
      CudaDelegateHandle* handle,
      const cache::LayerGeometry& layer,
      const std::string& name,
      size_t index) const {
    ET_CHECK_OR_RETURN_ERROR(
        handle->get_constant_dtype && handle->get_constant_data_size,
        NotSupported,
        "offgraph_kv: AOTI constant metadata APIs are unavailable");
    int32_t dtype = 0;
    size_t data_size = 0;
    ET_CHECK_OK_OR_RETURN_ERROR(
        handle->get_constant_dtype(handle->container_handle, index, &dtype));
    ET_CHECK_OK_OR_RETURN_ERROR(handle->get_constant_data_size(
        handle->container_handle, index, &data_size));
    ET_CHECK_OR_RETURN_ERROR(
        dtype == static_cast<int32_t>(storage_dtype_),
        InvalidProgram,
        "offgraph_kv: %s is compiled as dtype %d but the cache stores %d",
        name.c_str(),
        static_cast<int>(dtype),
        static_cast<int>(storage_dtype_));
    const size_t expected =
        row_bytes(layer) * static_cast<size_t>(declared_rows(layer));
    ET_CHECK_OR_RETURN_ERROR(
        data_size == expected,
        InvalidProgram,
        "offgraph_kv: %s is compiled with %zu bytes but the cache declares %zu; "
        "the cache's capacity, max_write, window or head geometry does not "
        "match the program",
        name.c_str(),
        data_size,
        expected);
    return Error::Ok;
  }

  // Binds each storage constant with the shape the program declared (BSHD at
  // the maximum rows) over the current allocation, which may hold fewer rows.
  // That is safe because every access the program makes is bounded by kv_len
  // along the sequence, and prepare_step() has grown the allocation past it.
  Error bind(CudaDelegateHandle* handle) {
    if (bound_.find(handle) != bound_.end()) {
      return Error::Ok;
    }
    const std::vector<Descriptor>& descriptors = descriptors_[handle];
    Bound bound;
    bound.tensors.reserve(descriptors.size());
    std::vector<aoti::AOTInductorConstantMapEntry> pairs;
    pairs.reserve(descriptors.size());
    for (const Descriptor& descriptor : descriptors) {
      const cache::LayerGeometry& layer =
          geometry_.layers[static_cast<size_t>(descriptor.layer_id)];
      const Allocation& allocation = allocations_.at(descriptor.layer_id);
      const int64_t declared = declared_rows(layer);
      ET_CHECK_OR_RETURN_ERROR(
          allocation.rows <= declared,
          Internal,
          "offgraph_kv: layer %lld holds %lld rows, more than the %lld declared",
          static_cast<long long>(descriptor.layer_id),
          static_cast<long long>(allocation.rows),
          static_cast<long long>(declared));
      const int64_t heads = layer.n_kv_heads;
      const int64_t dim = layer.head_dim;
      const int64_t sizes[] = {1, declared, heads, dim};
      const int64_t strides[] = {declared * heads * dim, heads * dim, dim, 1};
      auto tensor = std::make_unique<SlimTensor>(from_blob(
          descriptor.is_value ? allocation.v : allocation.k,
          ::executorch::runtime::makeArrayRef(sizes, 4),
          ::executorch::runtime::makeArrayRef(strides, 4),
          storage_dtype_,
          slimc10::Device(slimc10::DeviceType::CUDA, device_)));
      pairs.push_back(
          {descriptor.internal_name.c_str(),
           reinterpret_cast<aoti::AtenTensorHandle>(tensor.get())});
      bound.tensors.push_back(std::move(tensor));
    }
    if (!pairs.empty()) {
      ET_CHECK_OK_OR_RETURN_ERROR(
          handle->update_user_managed_constant_buffer_pairs(
              handle->container_handle,
              pairs.data(),
              pairs.size(),
              /*use_inactive=*/false,
              /*validate_full_update=*/false));
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
  bool validated_{false};
  // The stream of the latest step, and so of the latest use of the storage.
  cudaStream_t stream_{cudaStreamPerThread};
  // Recorded on stream_ when a step commits; a later step on another stream
  // waits on it before touching the storage.
  cudaEvent_t last_step_done_{nullptr};
  Error error_{Error::Ok};
  OffGraphKVMetrics metrics_;
  std::unordered_map<int64_t, Allocation> allocations_;
  std::unordered_map<CudaDelegateHandle*, std::vector<Descriptor>> descriptors_;
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
  // A ring layer is sized window + max_write - 1, and the lowered graph bakes
  // that same expression in when it traces index_copy_. Defaulting max_write
  // to the window here would allocate short of what the graph addresses.
  if (!cfg.max_write) {
    for (const cache::LayerGeometry& layer : geometry.layers) {
      if (is_ring(layer)) {
        ET_LOG(Error, "offgraph_kv: ring layers require max_write");
        return nullptr;
      }
    }
  }
  return std::make_shared<CudaSequenceKVCache>(geometry, cfg, storage_dtype);
}

runtime::Error attach_offgraph_kv_cache(
    CudaDelegateHandle& handle,
    const char* cache_key,
    const std::optional<OffGraphKVStepWidth>& step_width) {
  handle.kv_cache_shared = cache::CacheRegistry::global().get(cache_key);
  ET_CHECK_OR_RETURN_ERROR(
      handle.kv_cache_shared != nullptr,
      InvalidArgument,
      "init: cache_key '%s' is not installed in the CacheRegistry",
      cache_key);
  handle.kv_cache = handle.kv_cache_shared->as<CudaKVCache>();
  ET_CHECK_OR_RETURN_ERROR(
      handle.kv_cache != nullptr,
      InvalidArgument,
      "init: cache under key '%s' is not a CUDA cache",
      cache_key);
  auto serves = handle.kv_cache->note_handle(&handle);
  ET_CHECK_OK_OR_RETURN_ERROR(serves.error());
  if (!serves.get()) {
    // An embedding or vision pass: it carries no KV storage, so it has no use
    // for the cache and must not be asked to step it.
    handle.kv_cache = nullptr;
    handle.kv_cache_shared.reset();
    return runtime::Error::Ok;
  }
  ET_CHECK_OR_RETURN_ERROR(
      step_width.has_value(),
      InvalidArgument,
      "off-graph KV cache needs an %s compile spec",
      kOffGraphKVStepWidthSpec);
  handle.kv_step_width = *step_width;
  return runtime::Error::Ok;
}

namespace {

// Publish this backend's off-graph KV cache layouts. A runner asks the factory
// for (backend_id, kind) and gets back a neutral Cache it can install, without
// naming any CUDA type. Lives beside the cache so a build without the LLM
// extension, which drops this file, registers nothing.
const bool cuda_cache_builders_registered = [] {
  const auto error = cache::CacheFactory::global().register_builder(
      kCudaBackendId,
      cache::kind::kSingle,
      [](const cache::CacheGeometry& geometry, const cache::CacheConfig& cfg) {
        return make_cuda_sequence_kv_cache(geometry, cfg);
      });
  if (error != runtime::Error::Ok) {
    ET_LOG(
        Error,
        "Failed to register cache builder for %s:%s",
        kCudaBackendId,
        cache::kind::kSingle);
  }
  return true;
}();

} // namespace

} // namespace executorch::backends::cuda
