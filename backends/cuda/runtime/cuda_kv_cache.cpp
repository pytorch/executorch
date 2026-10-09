/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cuda/runtime/cuda_kv_cache.h>

#include <limits>
#include <memory>
#include <mutex>
#include <vector>

#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/backends/aoti/slim/factory/from_blob.h>
#include <executorch/backends/cuda/runtime/cuda_kv_pool.h>
#include <executorch/extension/cuda/cuda_allocator.h>
#include <executorch/extension/llm/cache/cache_registry.h>
#include <executorch/extension/llm/cache/sequence_cache.h>
#include <executorch/runtime/platform/log.h>

namespace executorch::backends::cuda {
namespace {

namespace slimc10 = ::executorch::backends::aoti::slim::c10;
using ::executorch::runtime::Error;

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

// Slots a ring layer needs to serve one step of up to max_write tokens: the
// step writes all of them before attending, and its earliest query still
// reads back window - 1 positions. Same formula as cache::RingPolicy and as
// ring_physical_capacity() in backends/cuda/passes/lower_offgraph_kv.py.
// make_cuda_sequence_kv_cache() refuses ring layers without max_write.
std::vector<CudaKVPool::Layer> sequence_pool_layers(
    const cache::CacheGeometry& geometry,
    const cache::CacheConfig& cfg) {
  std::vector<CudaKVPool::Layer> layers;
  layers.reserve(geometry.layers.size());
  for (const cache::LayerGeometry& layer : geometry.layers) {
    const bool ring = is_ring(layer);
    layers.push_back(CudaKVPool::Layer{
        layer.n_kv_heads,
        layer.head_dim,
        ring ? static_cast<int64_t>(layer.policy.window) +
                cfg.max_write.value_or(1) - 1
             : static_cast<int64_t>(cfg.capacity),
        /*growable=*/!ring});
  }
  return layers;
}

// One sequence's KV storage on one CUDA device.
//
// The neutral base owns the logical length, admission and rewind; the pool
// owns the bytes and the AOTI bindings that point the compiled program at
// them. Physical slot math stays in the compiled program, on device, so a
// captured CUDA graph replays against the current positions rather than the
// ones live at capture.
class CudaSequenceKVCache final : public cache::SequenceCache,
                                  public CudaKVCache {
 public:
  CudaSequenceKVCache(
      const cache::CacheGeometry& geometry,
      const cache::CacheConfig& cfg,
      slimc10::ScalarType storage_dtype)
      : cache::SequenceCache(geometry, cfg),
        geometry_(geometry),
        pool_(
            sequence_pool_layers(geometry, cfg),
            /*side_buffers=*/{},
            storage_dtype,
            cfg.initial_capacity) {}

  CudaSequenceKVCache(const CudaSequenceKVCache&) = delete;
  CudaSequenceKVCache& operator=(const CudaSequenceKVCache&) = delete;
  CudaSequenceKVCache(CudaSequenceKVCache&&) = delete;
  CudaSequenceKVCache& operator=(CudaSequenceKVCache&&) = delete;

  ~CudaSequenceKVCache() override = default;

  // cache::SequenceControl. Keeps the storage: a reset session reuses the
  // grown allocations, so it neither regrows nor invalidates captured graphs.
  void clear() override {
    std::lock_guard<std::mutex> guard(mutex_);
    cache::SequenceCache::clear();
    ET_LOG(
        Info,
        "offgraph_kv: reset flat_capacity=%lld allocated_bytes=%lld",
        static_cast<long long>(pool_.rows()),
        static_cast<long long>(pool_.allocated_bytes()));
  }

  // CudaKVCache.
  runtime::Result<bool> note_handle(CudaDelegateHandle* handle) override {
    std::lock_guard<std::mutex> guard(mutex_);
    auto serves = pool_.note_handle(handle);
    if (!serves.ok()) {
      error_ = serves.error();
    }
    return serves;
  }

  void forget_handle(CudaDelegateHandle* handle) override {
    std::lock_guard<std::mutex> guard(mutex_);
    pool_.forget_handle(handle);
  }

  Error rebind_for_execute(CudaDelegateHandle* handle) override {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!pool_.serves(handle)) {
      return Error::Ok; // a handle with no off-graph storage
    }
    ET_CHECK_OK_OR_RETURN_ERROR(error_);
    ET_CHECK_OR_RETURN_ERROR(
        pool_.allocated(),
        InvalidState,
        "offgraph_kv: prepare_step must run before execute");
    return pool_.bind(handle);
  }

  Error prepare_step(int64_t write_length, cudaStream_t stream) override {
    std::lock_guard<std::mutex> guard(mutex_);
    ET_CHECK_OK_OR_RETURN_ERROR(check_write_length(write_length));
    ET_CHECK_OK_OR_RETURN_ERROR(error_);
    if (!validated_) {
      // Whole-model check, so it cannot run until every program has been
      // loaded and registered. The first step is the earliest moment that is
      // guaranteed.
      const Error valid = pool_.validate();
      if (valid != Error::Ok) {
        error_ = valid;
        return valid;
      }
      validated_ = true;
    }
    const int position = length();
    ET_CHECK_OK_OR_RETURN_ERROR(admit(position, write_length));
    return pool_.prepare(position + write_length, position, stream);
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
    return pool_.mark_step_done();
  }

  OffGraphKVMetrics metrics() const override {
    std::lock_guard<std::mutex> guard(mutex_);
    OffGraphKVMetrics metrics;
    metrics.logical_length = length();
    metrics.flat_capacity = pool_.rows();
    metrics.growth_count = pool_.growth_count();
    metrics.allocated_bytes = pool_.allocated_bytes();
    return metrics;
  }

 protected:
  void* face(cache::FaceId id) override {
    if (void* p = cache::SequenceCache::face(id)) {
      return p;
    }
    return cache::expose<CudaKVCache>(this, id);
  }

 private:
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

  // Guards everything below. The engine serialises its own calls, but the
  // delegate reaches note_handle/rebind from whichever thread loads or runs a
  // method.
  mutable std::mutex mutex_;

  cache::CacheGeometry geometry_;
  CudaKVPool pool_;
  bool validated_{false};
  Error error_{Error::Ok};
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
