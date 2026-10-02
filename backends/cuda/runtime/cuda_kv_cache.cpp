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
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/cuda/runtime/cuda_kv_pool.h>
#include <executorch/extension/llm/cache/cache_registry.h>
#include <executorch/extension/llm/cache/cell_cache.h>
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


// The window each layer attends over, 0 = its whole history. Layers agreeing
// on a window read one mask, as the lowering pass declares it.
int layer_window(const cache::LayerGeometry& layer) {
  return is_ring(layer) ? layer.policy.window : 0;
}

std::string cell_mask_fqn(int window) {
  return "__et_offgraph_kv_mask_w" + std::to_string(window);
}

// Side buffers of the cell layout, in the order CudaCellCache addresses them:
// cells, read_len, then one mask per distinct window. Names, dtypes and shapes
// match LowerOffGraphKVPass's cell layout.
constexpr size_t kCellsBuffer = 0;
constexpr size_t kReadLenBuffer = 1;
constexpr size_t kFirstMaskBuffer = 2;

// Many sequences over one pool of per-token cells on one CUDA device.
//
// The neutral base owns the cell table: placement, ownership, and the verbs a
// runner drives between forwards. The pool owns the bytes. Before each
// forward, prepare_step places the declared tokens and writes the placement
// (cells), the extent (read_len) and one visibility mask per window into
// buffers the program reads at fixed addresses -- so a captured CUDA graph
// serves any mix of sequences, and only growth forces a recapture.
class CudaCellCache final : public cache::CellCache, public CudaKVCache {
 public:
  CudaCellCache(
      const cache::CacheGeometry& geometry,
      const cache::CacheConfig& cfg,
      slimc10::ScalarType storage_dtype,
      std::vector<int> windows)
      : cache::CellCache(geometry, cfg),
        max_write_(*cfg.max_write),
        windows_(std::move(windows)),
        pool_(
            pool_layers(geometry, cfg),
            side_buffers(cfg, windows_),
            storage_dtype,
            cfg.initial_capacity) {
    // The first layer of each window: placing through it memoizes that
    // window's step for the forward.
    for (const int window : windows_) {
      for (size_t index = 0; index < geometry.layers.size(); ++index) {
        if (layer_window(geometry.layers[index]) == window) {
          window_layers_.push_back(static_cast<int>(index));
          break;
        }
      }
    }
  }

  CudaCellCache(const CudaCellCache&) = delete;
  CudaCellCache& operator=(const CudaCellCache&) = delete;
  CudaCellCache(CudaCellCache&&) = delete;
  CudaCellCache& operator=(CudaCellCache&&) = delete;

  ~CudaCellCache() override = default;

  // -- CacheControl / BatchControl, serialized against the delegate. Keeps the
  // storage: a reset reuses the grown pools.

  void clear() override {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    cache::CellCache::clear();
    step_seq_ids_.clear();
  }

  bool declare_step(const std::vector<int32_t>& seq_ids) override {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    if (!cache::CellCache::declare_step(seq_ids)) {
      step_seq_ids_.clear();
      return false;
    }
    step_seq_ids_ = seq_ids;
    return true;
  }

  std::optional<int32_t> seq_new() override {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    return cache::CellCache::seq_new();
  }

  std::optional<int32_t> seq_clone(int32_t src, std::optional<int> upto)
      override {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    return cache::CellCache::seq_clone(src, upto);
  }

  bool seq_rm(int32_t seq_id) override {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    return cache::CellCache::seq_rm(seq_id);
  }

  bool rewind(int32_t seq_id, int position) override {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    return cache::CellCache::rewind(seq_id, position);
  }

  // -- CudaKVCache.

  runtime::Result<bool> note_handle(CudaDelegateHandle* handle) override {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    auto serves = pool_.note_handle(handle);
    if (!serves.ok()) {
      error_ = serves.error();
    }
    return serves;
  }

  void forget_handle(CudaDelegateHandle* handle) override {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    pool_.forget_handle(handle);
  }

  Error rebind_for_execute(CudaDelegateHandle* handle) override {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    if (!pool_.serves(handle)) {
      return Error::Ok;
    }
    ET_CHECK_OK_OR_RETURN_ERROR(error_);
    ET_CHECK_OR_RETURN_ERROR(
        pool_.allocated(),
        InvalidState,
        "offgraph_kv: prepare_step must run before execute");
    return pool_.bind(handle);
  }

  // Places the declared step and writes it where the program reads it. Runs on
  // the host between executes, so its copies never land inside a capture.
  Error prepare_step(int64_t write_length, cudaStream_t stream) override {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    ET_CHECK_OK_OR_RETURN_ERROR(error_);
    if (!validated_) {
      const Error valid = pool_.validate();
      if (valid != Error::Ok) {
        error_ = valid;
        return valid;
      }
      validated_ = true;
    }
    const int width = static_cast<int>(write_length);
    ET_CHECK_OR_RETURN_ERROR(
        write_length > 0 && write_length <= max_write_,
        InvalidArgument,
        "offgraph_kv: a %lld-token step is outside [1, %d]",
        static_cast<long long>(write_length),
        max_write_);
    ET_CHECK_OR_RETURN_ERROR(
        static_cast<size_t>(width) == step_seq_ids_.size(),
        InvalidState,
        "offgraph_kv: the step carries %d tokens, declare_step declared %zu",
        width,
        step_seq_ids_.size());

    // Every declared token continues its sequence, so the positions follow
    // from the declaration; nothing is read back from the device.
    positions_.resize(width);
    std::vector<int32_t> next(kMaxSeqs, -1);
    for (int i = 0; i < width; ++i) {
      const int32_t seq_id = step_seq_ids_[i];
      if (next[seq_id] < 0) {
        next[seq_id] = cache::CellCache::pos(seq_id);
      }
      positions_[i] = next[seq_id]++;
    }

    // Grow before placing, to where placement could reach at most, so a
    // failed allocation leaves the table as it was. Lowest-free placement
    // never takes a cell past used_end + width.
    const int live = used_end();
    ET_CHECK_OK_OR_RETURN_ERROR(pool_.prepare(
        std::min<int64_t>(capacity(), static_cast<int64_t>(live) + width),
        live,
        stream));

    const cache::CellStep* first = nullptr;
    for (size_t index = 0; index < windows_.size(); ++index) {
      const cache::CellStep* step =
          place_step(window_layers_[index], positions_.data(), width);
      ET_CHECK_OR_RETURN_ERROR(
          step != nullptr,
          InvalidArgument,
          "offgraph_kv: the declared step does not place");
      if (first == nullptr) {
        first = step;
        ET_CHECK_OK_OR_RETURN_ERROR(write_placement(*step, stream));
      }
      ET_CHECK_OK_OR_RETURN_ERROR(
          write_mask(kFirstMaskBuffer + index, *step, stream));
    }
    step_seq_ids_.clear();
    return Error::Ok;
  }

  // The cells were claimed when the step was placed; the forward only filled
  // them.
  Error commit_step(int64_t write_length) override {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    ET_CHECK_OR_RETURN_ERROR(
        write_length > 0, InvalidArgument, "write length must be positive");
    ET_CHECK_OK_OR_RETURN_ERROR(error_);
    return pool_.mark_step_done();
  }

  // logical_length is the read extent: every occupied cell is below it.
  OffGraphKVMetrics metrics() const override {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    OffGraphKVMetrics metrics;
    metrics.logical_length = used_end();
    metrics.flat_capacity = pool_.rows();
    metrics.growth_count = pool_.growth_count();
    metrics.allocated_bytes = pool_.allocated_bytes();
    return metrics;
  }

 protected:
  void* face(cache::FaceId id) override {
    if (void* p = cache::CellCache::face(id)) {
      return p;
    }
    return cache::expose<CudaKVCache>(this, id);
  }

 private:
  static std::vector<CudaKVPool::Layer> pool_layers(
      const cache::CacheGeometry& geometry,
      const cache::CacheConfig& cfg) {
    std::vector<CudaKVPool::Layer> layers;
    layers.reserve(geometry.layers.size());
    for (const cache::LayerGeometry& layer : geometry.layers) {
      layers.push_back(CudaKVPool::Layer{
          layer.n_kv_heads,
          layer.head_dim,
          static_cast<int64_t>(cfg.capacity),
          /*growable=*/true});
    }
    return layers;
  }

  static std::vector<CudaKVPool::SideBuffer> side_buffers(
      const cache::CacheConfig& cfg,
      const std::vector<int>& windows) {
    const int64_t max_write = *cfg.max_write;
    std::vector<CudaKVPool::SideBuffer> buffers{
        {"__et_offgraph_kv_cells", slimc10::ScalarType::Long, {max_write}},
        {"__et_offgraph_kv_read_len", slimc10::ScalarType::Long, {1}},
    };
    for (const int window : windows) {
      buffers.push_back(
          {cell_mask_fqn(window),
           slimc10::ScalarType::Bool,
           {1, 1, max_write, static_cast<int64_t>(cfg.capacity)}});
    }
    return buffers;
  }

  // Pageable sources: the copy has consumed them when the call returns, so
  // the host vectors may be reused by the next step at once.
  Error write_placement(const cache::CellStep& step, cudaStream_t stream) {
    staged_cells_.assign(step.cells.begin(), step.cells.end());
    const int64_t read_len = step.read_len;
    for (const auto& [index, src, bytes] :
         {std::tuple{
              kCellsBuffer,
              static_cast<const void*>(staged_cells_.data()),
              staged_cells_.size() * sizeof(int64_t)},
          std::tuple{
              kReadLenBuffer,
              static_cast<const void*>(&read_len),
              sizeof(int64_t)}}) {
      const cudaError_t error = cudaMemcpyAsync(
          pool_.side_buffer(index), src, bytes, cudaMemcpyHostToDevice, stream);
      ET_CHECK_OR_RETURN_ERROR(
          error == cudaSuccess,
          Internal,
          "offgraph_kv: cannot write the step placement: %s",
          cudaGetErrorString(error));
    }
    return Error::Ok;
  }

  // Rows [0, length) over columns [0, read_len); the program bounds its sweep
  // by read_len, so columns past it keep whatever an earlier step left.
  Error write_mask(size_t buffer, const cache::CellStep& step, cudaStream_t stream) {
    if (step.read_len == 0) {
      return Error::Ok;
    }
    const size_t row = static_cast<size_t>(capacity());
    const cudaError_t error = cudaMemcpy2DAsync(
        pool_.side_buffer(buffer),
        row,
        step.mask_bits.data(),
        static_cast<size_t>(step.read_len),
        static_cast<size_t>(step.read_len),
        static_cast<size_t>(step.length),
        cudaMemcpyHostToDevice,
        stream);
    ET_CHECK_OR_RETURN_ERROR(
        error == cudaSuccess,
        Internal,
        "offgraph_kv: cannot write the step mask: %s",
        cudaGetErrorString(error));
    return Error::Ok;
  }

  // Guards everything below and the base's table. The runner drives the verbs
  // and the delegate the steps, from whichever threads they run on. Recursive
  // because the base's verbs call each other virtually: seq_clone takes its
  // new id through seq_new.
  mutable std::recursive_mutex mutex_;

  const int max_write_;
  const std::vector<int> windows_;
  std::vector<int> window_layers_;
  CudaKVPool pool_;
  std::vector<int32_t> step_seq_ids_;
  std::vector<int32_t> positions_;
  std::vector<int64_t> staged_cells_;
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

std::shared_ptr<cache::Cache> make_cuda_cell_kv_cache(
    const cache::CacheGeometry& geometry,
    const cache::CacheConfig& cfg) {
  slimc10::ScalarType storage_dtype = slimc10::ScalarType::BFloat16;
  if (!cache::valid(geometry, cfg) || cfg.initial_capacity <= 0 ||
      cfg.initial_capacity > cfg.capacity ||
      !storage_dtype_of(cfg.kv_dtype, storage_dtype)) {
    ET_LOG(Error, "offgraph_kv: invalid cache geometry or config");
    return nullptr;
  }
  // The step buffers are declared [max_write] and [max_write, max_cells]; a
  // cache without the program's widest step cannot address them.
  if (!cfg.max_write || *cfg.max_write <= 0 || *cfg.max_write > cfg.capacity) {
    ET_LOG(Error, "offgraph_kv: the cell layout requires max_write");
    return nullptr;
  }
  std::set<int> windows;
  for (const cache::LayerGeometry& layer : geometry.layers) {
    windows.insert(layer_window(layer));
  }
  return std::make_shared<CudaCellCache>(
      geometry,
      cfg,
      storage_dtype,
      std::vector<int>(windows.begin(), windows.end()));
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
  const struct {
    const char* kind;
    cache::CacheBuilder builder;
  } builders[] = {
      {cache::kind::kSingle, make_cuda_sequence_kv_cache},
      {cache::kind::kBatchedCell, make_cuda_cell_kv_cache},
      // The cell layout is the batch layout this backend serves.
      {cache::kind::kBatched, make_cuda_cell_kv_cache},
  };
  for (const auto& entry : builders) {
    const auto error = cache::CacheFactory::global().register_builder(
        kCudaBackendId, entry.kind, entry.builder);
    if (error != runtime::Error::Ok) {
      ET_LOG(
          Error,
          "Failed to register cache builder for %s:%s",
          kCudaBackendId,
          entry.kind);
    }
  }
  return true;
}();

} // namespace

} // namespace executorch::backends::cuda
