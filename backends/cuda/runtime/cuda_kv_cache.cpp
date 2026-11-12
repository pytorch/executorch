/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cuda/runtime/cuda_kv_cache.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/backends/aoti/slim/factory/from_blob.h>
#include <executorch/backends/cuda/runtime/cuda_kv_pool.h>
#include <executorch/extension/cuda/cuda_allocator.h>
#include <executorch/extension/cuda/device_guard.h>
#include <executorch/extension/llm/cache/cache_registry.h>
#include <executorch/extension/llm/cache/cell_cache.h>
#include <executorch/extension/llm/cache/sequence_cache.h>
#include <executorch/runtime/platform/log.h>

namespace executorch::backends::cuda {
namespace {

namespace slimc10 = ::executorch::backends::aoti::slim::c10;
using ::executorch::extension::cuda::CUDAGuard;
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
        // A ring cannot wrap before it holds all its slots, so until then it
        // grows as a flat layer does, and a growth copies a plain prefix.
        /*growable=*/true});
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
    // The pool's shared rows can exceed a small flat layer when a ring
    // ceiling dominates, so clamp to what the flat layers hold.
    metrics.flat_capacity = std::min<int64_t>(pool_.rows(), capacity());
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
//
// A program may be wider than the step it runs: a static method of width W
// serves any T <= W declared tokens, and rows T..W-1 are padding. Padding is
// never declared to the table. Its rows write their K/V to one scratch row, the
// last allocated pool row, which the table never hands out and every read_len
// stops short of, so no real row ever reads it. Each step rewrites all W rows:
// a padding row left from a wider step would otherwise still name a live cell
// and overwrite another sequence's history.
class CudaCellCache final : public cache::CellCache, public CudaKVCache {
 public:
  CudaCellCache(
      const cache::CacheGeometry& geometry,
      const cache::CacheConfig& cfg,
      slimc10::ScalarType storage_dtype,
      std::vector<int> windows)
      : cache::CellCache(geometry, table_config(cfg)),
        pool_cells_(cfg.capacity),
        max_write_(*cfg.max_write),
        dflash_block_size_(geometry.dflash_block_size),
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

  // The staging may still be read by the last step's uploads.
  ~CudaCellCache() override {
    if (staged_ != nullptr) {
      (void)cudaEventSynchronize(staged_);
      (void)cudaEventDestroy(staged_);
    }
    if (staging_ != nullptr) {
      (void)cudaFreeHost(staging_);
    }
  }

  // -- CacheControl / BatchControl, serialized against the delegate. Keeps the
  // storage: a reset reuses the grown pools.

  void clear() override {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    if (dflash_block_size_ && pool_.allocated()) {
      // A failed forward may be abandoned on another host thread, where
      // cudaStreamPerThread no longer names the originating stream.
      if (pool_.synchronize(/*all_streams=*/true) != Error::Ok) {
        // Recycling cells a forward may still write is unsafe; a later clear()
        // retries the fence.
        poisoned_ = true;
        return;
      }
    }
    cache::CellCache::clear();
    step_seq_ids_.clear();
    active_owners_.clear();
    pending_width_ = 0;
    poisoned_ = false;
  }

  bool declare_step(const std::vector<int32_t>& seq_ids) override {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    // A rejected declaration leaves an earlier accepted one standing, as the
    // base does.
    if (poisoned_ ||
        (dflash_block_size_ &&
         (pending_width_ != 0 ||
          seq_ids.size() > static_cast<size_t>(max_write_))) ||
        !cache::CellCache::declare_step(seq_ids)) {
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
    if (dflash_block_size_ && (pending_width_ != 0 || poisoned_)) {
      return std::nullopt;
    }
    return cache::CellCache::seq_clone(src, upto);
  }

  bool seq_rm(int32_t seq_id) override {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    if (dflash_block_size_ && pending_width_ != 0) {
      return false;
    }
    return cache::CellCache::seq_rm(seq_id);
  }

  bool rewind(int32_t seq_id, int position) override {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    if (dflash_block_size_ && (pending_width_ != 0 || poisoned_)) {
      return false;
    }
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
    ET_CHECK_OR_RETURN_ERROR(
        !dflash_block_size_ || (!poisoned_ && pending_width_ > 0),
        InvalidState,
        "offgraph_kv: execute requires a prepared, healthy draft step");
    const Error bound = pool_.bind(handle);
    if (bound != Error::Ok && dflash_block_size_) {
      poisoned_ = true;
    }
    return bound;
  }

  // Places the declared step and writes it where the program reads it. Runs on
  // the host between executes, so its copies never land inside a capture.
  Error prepare_step(int64_t write_length, cudaStream_t stream) override {
    std::lock_guard<std::recursive_mutex> guard(mutex_);
    ET_CHECK_OK_OR_RETURN_ERROR(error_);
    ET_CHECK_OR_RETURN_ERROR(
        !poisoned_ && (!dflash_block_size_ || pending_width_ == 0),
        InvalidState,
        "offgraph_kv: uncommitted or failed draft co-batch; clear required");
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
    // A static draft program pads its tail with scratch rows, as a static
    // target forward does; every query row stays a declared block row.
    const int tokens = static_cast<int>(step_seq_ids_.size());
    ET_CHECK_OR_RETURN_ERROR(
        tokens > 0 && tokens <= width,
        InvalidState,
        "offgraph_kv: a %d-token program cannot run %d declared tokens",
        width,
        tokens);

    // Every declared token continues its sequence, so the positions follow
    // from the declaration; nothing is read back from the device.
    positions_.resize(tokens);
    std::vector<int32_t> next(kMaxSeqs, -1);
    for (int i = 0; i < tokens; ++i) {
      const int32_t seq_id = step_seq_ids_[i];
      if (next[seq_id] < 0) {
        next[seq_id] = cache::CellCache::pos(seq_id);
      }
      positions_[i] = next[seq_id]++;
    }

    // Grow before placing, to where placement could reach at most, so a
    // failed allocation leaves the table as it was. Lowest-free placement
    // fills holes first, so the step reaches past used_end only once every
    // cell below it is occupied: at most occupied + tokens. One more row is
    // the scratch row, which therefore lies past every cell and every
    // read_len.
    const int64_t live = used_end();
    const int64_t occupied = capacity() - free_cells();
    ET_CHECK_OK_OR_RETURN_ERROR(pool_.prepare(
        std::min<int64_t>(
            pool_cells_, std::max<int64_t>(live, occupied + tokens) + 1),
        live,
        stream));
    const int64_t scratch = pool_.rows() - 1;

    if (dflash_block_size_) {
      active_owners_.clear();
      for (const int32_t owner : step_seq_ids_) {
        if (active_owners_.empty() || active_owners_.back() != owner) {
          active_owners_.push_back(owner);
        }
      }
      pending_width_ = width;
    }
    std::vector<const cache::CellStep*> steps;
    steps.reserve(windows_.size());
    for (size_t index = 0; index < windows_.size(); ++index) {
      const cache::CellStep* step =
          place_step(window_layers_[index], positions_.data(), tokens);
      if (step == nullptr) {
        poisoned_ = dflash_block_size_.has_value();
        return Error::InvalidArgument;
      }
      steps.push_back(step);
    }
    const Error uploaded = upload_step(steps, width, scratch, stream);
    if (uploaded != Error::Ok) {
      poisoned_ = dflash_block_size_.has_value();
      return uploaded;
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
    if (!dflash_block_size_) {
      return pool_.mark_step_done();
    }
    if (poisoned_ || pending_width_ == 0 || write_length != pending_width_) {
      poisoned_ = true;
      return Error::InvalidState;
    }
    const Error marked = pool_.mark_step_done();
    const Error fenced = marked == Error::Ok ? pool_.synchronize() : marked;
    if (fenced != Error::Ok) {
      poisoned_ = true;
      return fenced;
    }
    for (const int32_t owner : active_owners_) {
      if (!cache::CellCache::rewind(
              owner, cache::CellCache::pos(owner) - *dflash_block_size_)) {
        poisoned_ = true;
        return Error::InvalidState;
      }
    }
    active_owners_.clear();
    pending_width_ = 0;
    return Error::Ok;
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
  // The table's cells are the pool's rows but one: the last is scratch.
  static cache::CacheConfig table_config(const cache::CacheConfig& cfg) {
    cache::CacheConfig table = cfg;
    table.capacity = cfg.capacity - 1;
    return table;
  }

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

  // Writes the step's cells, read_len and one mask per window where the
  // program reads them, padded to the program's width. Padding tokens write
  // to the scratch row. Padding mask rows repeat the last real row: they must
  // see some cell, so every value they carry stays finite. A padding row with
  // nothing to attend would turn NaN, and a batch-wide reduction (an
  // activation scale, say) would spread it to the real rows.
  //
  // The copies read pinned staging the cache owns, never the base's step,
  // which the next verb invalidates: an async copy from pageable memory may
  // still be reading it after the call returns. The staging is refilled only
  // once the previous step's copies are done.
  Error upload_step(
      const std::vector<const cache::CellStep*>& steps,
      int width,
      int64_t scratch,
      cudaStream_t stream) {
    const cache::CellStep& first = *steps.front();
    const size_t rows = static_cast<size_t>(width);
    const size_t tokens = first.cells.size();
    const size_t queries =
        first.query_rows.empty() ? tokens : first.query_rows.size();
    const size_t mask_rows = dflash_block_size_ ? queries : rows;
    const size_t read_len = static_cast<size_t>(first.read_len);
    const size_t cells_bytes = rows * sizeof(int64_t);
    const size_t mask_bytes = mask_rows * read_len;
    ET_CHECK_OK_OR_RETURN_ERROR(reserve_staging(
        cells_bytes + sizeof(int64_t) + steps.size() * mask_bytes));
    auto* cells = reinterpret_cast<int64_t*>(staging_);
    std::copy(first.cells.begin(), first.cells.end(), cells);
    std::fill(cells + tokens, cells + rows, scratch);
    cells[rows] = first.read_len;
    uint8_t* masks = staging_ + cells_bytes + sizeof(int64_t);
    for (size_t index = 0; index < steps.size(); ++index) {
      uint8_t* mask = masks + index * mask_bytes;
      const cache::CellStep& step = *steps[index];
      for (size_t row = 0; row < queries; ++row) {
        const size_t source =
            step.query_rows.empty() ? row : step.query_rows[row];
        std::memcpy(
            mask + row * read_len,
            step.mask_bits.data() + source * read_len,
            read_len);
      }
      const uint8_t* last = mask + (queries - 1) * read_len;
      for (size_t row = queries; row < mask_rows; ++row) {
        std::memcpy(mask + row * read_len, last, read_len);
      }
    }
    const Error enqueued =
        enqueue_uploads(steps.size(), rows, mask_rows, read_len, stream);
    // Fenced even when a copy failed: those issued before it may still read
    // the staging.
    const Error fenced = fence_staging(stream);
    ET_CHECK_OK_OR_RETURN_ERROR(enqueued);
    return fenced;
  }

  // Masks fill rows [0, width) over columns [0, read_len); the program bounds
  // its sweep by read_len, so columns past it keep whatever an earlier step
  // left.
  Error enqueue_uploads(
      size_t windows,
      size_t width,
      size_t mask_rows,
      size_t read_len,
      cudaStream_t stream) {
    const size_t cells_bytes = width * sizeof(int64_t);
    for (const auto& [index, offset, bytes] :
         {std::tuple{kCellsBuffer, size_t{0}, cells_bytes},
          std::tuple{kReadLenBuffer, cells_bytes, sizeof(int64_t)}}) {
      const cudaError_t error = cudaMemcpyAsync(
          pool_.side_buffer(index),
          staging_ + offset,
          bytes,
          cudaMemcpyHostToDevice,
          stream);
      ET_CHECK_OR_RETURN_ERROR(
          error == cudaSuccess,
          Internal,
          "offgraph_kv: cannot write the step placement: %s",
          cudaGetErrorString(error));
    }
    const uint8_t* masks = staging_ + cells_bytes + sizeof(int64_t);
    for (size_t index = 0; index < windows && read_len > 0; ++index) {
      const cudaError_t error = cudaMemcpy2DAsync(
          pool_.side_buffer(kFirstMaskBuffer + index),
          static_cast<size_t>(pool_cells_),
          masks + index * mask_rows * read_len,
          read_len,
          read_len,
          mask_rows,
          cudaMemcpyHostToDevice,
          stream);
      ET_CHECK_OR_RETURN_ERROR(
          error == cudaSuccess,
          Internal,
          "offgraph_kv: cannot write the step mask: %s",
          cudaGetErrorString(error));
    }
    return Error::Ok;
  }

  // Waits for the previous step's copies to finish reading the staging, then
  // makes it hold at least `bytes`.
  Error reserve_staging(size_t bytes) {
    if (staged_ != nullptr) {
      const cudaError_t error = cudaEventSynchronize(staged_);
      ET_CHECK_OR_RETURN_ERROR(
          error == cudaSuccess,
          Internal,
          "offgraph_kv: cannot wait for the previous step's uploads: %s",
          cudaGetErrorString(error));
    }
    if (bytes <= staging_bytes_) {
      return Error::Ok;
    }
    const size_t grown = std::max(bytes, 2 * staging_bytes_);
    if (staging_ != nullptr) {
      (void)cudaFreeHost(staging_);
      staging_ = nullptr;
      staging_bytes_ = 0;
    }
    void* data = nullptr;
    const cudaError_t error = cudaMallocHost(&data, grown);
    ET_CHECK_OR_RETURN_ERROR(
        error == cudaSuccess,
        MemoryAllocationFailed,
        "offgraph_kv: cannot allocate %zu bytes of pinned staging: %s",
        grown,
        cudaGetErrorString(error));
    staging_ = static_cast<uint8_t*>(data);
    staging_bytes_ = grown;
    return Error::Ok;
  }

  // Marks where this step's copies end, for the next step to wait on. If it
  // cannot be marked, waits for them here instead.
  Error fence_staging(cudaStream_t stream) {
    cudaError_t error = cudaSuccess;
    if (staged_ == nullptr) {
      error = cudaEventCreateWithFlags(&staged_, cudaEventDisableTiming);
    }
    if (error == cudaSuccess) {
      error = cudaEventRecord(staged_, stream);
    }
    if (error != cudaSuccess) {
      (void)cudaStreamSynchronize(stream);
      ET_LOG(
          Error,
          "offgraph_kv: cannot fence the step uploads: %s",
          cudaGetErrorString(error));
      return Error::Internal;
    }
    return Error::Ok;
  }

  // Guards everything below and the base's table. The runner drives the verbs
  // and the delegate the steps, from whichever threads they run on. Recursive
  // because the base's verbs call each other virtually: seq_clone takes its
  // new id through seq_new.
  mutable std::recursive_mutex mutex_;

  // The program's declared cells: the pool's rows at most, and the mask row
  // pitch.
  const int pool_cells_;
  const int max_write_;
  const std::optional<int> dflash_block_size_;
  const std::vector<int> windows_;
  std::vector<int> window_layers_;
  CudaKVPool pool_;
  std::vector<int32_t> step_seq_ids_;
  std::vector<int32_t> positions_;
  std::vector<int32_t> active_owners_;
  int pending_width_{0};
  bool poisoned_{false};
  // Pinned host staging for one step's uploads, and the event marking where
  // the last step's copies from it end.
  uint8_t* staging_{nullptr};
  size_t staging_bytes_{0};
  cudaEvent_t staged_{nullptr};
  bool validated_{false};
  Error error_{Error::Ok};
};

// Batched-sequence layout: every sequence keeps a private history in its own
// column of each layer's storage, as MLX's batched-sequence cache keeps one
// buffer per sequence. A layer's storage is [rows, columns, H, D] (declared
// [1, rows * columns, H, D]): row = the position, or position % rows on a ring
// layer, so each sequence is contiguous by position and attention reads only
// its own column, bounded by its own length. The last column is scratch,
// written by static-width padding and never read.
//
// The neutral SequenceCache plans each sequence: admission against its
// capacity, ring limits, and rewind floors. The pool owns the bytes; rows grow
// together for every column. Before each forward, prepare_step writes each
// token's column (seq_cols) and one (q_start, q_len, column, length) row per
// run of one sequence's query rows (seq_info) into buffers the program reads
// at fixed addresses, so a captured CUDA graph serves any mix of sequences.
//
// A DFlash draft step declares, per owner, its new context then its block;
// only the block is queried, it sees all of that owner's history, and it is
// dropped again after the forward, as on the cell layout.
constexpr char kSeqColsFqn[] = "__et_offgraph_kv_seq_cols";
constexpr char kSeqInfoFqn[] = "__et_offgraph_kv_seq_info";
constexpr size_t kSeqColsBuffer = 0;
constexpr size_t kSeqInfoBuffer = 1;

class CudaBatchedSequenceCache final : public cache::Cache,
                                       public cache::BatchControl,
                                       public CudaKVCache {
 public:
  CudaBatchedSequenceCache(
      const cache::CacheGeometry& geometry,
      const cache::CacheConfig& cfg,
      slimc10::ScalarType storage_dtype)
      : geometry_(geometry),
        cfg_(cfg),
        max_seqs_(*cfg.max_seqs),
        columns_(*cfg.max_seqs + 1),
        max_write_(*cfg.max_write),
        max_segments_(batched_sequence_max_segments(*cfg.max_seqs)),
        dflash_block_size_(geometry.dflash_block_size),
        pool_(
            pool_layers(geometry, cfg),
            {{kSeqColsFqn, slimc10::ScalarType::Long, {*cfg.max_write}},
             {kSeqInfoFqn,
              slimc10::ScalarType::Int,
              {batched_sequence_max_segments(*cfg.max_seqs), 4}}},
            storage_dtype,
            static_cast<int64_t>(cfg.initial_capacity) * (*cfg.max_seqs + 1)) {
    for (const cache::LayerGeometry& layer : geometry.layers) {
      layer_rows_.push_back(rows_of(layer, cfg));
    }
  }

  CudaBatchedSequenceCache(const CudaBatchedSequenceCache&) = delete;
  CudaBatchedSequenceCache& operator=(const CudaBatchedSequenceCache&) = delete;

  ~CudaBatchedSequenceCache() override {
    if (staged_ != nullptr) {
      (void)cudaEventSynchronize(staged_);
      (void)cudaEventDestroy(staged_);
    }
    if (staging_ != nullptr) {
      (void)cudaFreeHost(staging_);
    }
  }

  // -- CacheControl / BatchControl.

  // Every sequence's capacity, summed: the most the cache holds at once.
  int capacity() const override {
    return cfg_.capacity * max_seqs_;
  }

  // Drops every sequence and keeps the storage, so a reset neither regrows
  // nor invalidates captured graphs.
  void clear() override {
    std::lock_guard<std::mutex> guard(mutex_);
    if (dflash_block_size_ && pool_.allocated()) {
      // A failed forward can be abandoned by a different host thread. Fence
      // the originating stream before reusing its sequence columns.
      if (pool_.synchronize(/*all_streams=*/true) != Error::Ok) {
        // Recycling columns a forward may still write is unsafe; a later
        // clear() retries the fence.
        poisoned_ = true;
        return;
      }
    }
    seqs_.clear();
    step_seq_ids_.clear();
    pending_owners_.clear();
    poisoned_ = false;
  }

  bool declare_step(const std::vector<int32_t>& seq_ids) override {
    std::lock_guard<std::mutex> guard(mutex_);
    if (seq_ids.empty() || seq_ids.size() > static_cast<size_t>(max_write_)) {
      return false;
    }
    std::map<int32_t, int> counts;
    int runs = 0;
    for (size_t i = 0; i < seq_ids.size(); ++i) {
      if (seqs_.count(seq_ids[i]) == 0) {
        return false;
      }
      ++counts[seq_ids[i]];
      runs += (i == 0 || seq_ids[i] != seq_ids[i - 1]) ? 1 : 0;
    }
    if (runs > max_segments_ ||
        (dflash_block_size_ && runs != static_cast<int>(counts.size()))) {
      return false;
    }
    for (const auto& [seq_id, count] : counts) {
      const cache::SequenceCache& plan = seqs_.at(seq_id).plan;
      if (dflash_block_size_ && count < *dflash_block_size_) {
        return false;
      }
      for (size_t layer = 0; layer < geometry_.layers.size(); ++layer) {
        if (!plan.plan(static_cast<int>(layer), plan.length(), count)) {
          return false;
        }
      }
    }
    step_seq_ids_ = seq_ids;
    return true;
  }

  std::optional<int> max_seqs() const override {
    return max_seqs_;
  }

  std::optional<int32_t> seq_new() override {
    std::lock_guard<std::mutex> guard(mutex_);
    const auto column = free_column();
    if (!column) {
      return std::nullopt;
    }
    seqs_.emplace(
        *column, Sequence{*column, cache::SequenceCache(geometry_, cfg_)});
    return *column;
  }

  // Copies the retained prefix into a free column: a sequence's history is
  // private, so a fork cannot share it.
  std::optional<int32_t> seq_clone(int32_t src, std::optional<int> upto)
      override {
    std::lock_guard<std::mutex> guard(mutex_);
    const auto source = seqs_.find(src);
    const auto column = free_column();
    if (source == seqs_.end() || !column || !pool_.allocated()) {
      return std::nullopt;
    }
    // The copies below run on this thread's stream, which may not be the
    // stream the previous step wrote on. Borrow the pool's device and order
    // behind that step first.
    const auto device = CUDAGuard::create(pool_.device());
    if (!device.ok()) {
      return std::nullopt;
    }
    const int length = source->second.plan.length();
    const int keep = std::min(upto.value_or(length), length);
    cache::SequenceCache plan(source->second.plan);
    if (keep <= 0 || !plan.rewind(keep) ||
        copy_column(source->second.column, *column, keep) != Error::Ok ||
        pool_.mark_step_done() != Error::Ok) {
      return std::nullopt;
    }
    seqs_.emplace(*column, Sequence{*column, std::move(plan)});
    return *column;
  }

  bool seq_rm(int32_t seq_id) override {
    std::lock_guard<std::mutex> guard(mutex_);
    return seqs_.erase(seq_id) > 0;
  }

  bool rewind(int32_t seq_id, int position) override {
    std::lock_guard<std::mutex> guard(mutex_);
    const auto it = seqs_.find(seq_id);
    return it != seqs_.end() && it->second.plan.rewind(position);
  }

  int pos(int32_t seq_id) const override {
    std::lock_guard<std::mutex> guard(mutex_);
    const auto it = seqs_.find(seq_id);
    return it == seqs_.end() ? 0 : it->second.plan.length();
  }

  // -- CudaKVCache.

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
      return Error::Ok;
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
    ET_CHECK_OK_OR_RETURN_ERROR(error_);
    ET_CHECK_OR_RETURN_ERROR(
        !poisoned_ && (!dflash_block_size_ || pending_owners_.empty()),
        InvalidState,
        "offgraph_kv: uncommitted or failed draft co-batch; clear required");
    if (!validated_) {
      const Error valid = pool_.validate();
      if (valid != Error::Ok) {
        error_ = valid;
        return valid;
      }
      validated_ = true;
    }
    const int tokens = static_cast<int>(step_seq_ids_.size());
    ET_CHECK_OR_RETURN_ERROR(
        write_length > 0 && write_length <= max_write_ && tokens > 0 &&
            tokens <= write_length,
        InvalidArgument,
        "offgraph_kv: a %lld-token program cannot run %d declared tokens",
        static_cast<long long>(write_length),
        tokens);

    // One run per segment, each continuing its sequence.
    struct Run {
      int32_t seq_id;
      int start;
      int length;
    };
    std::vector<Run> runs;
    std::map<int32_t, int> next;
    for (int i = 0; i < tokens; ++i) {
      const int32_t seq_id = step_seq_ids_[i];
      if (runs.empty() || runs.back().seq_id != seq_id) {
        runs.push_back({seq_id, i, 0});
      }
      ++runs.back().length;
      next.emplace(seq_id, seqs_.at(seq_id).plan.length());
    }
    // Rows every column must hold once this step lands, and those it holds now.
    int64_t live = 0;
    int64_t required = 0;
    for (const auto& [seq_id, sequence] : seqs_) {
      live = std::max<int64_t>(live, sequence.plan.length());
    }
    std::map<int32_t, int> ends = next;
    for (const Run& run : runs) {
      ends[run.seq_id] += run.length;
    }
    for (const auto& [seq_id, end] : ends) {
      required = std::max<int64_t>(required, end);
    }
    ET_CHECK_OK_OR_RETURN_ERROR(pool_.prepare(
        std::min<int64_t>(required, cfg_.capacity) * columns_,
        live * columns_,
        stream));

    ET_CHECK_OK_OR_RETURN_ERROR(reserve_staging(
        static_cast<size_t>(max_write_) * sizeof(int64_t) +
        static_cast<size_t>(max_segments_) * 4 * sizeof(int32_t)));
    auto* cols = reinterpret_cast<int64_t*>(staging_);
    auto* info = reinterpret_cast<int32_t*>(cols + max_write_);
    std::fill(cols, cols + write_length, static_cast<int64_t>(max_seqs_));
    std::fill(info, info + max_segments_ * 4, 0);
    int q_start = 0;
    for (size_t index = 0; index < runs.size(); ++index) {
      const Run& run = runs[index];
      const int column = seqs_.at(run.seq_id).column;
      std::fill(cols + run.start, cols + run.start + run.length, column);
      int32_t* row = info + index * 4;
      const int end = next[run.seq_id] + run.length;
      next[run.seq_id] = end;
      // A draft queries only each owner's block, packed owner by owner.
      const int queries = dflash_block_size_ ? *dflash_block_size_ : run.length;
      row[0] = dflash_block_size_ ? q_start : run.start;
      row[1] = queries;
      row[2] = column;
      row[3] = end;
      q_start += queries;
    }
    const Error enqueued = upload(static_cast<size_t>(write_length), stream);
    const Error fenced = fence_staging(stream);
    ET_CHECK_OK_OR_RETURN_ERROR(enqueued);
    ET_CHECK_OK_OR_RETURN_ERROR(fenced);

    // The rows are claimed now, as the cell layout claims its cells; the
    // forward only fills them.
    for (const Run& run : runs) {
      cache::SequenceCache& plan = seqs_.at(run.seq_id).plan;
      const auto step = plan.plan(0, plan.length(), run.length);
      ET_CHECK_OR_RETURN_ERROR(
          step.has_value(), InvalidArgument, "offgraph_kv: uncommittable step");
      plan.commit(*step);
      if (dflash_block_size_) {
        pending_owners_.push_back(run.seq_id);
      }
    }
    step_seq_ids_.clear();
    return Error::Ok;
  }

  Error commit_step(int64_t write_length) override {
    std::lock_guard<std::mutex> guard(mutex_);
    ET_CHECK_OR_RETURN_ERROR(
        write_length > 0, InvalidArgument, "write length must be positive");
    ET_CHECK_OK_OR_RETURN_ERROR(error_);
    ET_CHECK_OR_RETURN_ERROR(
        !poisoned_,
        InvalidState,
        "offgraph_kv: uncommitted or failed draft co-batch; clear required");
    ET_CHECK_OK_OR_RETURN_ERROR(pool_.mark_step_done());
    // A draft's block is scratch: the next draft rewrites it.
    for (const int32_t owner : pending_owners_) {
      const auto it = seqs_.find(owner);
      if (it != seqs_.end() &&
          !it->second.plan.rewind(
              it->second.plan.length() - *dflash_block_size_)) {
        pending_owners_.clear();
        return Error::InvalidState;
      }
    }
    pending_owners_.clear();
    return Error::Ok;
  }

  // logical_length is the longest sequence: every column holds that many rows.
  OffGraphKVMetrics metrics() const override {
    std::lock_guard<std::mutex> guard(mutex_);
    OffGraphKVMetrics metrics;
    for (const auto& [seq_id, sequence] : seqs_) {
      metrics.logical_length =
          std::max<int64_t>(metrics.logical_length, sequence.plan.length());
    }
    metrics.flat_capacity =
        std::min<int64_t>(pool_.rows() / columns_, cfg_.capacity);
    metrics.growth_count = pool_.growth_count();
    metrics.allocated_bytes = pool_.allocated_bytes();
    return metrics;
  }

 protected:
  void* face(cache::FaceId id) override {
    return cache::expose<cache::BatchControl, CudaKVCache>(this, id);
  }

 private:
  struct Sequence {
    int column;
    cache::SequenceCache plan;
  };

  // A ring never needs more rows than a sequence can hold; it then never wraps.
  static int64_t rows_of(
      const cache::LayerGeometry& layer,
      const cache::CacheConfig& cfg) {
    if (!is_ring(layer)) {
      return cfg.capacity;
    }
    return std::min<int64_t>(
        cfg.capacity,
        static_cast<int64_t>(layer.policy.window) + *cfg.max_write - 1);
  }

  static std::vector<CudaKVPool::Layer> pool_layers(
      const cache::CacheGeometry& geometry,
      const cache::CacheConfig& cfg) {
    std::vector<CudaKVPool::Layer> layers;
    for (const cache::LayerGeometry& layer : geometry.layers) {
      layers.push_back(CudaKVPool::Layer{
          layer.n_kv_heads,
          layer.head_dim,
          rows_of(layer, cfg) * (*cfg.max_seqs + 1),
          // A ring cannot wrap before it holds all its rows, so until then it
          // grows with the flat layers.
          /*growable=*/true});
    }
    return layers;
  }

  std::optional<int> free_column() const {
    for (int column = 0; column < max_seqs_; ++column) {
      if (seqs_.count(column) == 0) {
        return column;
      }
    }
    return std::nullopt;
  }

  // Copies rows [0, keep) of every flat layer, and a ring layer's whole ring,
  // from one column to another, ordered behind the previous step: the copies
  // run on this thread's stream, which may not be the stream that step wrote
  // on.
  Error copy_column(int from, int to, int keep) {
    ET_CHECK_OK_OR_RETURN_ERROR(pool_.follow_previous_step(pool_.stream()));
    for (size_t index = 0; index < geometry_.layers.size(); ++index) {
      const cache::LayerGeometry& layer = geometry_.layers[index];
      const size_t slot = static_cast<size_t>(layer.n_kv_heads) *
          layer.head_dim * slimc10::elementSize(storage_dtype());
      const size_t pitch = slot * columns_;
      const int64_t held =
          std::min(layer_rows_[index], pool_.rows() / columns_);
      const int64_t rows =
          is_ring(layer) ? held : std::min<int64_t>(keep, held);
      for (const bool value : {false, true}) {
        auto* base = static_cast<uint8_t*>(pool_.layer_data(index, value));
        const cudaError_t error = cudaMemcpy2DAsync(
            base + slot * to,
            pitch,
            base + slot * from,
            pitch,
            slot,
            static_cast<size_t>(rows),
            cudaMemcpyDeviceToDevice,
            pool_.stream());
        if (error != cudaSuccess) {
          // Earlier layers may already have queued copies into the free
          // column. Fence them before a failed clone lets that column be
          // reused.
          (void)pool_.synchronize();
          ET_LOG(
              Error,
              "offgraph_kv: cannot copy a sequence's history: %s",
              cudaGetErrorString(error));
          return Error::Internal;
        }
      }
    }
    return Error::Ok;
  }

  slimc10::ScalarType storage_dtype() const {
    slimc10::ScalarType dtype = slimc10::ScalarType::BFloat16;
    (void)storage_dtype_of(cfg_.kv_dtype, dtype);
    return dtype;
  }

  Error upload(size_t width, cudaStream_t stream) {
    const size_t cols_bytes = static_cast<size_t>(max_write_) * sizeof(int64_t);
    for (const auto& [index, offset, bytes] :
         {std::tuple{kSeqColsBuffer, size_t{0}, width * sizeof(int64_t)},
          std::tuple{
              kSeqInfoBuffer,
              cols_bytes,
              static_cast<size_t>(max_segments_) * 4 * sizeof(int32_t)}}) {
      const cudaError_t error = cudaMemcpyAsync(
          pool_.side_buffer(index),
          staging_ + offset,
          bytes,
          cudaMemcpyHostToDevice,
          stream);
      ET_CHECK_OR_RETURN_ERROR(
          error == cudaSuccess,
          Internal,
          "offgraph_kv: cannot write the step's sequences: %s",
          cudaGetErrorString(error));
    }
    return Error::Ok;
  }

  // Waits for the previous step's copies to finish reading the staging, then
  // makes it hold at least `bytes`.
  Error reserve_staging(size_t bytes) {
    if (staged_ != nullptr) {
      const cudaError_t error = cudaEventSynchronize(staged_);
      ET_CHECK_OR_RETURN_ERROR(
          error == cudaSuccess,
          Internal,
          "offgraph_kv: cannot wait for the previous step's uploads: %s",
          cudaGetErrorString(error));
    }
    if (bytes <= staging_bytes_) {
      return Error::Ok;
    }
    if (staging_ != nullptr) {
      (void)cudaFreeHost(staging_);
      staging_ = nullptr;
      staging_bytes_ = 0;
    }
    void* data = nullptr;
    const cudaError_t error = cudaMallocHost(&data, bytes);
    ET_CHECK_OR_RETURN_ERROR(
        error == cudaSuccess,
        MemoryAllocationFailed,
        "offgraph_kv: cannot allocate %zu bytes of pinned staging: %s",
        bytes,
        cudaGetErrorString(error));
    staging_ = static_cast<uint8_t*>(data);
    staging_bytes_ = bytes;
    return Error::Ok;
  }

  Error fence_staging(cudaStream_t stream) {
    cudaError_t error = cudaSuccess;
    if (staged_ == nullptr) {
      error = cudaEventCreateWithFlags(&staged_, cudaEventDisableTiming);
    }
    if (error == cudaSuccess) {
      error = cudaEventRecord(staged_, stream);
    }
    if (error != cudaSuccess) {
      (void)cudaStreamSynchronize(stream);
      ET_LOG(
          Error,
          "offgraph_kv: cannot fence the step uploads: %s",
          cudaGetErrorString(error));
      return Error::Internal;
    }
    return Error::Ok;
  }

  // Guards everything below. The runner drives the verbs and the delegate the
  // steps, from whichever threads they run on.
  mutable std::mutex mutex_;

  const cache::CacheGeometry geometry_;
  const cache::CacheConfig cfg_;
  const int max_seqs_;
  const int columns_;
  const int max_write_;
  const int max_segments_;
  const std::optional<int> dflash_block_size_;
  std::vector<int64_t> layer_rows_;
  CudaKVPool pool_;
  std::map<int32_t, Sequence> seqs_;
  std::vector<int32_t> step_seq_ids_;
  std::vector<int32_t> pending_owners_;
  uint8_t* staging_{nullptr};
  size_t staging_bytes_{0};
  cudaEvent_t staged_{nullptr};
  bool validated_{false};
  // Set when a clear() fence fails; prepare/commit refuse until a later
  // clear() fences cleanly.
  bool poisoned_{false};
  Error error_{Error::Ok};
};

} // namespace

std::shared_ptr<cache::Cache> make_cuda_sequence_kv_cache(
    const cache::CacheGeometry& geometry,
    const cache::CacheConfig& cfg) {
  if (geometry.dflash_block_size.has_value()) {
    ET_LOG(Error, "offgraph_kv: DFlash is unsupported on CUDA");
    return nullptr;
  }
  slimc10::ScalarType storage_dtype = slimc10::ScalarType::BFloat16;
  // Checked before constructing: SequenceCache asserts on an invalid geometry,
  // and returning null lets CacheFactory report the failure instead.
  if (!cache::valid(geometry, cfg) || geometry.dflash_block_size ||
      cfg.initial_capacity <= 0 || cfg.initial_capacity > cfg.capacity ||
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
  // One pool row is the padding scratch; the table needs at least one cell.
  if (cfg.capacity < 2 ||
      (geometry.dflash_block_size &&
       (*geometry.dflash_block_size >= cfg.capacity ||
        *geometry.dflash_block_size > *cfg.max_write))) {
    ET_LOG(Error, "offgraph_kv: the cell layout needs at least two cells");
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

std::shared_ptr<cache::Cache> make_cuda_batched_sequence_kv_cache(
    const cache::CacheGeometry& geometry,
    const cache::CacheConfig& cfg) {
  slimc10::ScalarType storage_dtype = slimc10::ScalarType::BFloat16;
  if (!cache::valid(geometry, cfg) || cfg.initial_capacity <= 0 ||
      cfg.initial_capacity > cfg.capacity ||
      !storage_dtype_of(cfg.kv_dtype, storage_dtype)) {
    ET_LOG(Error, "offgraph_kv: invalid cache geometry or config");
    return nullptr;
  }
  // The storage is one column per sequence and the step buffers are declared
  // [max_write] and [segments, 4]; all are fixed by the program.
  if (!cfg.max_seqs || !cfg.max_write || *cfg.max_write > cfg.capacity ||
      (geometry.dflash_block_size &&
       (*geometry.dflash_block_size <= 0 ||
        *geometry.dflash_block_size > *cfg.max_write))) {
    ET_LOG(
        Error,
        "offgraph_kv: the batched-sequence layout requires max_seqs and "
        "max_write");
    return nullptr;
  }
  return std::make_shared<CudaBatchedSequenceCache>(
      geometry, cfg, storage_dtype);
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
      {cache::kind::kBatchedSequence, make_cuda_batched_sequence_kv_cache},
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
