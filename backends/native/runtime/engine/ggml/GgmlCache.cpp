// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/engine/ggml/GgmlCache.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <executorch/backends/native/runtime/engine/ggml/GgmlEngine.h>
#include <executorch/backends/native/runtime/graph/ScalarType.h>
#include <executorch/extension/llm/cache/cache_registry.h>
#include <executorch/runtime/platform/log.h>

namespace ptn {

namespace cache = executorch::extension::llm::cache;

namespace {

ggml_type kv_type(int kv_dtype) {
  switch (static_cast<ScalarType>(kv_dtype)) {
    case ScalarType::Float:
      return GGML_TYPE_F32;
    case ScalarType::Half:
      return GGML_TYPE_F16;
    default:
      throw std::invalid_argument(
          "ggml cache: kv_dtype must be Float or Half, got " +
          std::to_string(kv_dtype));
  }
}

const cache::CacheGeometry& checked(
    const cache::CacheGeometry& geometry,
    const cache::CacheConfig& cfg) {
  if (!cache::valid(geometry, cfg)) {
    throw std::invalid_argument("ggml cache: invalid geometry or config");
  }
  if (geometry.dflash_block_size) {
    throw std::invalid_argument("ggml cache: DFlash geometry is unsupported");
  }
  return geometry;
}

// SequenceCache's ring: window + max_write - 1 slots, max_write defaulting to
// the window. Positions below capacity never reach a slot past it.
std::vector<int> ring_rows(
    const cache::CacheGeometry& geometry,
    const cache::CacheConfig& cfg) {
  std::vector<int> out;
  out.reserve(geometry.layers.size());
  for (const cache::LayerGeometry& layer : geometry.layers) {
    const int window = layer.policy.window;
    out.push_back(
        layer.policy.kind == cache::LayerPolicy::Kind::Ring
            ? std::min(
                  window + cfg.max_write.value_or(window) - 1, cfg.capacity)
            : 0);
  }
  return out;
}

} // namespace

GgmlKvPool::GgmlKvPool(
    const cache::CacheGeometry& geometry,
    int kv_dtype,
    int initial_rows,
    int capacity,
    std::vector<int> ring_rows)
    : geometry_(geometry),
      ring_rows_(std::move(ring_rows)),
      type_(kv_type(kv_dtype)),
      initial_rows_(std::clamp(initial_rows, 1, capacity)),
      capacity_(capacity) {}

GgmlSequenceCache::GgmlSequenceCache(
    const cache::CacheGeometry& geometry,
    const cache::CacheConfig& cfg)
    : SequenceCache(checked(geometry, cfg), cfg),
      pool_(
          geometry,
          cfg.kv_dtype,
          cfg.initial_capacity,
          cfg.capacity,
          ring_rows(geometry, cfg)) {}

void GgmlKvPool::reserve(ggml_backend_t backend, int rows) {
  ggml_backend_buffer_type_t buffer_type =
      ggml_backend_get_default_buffer_type(backend);
  if (storage_ && buffer_type != buffer_type_) {
    throw std::runtime_error(
        std::string("ggml cache: storage lives in ") +
        ggml_backend_buft_name(buffer_type_) + ", not " +
        ggml_backend_buft_name(buffer_type));
  }
  if (rows > capacity_) {
    throw std::out_of_range(
        "ggml cache: " + std::to_string(rows) +
        " rows exceed the capacity of " + std::to_string(capacity_));
  }
  if (storage_ && rows <= storage_->rows) {
    return;
  }
  int64_t target = storage_ ? storage_->rows : initial_rows_;
  while (target < rows) {
    target *= 2;
  }
  auto next = std::make_shared<Storage>();
  next->rows = static_cast<int>(std::min<int64_t>(target, capacity_));

  ggml_init_params params{
      .mem_size = ggml_tensor_overhead() * (2 * geometry_.layers.size() + 1),
      .mem_buffer = nullptr,
      .no_alloc = true,
  };
  next->context.reset(ggml_init(params));
  if (!next->context) {
    throw std::runtime_error("ggml cache: failed to create tensor context");
  }
  for (size_t i = 0; i < geometry_.layers.size(); ++i) {
    const cache::LayerGeometry& layer = geometry_.layers[i];
    Layer out{.window = layer.policy.window, .ring = ring_rows_[i] > 0};
    for (ggml_tensor** t : {&out.k, &out.v}) {
      *t = ggml_new_tensor_3d(
          next->context.get(),
          type_,
          layer.head_dim,
          ring_rows_[i] > 0 ? ring_rows_[i] : next->rows,
          layer.n_kv_heads);
      ggml_format_name(*t, "kv_cache.%s.%zu", t == &out.k ? "k" : "v", i);
    }
    next->layers.push_back(out);
  }
  next->buffer.reset(
      ggml_backend_alloc_ctx_tensors_from_buft(next->context.get(), buffer_type));
  if (!next->buffer) {
    throw std::runtime_error(
        "ggml cache: failed to allocate K/V storage of " +
        std::to_string(next->rows) + " rows");
  }
  ggml_backend_buffer_clear(next->buffer.get(), 0);

  // Each head's rows are contiguous, but a flat layer's head stride grows with
  // its rows. A ring layer keeps its size, so its rows land where they were.
  std::vector<uint8_t> staging;
  for (size_t i = 0; storage_ && i < storage_->layers.size(); ++i) {
    const Layer& old = storage_->layers[i];
    for (const auto& [from, to] :
         {std::pair{old.k, next->layers[i].k},
          std::pair{old.v, next->layers[i].v}}) {
      staging.resize(ggml_nbytes(from));
      ggml_backend_tensor_get(from, staging.data(), 0, staging.size());
      const size_t head_bytes = static_cast<size_t>(from->ne[1]) * from->nb[1];
      for (int64_t h = 0; h < from->ne[2]; ++h) {
        ggml_backend_tensor_set(
            to,
            staging.data() + h * from->nb[2],
            h * to->nb[2],
            head_bytes);
      }
    }
  }

  buffer_type_ = buffer_type;
  storage_ = std::move(next);
}

void* GgmlSequenceCache::face(cache::FaceId id) {
  return cache::expose<
      cache::SequenceControl,
      cache::SequencePlanner,
      GgmlSequenceCache>(this, id);
}

GgmlCellCache::GgmlCellCache(
    const cache::CacheGeometry& geometry,
    const cache::CacheConfig& cfg)
    : CellCache(checked(geometry, cfg), cfg),
      pool_(
          geometry,
          cfg.kv_dtype,
          cfg.initial_capacity,
          cfg.capacity,
          std::vector<int>(geometry.layers.size(), 0)) {}

void* GgmlCellCache::face(cache::FaceId id) {
  return cache::
      expose<cache::BatchControl, cache::CellStepper, GgmlCellCache>(this, id);
}

namespace {

// CacheFactory::build reports a null cache as an error, so a cache the engine
// cannot serve comes back as one rather than as an exception.
template <typename CacheT>
cache::CacheBuilder builder() {
  return [](const cache::CacheGeometry& geometry,
            const cache::CacheConfig& cfg) -> std::shared_ptr<cache::Cache> {
    try {
      return std::make_shared<CacheT>(geometry, cfg);
    } catch (const std::exception& error) {
      ET_LOG(Error, "%s", error.what());
      return nullptr;
    }
  };
}

} // namespace

void register_ggml_cache_builders() {
  // Registering twice is harmless; the first builder stays.
  (void)cache::CacheFactory::global().register_builder(
      kGgmlCacheBackendId, cache::kind::kSingle, builder<GgmlSequenceCache>());
  for (const char* kind : {cache::kind::kBatched, cache::kind::kBatchedCell}) {
    (void)cache::CacheFactory::global().register_builder(
        kGgmlCacheBackendId, kind, builder<GgmlCellCache>());
  }
}

} // namespace ptn
