// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/engine/ggml/GgmlCache.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <executorch/backends/native/runtime/engine/ggml/GgmlEngine.h>
#include <executorch/backends/native/runtime/graph/ScalarType.h>
#include <executorch/extension/llm/cache/cache_registry.h>

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
  for (const cache::LayerGeometry& layer : geometry.layers) {
    if (layer.policy.kind != cache::LayerPolicy::Kind::Flat) {
      throw std::invalid_argument(
          "ggml cache: only flat (full-history) layers are supported");
    }
  }
  return geometry;
}

} // namespace

GgmlSequenceCache::GgmlSequenceCache(
    const cache::CacheGeometry& geometry,
    const cache::CacheConfig& cfg)
    : SequenceCache(checked(geometry, cfg), cfg),
      geometry_(geometry),
      type_(kv_type(cfg.kv_dtype)),
      initial_rows_(std::clamp(cfg.initial_capacity, 1, cfg.capacity)) {}

void GgmlSequenceCache::reserve(ggml_backend_t backend, int rows) {
  ggml_backend_buffer_type_t buffer_type =
      ggml_backend_get_default_buffer_type(backend);
  if (storage_ && buffer_type != buffer_type_) {
    throw std::runtime_error(
        std::string("ggml cache: storage lives in ") +
        ggml_backend_buft_name(buffer_type_) + ", not " +
        ggml_backend_buft_name(buffer_type));
  }
  if (rows > capacity()) {
    throw std::out_of_range(
        "ggml cache: " + std::to_string(rows) +
        " rows exceed the capacity of " + std::to_string(capacity()));
  }
  if (storage_ && rows <= storage_->rows) {
    return;
  }
  int64_t target = storage_ ? storage_->rows : initial_rows_;
  while (target < rows) {
    target *= 2;
  }
  auto next = std::make_shared<Storage>();
  next->rows = static_cast<int>(std::min<int64_t>(target, capacity()));

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
    Layer out;
    for (ggml_tensor** t : {&out.k, &out.v}) {
      *t = ggml_new_tensor_3d(
          next->context.get(),
          type_,
          layer.head_dim,
          next->rows,
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

  // Each head's rows are contiguous, but the head stride grows with the rows.
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

void register_ggml_cache_builders() {
  // Registering twice is harmless; the first builder stays.
  (void)cache::CacheFactory::global().register_builder(
      kGgmlCacheBackendId,
      cache::kind::kSingle,
      [](const cache::CacheGeometry& geometry, const cache::CacheConfig& cfg) {
        return std::make_shared<GgmlSequenceCache>(geometry, cfg);
      });
}

} // namespace ptn
