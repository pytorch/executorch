// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml.h>

#include <executorch/extension/llm/cache/cell_cache.h>
#include <executorch/extension/llm/cache/sequence_cache.h>

namespace ptn {

// K/V storage per cache layer in the buffer type of the first backend that
// binds it. Each layer's K and V are ggml [head_dim, rows, n_kv_heads] (torch
// [1, n_kv_heads, rows, head_dim]). A growing layer starts at `initial_rows`
// and doubles toward `capacity` as steps need more; a ring layer holds its
// fixed rows from the start. Not thread-safe.
class GgmlKvPool {
 public:
  struct Layer {
    ggml_tensor* k = nullptr;
    ggml_tensor* v = nullptr;
    // The layer's sliding window, 0 for none; layers sharing one share a mask.
    int window = 0;
    // A ring layer attends all its fixed rows; a growing one, a prefix.
    bool ring = false;
  };

  // One allocation of K/V storage. A regrow moves the rows into a new one;
  // the old one lives on while any executable still holds it.
  struct Storage {
    ggml_context_ptr context;
    ggml_backend_buffer_ptr buffer;
    std::vector<Layer> layers;
    // Rows of every growing layer.
    int rows = 0;
  };

  // `ring_rows` holds each layer's fixed rows, or 0 for a growing layer.
  // Throws for a kv_dtype other than Float or Half.
  GgmlKvPool(
      const executorch::extension::llm::cache::CacheGeometry& geometry,
      int kv_dtype,
      int initial_rows,
      int capacity,
      std::vector<int> ring_rows);

  // Ensures growing layers of at least `rows` rows in `backend`'s default
  // buffer type, moving the rows it holds into a larger allocation if needed.
  // Throws if `rows` exceeds capacity or an earlier call used another buffer
  // type.
  void reserve(ggml_backend_t backend, int rows);

  // The current storage; null before the first reserve. Graphs built over a
  // storage stay valid while their executable holds it.
  const std::shared_ptr<const Storage>& storage() const {
    return storage_;
  }

 private:
  executorch::extension::llm::cache::CacheGeometry geometry_;
  std::vector<int> ring_rows_;
  ggml_type type_;
  int initial_rows_;
  int capacity_;
  ggml_backend_buffer_type_t buffer_type_ = nullptr;
  std::shared_ptr<const Storage> storage_;
};

// The off-graph KV cache of one sequence for the ggml engine: the neutral
// SequenceCache controller over a GgmlKvPool. A flat layer's row r holds
// logical position r; a ring layer holds its window + max_write - 1 rows, and
// the SequencePlanner says which rows a step writes and reads. Like
// SequenceCache, it is not thread-safe: executables bound to one cache must be
// serialized.
class GgmlSequenceCache final
    : public executorch::extension::llm::cache::SequenceCache {
 public:
  static constexpr const char* kFaceName = "ptn.ggml.SequenceCache";

  // Throws on geometry or config the ggml engine cannot serve.
  GgmlSequenceCache(
      const executorch::extension::llm::cache::CacheGeometry& geometry,
      const executorch::extension::llm::cache::CacheConfig& cfg);

  GgmlSequenceCache(const GgmlSequenceCache&) = delete;
  GgmlSequenceCache& operator=(const GgmlSequenceCache&) = delete;

  GgmlKvPool& pool() {
    return pool_;
  }

 protected:
  void* face(executorch::extension::llm::cache::FaceId id) override;

 private:
  GgmlKvPool pool_;
};

// The off-graph KV cache of many sequences for the ggml engine: the neutral
// CellCache over a GgmlKvPool whose rows are its cells, every layer growing
// toward the pool's capacity. A step's cells and masks come from place_step; a
// windowed layer keeps every cell and its mask narrows to the window. Like
// CellCache, it is not thread-safe: executables bound to one cache must be
// serialized.
class GgmlCellCache final
    : public executorch::extension::llm::cache::CellCache {
 public:
  static constexpr const char* kFaceName = "ptn.ggml.CellCache";

  // Throws on geometry or config the ggml engine cannot serve.
  GgmlCellCache(
      const executorch::extension::llm::cache::CacheGeometry& geometry,
      const executorch::extension::llm::cache::CacheConfig& cfg);

  GgmlCellCache(const GgmlCellCache&) = delete;
  GgmlCellCache& operator=(const GgmlCellCache&) = delete;

  GgmlKvPool& pool() {
    return pool_;
  }

 protected:
  void* face(executorch::extension::llm::cache::FaceId id) override;

 private:
  GgmlKvPool pool_;
};

// The off-graph cache an executable binds: its storage and its layout, one of
// a single sequence or cells.
struct GgmlKvCache {
  GgmlSequenceCache* sequence = nullptr;
  GgmlCellCache* cells = nullptr;

  explicit operator bool() const {
    return sequence != nullptr || cells != nullptr;
  }
  GgmlKvPool& pool() const {
    return sequence != nullptr ? sequence->pool() : cells->pool();
  }
  int capacity() const {
    return sequence != nullptr ? sequence->capacity() : cells->capacity();
  }
};

} // namespace ptn
