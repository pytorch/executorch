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

#include <executorch/extension/llm/cache/sequence_cache.h>

namespace ptn {

// The off-graph KV cache of one sequence for the ggml engine: the neutral
// SequenceCache controller plus K/V storage per layer in the buffer type of
// the first backend that binds it. Each layer's K and V are ggml
// [head_dim, rows, n_kv_heads] (torch [1, n_kv_heads, rows, head_dim]); row r
// holds logical position r. Storage starts at the config's initial_capacity
// rows and doubles toward capacity as steps need more. Like SequenceCache, it
// is not thread-safe: executables bound to one cache must be serialized.
class GgmlSequenceCache final
    : public executorch::extension::llm::cache::SequenceCache {
 public:
  static constexpr const char* kFaceName = "ptn.ggml.SequenceCache";

  struct Layer {
    ggml_tensor* k = nullptr;
    ggml_tensor* v = nullptr;
  };

  // One allocation of K/V storage. A regrow moves the rows into a new one;
  // the old one lives on while any executable still holds it.
  struct Storage {
    ggml_context_ptr context;
    ggml_backend_buffer_ptr buffer;
    std::vector<Layer> layers;
    int rows = 0;
  };

  // Throws on geometry or config the ggml engine cannot serve.
  GgmlSequenceCache(
      const executorch::extension::llm::cache::CacheGeometry& geometry,
      const executorch::extension::llm::cache::CacheConfig& cfg);

  GgmlSequenceCache(const GgmlSequenceCache&) = delete;
  GgmlSequenceCache& operator=(const GgmlSequenceCache&) = delete;

  // Ensures storage of at least `rows` rows in `backend`'s default buffer
  // type, moving the rows it holds into a larger allocation if needed. Throws
  // if `rows` exceeds capacity or an earlier call used another buffer type.
  void reserve(ggml_backend_t backend, int rows);

  // The current storage; null before the first reserve. Graphs built over a
  // storage stay valid while their executable holds it.
  const std::shared_ptr<const Storage>& storage() const {
    return storage_;
  }

 protected:
  void* face(executorch::extension::llm::cache::FaceId id) override;

 private:
  executorch::extension::llm::cache::CacheGeometry geometry_;
  ggml_type type_;
  int initial_rows_;
  ggml_backend_buffer_type_t buffer_type_ = nullptr;
  std::shared_ptr<const Storage> storage_;
};

} // namespace ptn
