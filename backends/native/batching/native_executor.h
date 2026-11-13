// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

// A batching Executor for a native-backend .ptn: the native counterpart of
// extension/llm/batching's ModuleExecutor. The package must be exported with
// an off-graph KV cache and selected logits (export_hf_llm.py --offgraph_cache
// --logits_to_keep selected), so its method takes tokens [1, T], positions [T]
// and the token indices [K] whose logits [1, K, vocab] it returns. A session is
// one sequence of the engine's batched KV cache, a batch is one or more
// forwards carrying every input's tokens on a single axis, and a clone shares
// its source's cells.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include <executorch/backends/native/runtime/engine/Engine.h>
#include <executorch/backends/native/runtime/graph/ScalarType.h>
#include <executorch/extension/llm/batching/executor.h>
#include <executorch/runtime/core/result.h>

namespace ptn {

struct NativeExecutorConfig {
  std::string model_path;
  // Creates the engine host that runs the method.
  std::function<std::unique_ptr<EngineHost>()> make_host;
  // CacheFactory backend id of that engine's caches, whose kind::kBatched
  // builder must be registered, e.g. kGgmlCacheBackendId.
  std::string cache_backend_id;
  int max_sessions = 1;
  // Zero uses the package's context length.
  int max_session_tokens = 0;
  ScalarType kv_dtype = ScalarType::Float;
  std::string method = "forward";
};

class NativeExecutor final
    : public executorch::extension::llm::batching::Executor {
 public:
  // Loads the package and builds a cache holding every session's context;
  // its storage grows with the cells in use. Returns an error for a package
  // without an off-graph cache or selected logits, unusable limits, or a
  // cache backend without a batched builder.
  static executorch::runtime::Result<std::unique_ptr<NativeExecutor>> create(
      NativeExecutorConfig config);
  ~NativeExecutor() override;

  int32_t vocab_size() const;
  int max_session_tokens() const;

  // Compiles the method on the engine thread.
  bool initialize() override;
  // The widest forward the method takes; a wider batch runs as several.
  size_t preferred_batch_tokens() const override;
  std::optional<executorch::extension::llm::batching::SessionId> open_session()
      override;
  void close_session(
      executorch::extension::llm::batching::SessionId session) override;
  std::optional<executorch::extension::llm::batching::SessionId> clone(
      executorch::extension::llm::batching::SessionId source,
      executorch::extension::llm::batching::Position upto) override;
  void set_sampling(
      executorch::extension::llm::batching::SessionId session,
      const executorch::extension::llm::batching::SamplingParams& params,
      std::optional<uint64_t> seed) override;
  bool execute(
      const executorch::extension::llm::batching::BatchInput& batch,
      executorch::extension::llm::batching::BatchOutput& out) override;

 private:
  struct State;
  explicit NativeExecutor(std::unique_ptr<State> state);

  std::unique_ptr<State> state_;
};

} // namespace ptn
