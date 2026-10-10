/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/examples/models/muse-glimmer/runtime/backend_factory.h>
#include <executorch/extension/llm/cache/cache_registry.h>
#include <executorch/extension/module/module.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace executorch::extension::llm {

class MuseGlimmerMLXExecutor final : public batching::Executor {
 public:
  static runtime::Result<MuseGlimmerBackend> create(
      const MuseGlimmerBackendConfig& config);
  ~MuseGlimmerMLXExecutor() override;

  bool initialize() override;
  size_t preferred_batch_tokens() const override;
  std::optional<batching::SessionId> open_session() override;
  void close_session(batching::SessionId session) override;
  std::optional<batching::SessionId> clone(
      batching::SessionId source,
      batching::Position upto) override;
  void set_sampling(
      batching::SessionId session,
      const batching::SamplingParams& params,
      std::optional<uint64_t> seed) override;
  bool accepts(const batching::PreparedInput& input) const override;
  bool execute(const batching::BatchInput& batch, batching::BatchOutput& out)
      override;

 private:
  struct EngineEmbeddings;
  struct SessionState {
    int32_t seq_id;
    std::optional<batching::SamplingParams> sampling;
    uint64_t seed = 0;
  };
  using Rewinds = std::vector<std::pair<int32_t, int>>;

  MuseGlimmerMLXExecutor(
      std::unique_ptr<Module> module,
      std::shared_ptr<cache::Cache> cache,
      std::shared_ptr<const MuseGlimmerPreparationSpec> spec,
      std::string pos_embed_path,
      int max_sessions);
  runtime::Result<Rewinds> validate_batch(
      const batching::BatchInput& batch) const;
  std::optional<batching::SessionId> publish_session(
      int32_t seq_id,
      batching::Position position);
  std::optional<batching::Token> sample_row(
      aten::Tensor& logits,
      int row,
      batching::SessionId session,
      batching::Position prediction_position);

  // Reverse destruction: vision before its module/mutex, then the module before
  // the registry entry whose cache its delegate resolved.
  cache::InstallGuard install_guard_;
  std::unique_ptr<Module> module_;
  cache::BatchControl* const ctl_;
  const std::shared_ptr<const MuseGlimmerPreparationSpec> spec_;
  std::unique_ptr<EngineEmbeddings> embeddings_;
  const int max_sessions_;
  bool initialization_attempted_ = false;
  bool initialized_ = false;
  batching::SessionId next_session_ = 1;
  std::unordered_map<batching::SessionId, SessionState> sessions_;
};

} // namespace executorch::extension::llm
