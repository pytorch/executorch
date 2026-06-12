/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Engine/Session adapter for the Qwen3.5 MoE model, implementing the
// model-agnostic LLMEngine/LLMSession serving contract (llm_session.h) over the
// model's exported prefill/decode methods.
//
// The public surface is backend-agnostic: the server receives an LLMEngine and
// never branches on CUDA vs MLX. Backend-specific execution (CUDA in-graph
// sampling, the weight-sharing backend option, per-session mutable rebinding,
// device sync) is isolated behind EXECUTORCH_BUILD_CUDA inside the .cpp; those
// isolated points are where an MLX runtime would slot in. MLX is NOT
// implemented or validated here.
//
// V2 (CUDA): the ENGINE is multi-session — one shared Module (weights loaded
// once); create_session() hands out multiple logical sessions, each rebinding
// its own GPU buffers for the model's mutable state (KV/conv/recurrent) before
// execute, serialized by the engine lock. serving_capacity() reports how many
// such sessions fit without duplicating weights, or 1 if the backend cannot
// rebind. The per-session rebind machinery is CUDA-backend-private (see
// backends/cuda/runtime/cuda_mutable_state).
//
// The SERVING path (qwen3_5_moe_worker + control plane) exposes this over the
// worker protocol: the worker routes requests to per-session_id state (V2a) and
// reuses each session's resident context across requests (warm append-only
// resume, V2b.1). Execution stays serialized (one in-flight request).

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <executorch/extension/llm/runner/llm_session.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/result.h>
#include <pytorch/tokenizers/tokenizer.h>

namespace executorch::extension::llm {

/// Immutable configuration for a Qwen3.5 MoE engine.
struct Qwen35MoEConfig {
  std::string model_path; // .pte
  std::string data_path; // .ptd (CUDA delegate blob); empty if none
  std::string tokenizer_path; // HuggingFace tokenizer.json
  // V2 multi-session: max physical sessions to advertise when the backend can
  // host them without weight duplication (CUDA per-session mutable rebinding).
  // Clamped to 1 if the backend cannot rebind.
  int32_t max_sessions = 1;
};

/// Engine over one loaded Qwen3.5 MoE Program. Owns immutable resources
/// (tokenizer, metadata, eos ids, config) plus one shared Module (weights
/// loaded once); creates sessions that share that Module but each own their
/// per-session mutable state (KV/recurrent/conv), rebound before execute under
/// the engine lock.
class ET_EXPERIMENTAL Qwen35MoEEngine : public LLMEngine {
 public:
  static ::executorch::runtime::Result<std::unique_ptr<Qwen35MoEEngine>> create(
      const Qwen35MoEConfig& config);

  ~Qwen35MoEEngine() override;

  ::executorch::runtime::Result<std::unique_ptr<LLMSession>> create_session()
      override;

  // CUDA V2: one shared Module (one weight allocation); each session rebinds
  // its own GPU buffers for the model's mutable state. Reports
  // config.max_sessions when the backend supports per-session rebinding, else
  // fails closed to 1.
  LLMServingCapacity serving_capacity() const override;

  const std::unordered_map<std::string, int64_t>& metadata() const override {
    return metadata_;
  }

  // Non-owning; valid for the engine's lifetime (the engine must outlive any
  // session and any caller using this). Used by the runner to encode prompts;
  // not part of the model-agnostic LLMEngine surface the server depends on.
  ::tokenizers::Tokenizer* tokenizer() const {
    return tokenizer_.get();
  }

  Qwen35MoEEngine(const Qwen35MoEEngine&) = delete;
  Qwen35MoEEngine& operator=(const Qwen35MoEEngine&) = delete;

 private:
  Qwen35MoEEngine(
      Qwen35MoEConfig config,
      std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
      std::unordered_map<std::string, int64_t> metadata,
      std::unordered_set<uint64_t> eos_ids,
      std::unique_ptr<Module> shared_module,
      bool rebind_available,
      int mutable_ctx)
      : config_(std::move(config)),
        tokenizer_(std::move(tokenizer)),
        metadata_(std::move(metadata)),
        eos_ids_(std::move(eos_ids)),
        shared_module_(std::move(shared_module)),
        rebind_available_(rebind_available),
        mutable_ctx_(mutable_ctx) {}

  Qwen35MoEConfig config_;
  std::unique_ptr<::tokenizers::Tokenizer> tokenizer_;
  std::unordered_map<std::string, int64_t> metadata_;
  std::unordered_set<uint64_t> eos_ids_;

  // One physical model shared by all sessions (one weight allocation). Sessions
  // hold a non-owning pointer to it and execute under exec_mutex_.
  std::unique_ptr<Module> shared_module_;
  std::mutex exec_mutex_;
  // Whether the loaded CUDA delegate supports per-session mutable rebinding.
  bool rebind_available_ = false;
  // CUDA mutable-state context for this engine's model (per-engine, not
  // global); destroyed in the destructor. kInvalidMutableContext (0) when
  // unused.
  int mutable_ctx_ = 0;
  // Live sessions, enforced against serving_capacity() so the engine never
  // hands out more sessions than it can host without sharing state /
  // duplicating weights. Decremented when a session is destroyed.
  std::atomic<int> live_sessions_{0};
};

} // namespace executorch::extension::llm
