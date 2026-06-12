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
// sampling, weight-sharing/cuda-graph backend options, device sync) is isolated
// behind EXECUTORCH_BUILD_CUDA inside the .cpp; those isolated points are where
// an MLX runtime would slot in. MLX is NOT implemented or validated here.
//
// V1: serving_capacity() reports a single physical session (one Module = one
// weight allocation). Multiple weight-sharing sessions are a measured V2 step.

#pragma once

#include <atomic>
#include <memory>
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
  bool cuda_graph = false; // enable CUDA graph capture for the decode method
};

/// Engine over one loaded Qwen3.5 MoE Program. Owns immutable resources
/// (tokenizer, metadata, eos ids, config) and creates sessions that each own a
/// physical Module with its own KV/recurrent/conv state.
class ET_EXPERIMENTAL Qwen35MoEEngine : public LLMEngine {
 public:
  static ::executorch::runtime::Result<std::unique_ptr<Qwen35MoEEngine>> create(
      const Qwen35MoEConfig& config);

  ::executorch::runtime::Result<std::unique_ptr<LLMSession>> create_session()
      override;

  // V1: one physical session; weight sharing across sessions is unproven, so we
  // fail closed to 1 (the server queues concurrent requests on the resident
  // session rather than duplicating ~18GB of weights).
  LLMServingCapacity serving_capacity() const override {
    return LLMServingCapacity{};
  }

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
      std::unordered_set<uint64_t> eos_ids)
      : config_(std::move(config)),
        tokenizer_(std::move(tokenizer)),
        metadata_(std::move(metadata)),
        eos_ids_(std::move(eos_ids)) {}

  Qwen35MoEConfig config_;
  std::unique_ptr<::tokenizers::Tokenizer> tokenizer_;
  std::unordered_map<std::string, int64_t> metadata_;
  std::unordered_set<uint64_t> eos_ids_;
};

} // namespace executorch::extension::llm
