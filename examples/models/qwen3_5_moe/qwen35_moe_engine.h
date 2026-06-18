/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Engine/Session adapter for the Qwen3.5 MoE exported prefill/decode methods.
// CUDA builds can host multiple sessions on one loaded model by rebinding the
// model's mutable buffers before each execute.

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

#ifdef EXECUTORCH_BUILD_CUDA
#include <executorch/backends/cuda/runtime/cuda_mutable_state.h>
#elif defined(EXECUTORCH_BUILD_MLX)
#include <executorch/backends/mlx/runtime/mlx_mutable_state.h>
#endif

#if defined(EXECUTORCH_BUILD_CUDA) || defined(EXECUTORCH_BUILD_MLX)
#define QWEN_HAS_MUTABLE_STATE 1
#endif

namespace executorch::extension::llm {

#if defined(EXECUTORCH_BUILD_CUDA)
using MutableStateContextOwner =
    ::executorch::backends::cuda::MutableStateContextOwner;
constexpr int kNoMutableSession =
    ::executorch::backends::cuda::kNoMutableSession;
#elif defined(EXECUTORCH_BUILD_MLX)
using MutableStateContextOwner =
    ::executorch::backends::mlx::MutableStateContextOwner;
constexpr int kNoMutableSession =
    ::executorch::backends::mlx::kNoMutableSession;
#endif

/// Immutable configuration for a Qwen3.5 MoE engine.
struct Qwen35MoEConfig {
  std::string model_path; // .pte
  std::string data_path; // .ptd (CUDA delegate blob); empty if none
  std::string tokenizer_path; // HuggingFace tokenizer.json
  // Clamped to 1 if the backend cannot isolate per-session mutable state.
  int32_t max_sessions = 1;
  // CUDA-only: graph-capture decode for single-session runner use. Incompatible
  // with per-session mutable-state rebinding, so capacity remains 1.
  bool enable_cuda_graph = false;
};

/// Engine over one loaded Qwen3.5 MoE program.
class ET_EXPERIMENTAL Qwen35MoEEngine : public LLMEngine {
 public:
  static ::executorch::runtime::Result<std::unique_ptr<Qwen35MoEEngine>> create(
      const Qwen35MoEConfig& config);

  ~Qwen35MoEEngine() override;

  ::executorch::runtime::Result<std::unique_ptr<LLMSession>> create_session()
      override;

  LLMServingCapacity serving_capacity() const override;

  const std::unordered_map<std::string, int64_t>& metadata() const override {
    return metadata_;
  }

  // Non-owning; valid for the engine's lifetime.
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
      bool rebind_available
#ifdef QWEN_HAS_MUTABLE_STATE
      ,
      std::unique_ptr<MutableStateContextOwner> mutable_state
#endif
      )
      : config_(std::move(config)),
        tokenizer_(std::move(tokenizer)),
        metadata_(std::move(metadata)),
        eos_ids_(std::move(eos_ids)),
        shared_module_(std::move(shared_module)),
        rebind_available_(rebind_available)
#ifdef QWEN_HAS_MUTABLE_STATE
        ,
        mutable_state_(std::move(mutable_state))
#endif
  {
  }

  Qwen35MoEConfig config_;
  std::unique_ptr<::tokenizers::Tokenizer> tokenizer_;
  std::unordered_map<std::string, int64_t> metadata_;
  std::unordered_set<uint64_t> eos_ids_;

  std::unique_ptr<Module> shared_module_;
  std::mutex exec_mutex_;
  bool rebind_available_ = false;
#ifdef QWEN_HAS_MUTABLE_STATE
  std::unique_ptr<MutableStateContextOwner> mutable_state_;
#endif
  std::atomic<int> live_sessions_{0};
};

} // namespace executorch::extension::llm
