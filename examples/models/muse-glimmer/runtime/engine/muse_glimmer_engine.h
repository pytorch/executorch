/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Engine/Session adapter for the shared embeddings-input Muse Glimmer contract.

#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <executorch/examples/models/muse-glimmer/runtime/engine/dflash_timing.h>
#include <executorch/examples/models/muse-glimmer/runtime/engine/muse_glimmer_vision_runtime.h>
#include <executorch/extension/llm/runner/llm_session.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/core/result.h>
#include <pytorch/tokenizers/tokenizer.h>

#ifdef EXECUTORCH_BUILD_CUDA
#include <executorch/backends/cuda/runtime/cuda_mutable_state.h>
#elif defined(EXECUTORCH_BUILD_MLX)
#include <executorch/backends/mlx/runtime/backend_options.h>
#include <executorch/backends/mlx/runtime/mlx_mutable_state.h>
#else
#error \
    "MuseGlimmerEngine requires EXECUTORCH_BUILD_CUDA or EXECUTORCH_BUILD_MLX"
#endif

namespace executorch::extension::llm {

// <|patch|>. One row of image soft tokens is spliced over each occurrence.
// MuseGlimmerEngine::create checks this against the loaded tokenizer, so a
// model whose vocabulary disagrees fails at load rather than generating
// nonsense.
inline constexpr uint64_t kMuseGlimmerImagePatchTokenId = 200092;
inline constexpr char kMuseGlimmerImagePlaceholder[] = "<img>";

::tokenizers::Result<std::vector<uint64_t>> tokenize_muse_glimmer_prompt(
    const ::tokenizers::Tokenizer& tokenizer,
    const std::string& prompt,
    uint64_t bos_id,
    std::optional<int64_t> image_tokens = std::nullopt);

enum class MuseGlimmerArtifactMode {
  Auto,
  Autoregressive,
  DFlash,
};

#if defined(EXECUTORCH_BUILD_CUDA)
using MuseGlimmerMutableStateContextOwner =
    ::executorch::backends::cuda::MutableStateContextOwner;
constexpr int kMuseGlimmerNoMutableSession =
    ::executorch::backends::cuda::kNoMutableSession;
#elif defined(EXECUTORCH_BUILD_MLX)
using MuseGlimmerMutableStateContextOwner =
    ::executorch::backends::mlx::MutableStateContextOwner;
constexpr int kMuseGlimmerNoMutableSession =
    ::executorch::backends::mlx::kNoMutableSession;
#endif

struct MuseGlimmerConfig {
  std::string model_path;
  std::string data_path;
  std::string tokenizer_path;
  std::string pos_embed_path;
  size_t max_image_bytes = 20 * 1024 * 1024;
  int32_t max_sessions = 1;
  int64_t eos_id = 200001;
  bool enable_cuda_graph = false;
  MuseGlimmerArtifactMode artifact_mode = MuseGlimmerArtifactMode::Auto;
  int32_t dflash_block_length = 0;
  int32_t dflash_n_draft = 0;
  bool dflash_draft_argmax = true;
  bool dflash_ignore_eos = false;
  DFlashDecodeTiming* dflash_timing = nullptr;
};

class MuseGlimmerMultimodalSession {
 public:
  virtual ~MuseGlimmerMultimodalSession() = default;

  virtual void stage_image_for_next_prefill(PreparedMuseGlimmerImage image) = 0;
  virtual void clear_staged_image() = 0;
};

inline MuseGlimmerMultimodalSession* as_muse_glimmer_multimodal_session(
    LLMSession* session) {
  return dynamic_cast<MuseGlimmerMultimodalSession*>(session);
}

class ET_EXPERIMENTAL MuseGlimmerEngine : public LLMEngine {
 public:
  static ::executorch::runtime::Result<std::unique_ptr<MuseGlimmerEngine>>
  create(const MuseGlimmerConfig& config);

  ~MuseGlimmerEngine() override;

  ::executorch::runtime::Result<std::unique_ptr<LLMSession>> create_session()
      override;

  LLMServingCapacity serving_capacity() const override;

  const std::unordered_map<std::string, int64_t>& metadata() const override {
    return metadata_;
  }

  ::tokenizers::Tokenizer* tokenizer() const {
    return tokenizer_.get();
  }

  bool has_vision() const {
    return vision_runtime_ != nullptr;
  }

  ::executorch::runtime::Result<PreparedMuseGlimmerImage>
  prepare_image_from_file(const std::string& image_path) const;

  ::executorch::runtime::Result<PreparedMuseGlimmerImage>
  prepare_image_from_bytes(
      ::executorch::runtime::Span<const uint8_t> encoded_image) const;

  ::executorch::runtime::Result<PreparedMuseGlimmerImage>
  prepare_image_from_bytes(const std::vector<uint8_t>& encoded_image) const {
    return prepare_image_from_bytes(::executorch::runtime::Span<const uint8_t>(
        encoded_image.data(), encoded_image.size()));
  }

  MuseGlimmerArtifactMode artifact_mode() const {
    return artifact_mode_;
  }

  MuseGlimmerEngine(const MuseGlimmerEngine&) = delete;
  MuseGlimmerEngine& operator=(const MuseGlimmerEngine&) = delete;

 private:
  MuseGlimmerEngine(
      MuseGlimmerConfig config,
      std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
      std::unordered_map<std::string, int64_t> metadata,
      std::unordered_set<uint64_t> eos_ids,
      std::unique_ptr<Module> shared_module,
      MuseGlimmerArtifactMode artifact_mode,
      int64_t max_prefill_chunk,
      int64_t min_prefill_chunk,
      int64_t dflash_block_size,
      int64_t dflash_mask_token_id,
      int64_t dflash_n_target_layers,
      int64_t dflash_sliding_window,
      int64_t dflash_min_target_prefill_chunk,
      int64_t dflash_max_target_prefill_chunk,
      std::string activation_dtype,
      ::executorch::extension::TensorPtr decode_pos_table_dev,
      std::unique_ptr<MuseGlimmerVisionRuntime> vision_runtime,
      bool rebind_available,
      std::unique_ptr<MuseGlimmerMutableStateContextOwner> mutable_state)
      : config_(std::move(config)),
        tokenizer_(std::move(tokenizer)),
        metadata_(std::move(metadata)),
        eos_ids_(std::move(eos_ids)),
        shared_module_(std::move(shared_module)),
        artifact_mode_(artifact_mode),
        max_prefill_chunk_(max_prefill_chunk),
        min_prefill_chunk_(min_prefill_chunk),
        dflash_block_size_(dflash_block_size),
        dflash_mask_token_id_(dflash_mask_token_id),
        dflash_n_target_layers_(dflash_n_target_layers),
        dflash_sliding_window_(dflash_sliding_window),
        dflash_min_target_prefill_chunk_(dflash_min_target_prefill_chunk),
        dflash_max_target_prefill_chunk_(dflash_max_target_prefill_chunk),
        activation_dtype_(std::move(activation_dtype)),
        decode_pos_table_dev_(std::move(decode_pos_table_dev)),
        vision_runtime_(std::move(vision_runtime)),
        rebind_available_(rebind_available),
        mutable_state_(std::move(mutable_state)) {}

  MuseGlimmerConfig config_;
  std::unique_ptr<::tokenizers::Tokenizer> tokenizer_;
  std::unordered_map<std::string, int64_t> metadata_;
  std::unordered_set<uint64_t> eos_ids_;
  std::unique_ptr<Module> shared_module_;
  std::mutex exec_mutex_;
  MuseGlimmerArtifactMode artifact_mode_ =
      MuseGlimmerArtifactMode::Autoregressive;
  int64_t max_prefill_chunk_ = 0;
  int64_t min_prefill_chunk_ = 1;
  int64_t dflash_block_size_ = 0;
  int64_t dflash_mask_token_id_ = 0;
  int64_t dflash_n_target_layers_ = 0;
  int64_t dflash_sliding_window_ = 0;
  int64_t dflash_min_target_prefill_chunk_ = 0;
  int64_t dflash_max_target_prefill_chunk_ = 0;
  std::string activation_dtype_;
  ::executorch::extension::TensorPtr decode_pos_table_dev_;
  std::unique_ptr<MuseGlimmerVisionRuntime> vision_runtime_;
  bool rebind_available_ = false;
  std::unique_ptr<MuseGlimmerMutableStateContextOwner> mutable_state_;
  std::atomic<int> live_sessions_{0};
};

} // namespace executorch::extension::llm
