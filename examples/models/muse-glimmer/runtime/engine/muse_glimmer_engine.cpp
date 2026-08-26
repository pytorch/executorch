/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/muse-glimmer/runtime/engine/muse_glimmer_engine.h>

#include <executorch/examples/models/muse-glimmer/runtime/engine/dflash_session.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/core/portable_type/device.h>
#include <executorch/runtime/platform/log.h>
#include <pytorch/tokenizers/hf_tokenizer.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <vector>

#ifdef EXECUTORCH_BUILD_CUDA
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#else
#include <executorch/extension/llm/sampler/util.h>
#endif

namespace executorch::extension::llm {

using ::executorch::extension::clone_tensor_ptr_to;
using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::extension::TensorPtr;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;
using ::executorch::runtime::Result;
using SizesType = executorch::aten::SizesType;

::tokenizers::Result<std::vector<uint64_t>> tokenize_muse_glimmer_prompt(
    const ::tokenizers::Tokenizer& tokenizer,
    const std::string& prompt,
    uint64_t bos_id,
    std::optional<int64_t> image_tokens) {
  std::vector<uint64_t> tokens{bos_id};
  if (!image_tokens.has_value()) {
    auto encoded = tokenizer.encode(prompt, /*bos=*/0, /*eos=*/0);
    if (!encoded.ok()) {
      return encoded.error();
    }
    tokens.insert(tokens.end(), encoded->begin(), encoded->end());
    return tokens;
  }

  if (*image_tokens <= 0) {
    return ::tokenizers::Error::EncodeFailure;
  }
  const std::string placeholder = kMuseGlimmerImagePlaceholder;
  const size_t marker = prompt.find(placeholder);
  if (marker == std::string::npos ||
      prompt.find(placeholder, marker + placeholder.size()) !=
          std::string::npos) {
    return ::tokenizers::Error::EncodeFailure;
  }

  auto before =
      tokenizer.encode(prompt.substr(0, marker), /*bos=*/0, /*eos=*/0);
  if (!before.ok()) {
    return before.error();
  }
  auto after = tokenizer.encode(
      prompt.substr(marker + placeholder.size()), /*bos=*/0, /*eos=*/0);
  if (!after.ok()) {
    return after.error();
  }
  tokens.insert(tokens.end(), before->begin(), before->end());
  tokens.insert(
      tokens.end(),
      static_cast<size_t>(*image_tokens),
      kMuseGlimmerImagePatchTokenId);
  tokens.insert(tokens.end(), after->begin(), after->end());
  return tokens;
}

namespace {

constexpr const char* kForwardFromEmbeddingsMethod = "forward_from_embeddings";
constexpr const char* kDecodeFromEmbeddingMethod = "decode_from_embedding";

constexpr const char* kMaxPrefillChunk = "get_max_prefill_chunk";
constexpr const char* kMinPrefillChunk = "get_min_prefill_chunk";
constexpr const char* kUseSampling = "use_sampling";
constexpr const char* kTargetForwardFromEmbeddingsMethod =
    "target_forward_from_embeddings";
constexpr const char* kTargetPrefillFromEmbeddingsMethod =
    "target_prefill_from_embeddings";
constexpr const char* kEmbedTextMethod = "embed_text";
constexpr const char* kMuseGlimmerVisionEncoderMethod = "vision_encoder";
constexpr const char* kDraftForwardMethod = "draft_forward";
constexpr const char* kDraftPrefillMethod = "draft_prefill";
constexpr const char* kDFlashBlockSize = "get_block_size";
constexpr const char* kDFlashMaskTokenId = "get_mask_token_id";
constexpr const char* kDFlashTargetLayers = "get_n_target_layers";
constexpr const char* kDFlashSlidingWindow = "get_draft_sliding_window";
constexpr const char* kDFlashMinTargetPrefillChunk =
    "get_min_target_prefill_chunk";
constexpr const char* kDFlashMaxTargetPrefillChunk =
    "get_max_target_prefill_chunk";
constexpr const char* kDFlashMinDraftPrefillChunk =
    "get_min_draft_prefill_chunk";
constexpr const char* kDFlashMaxDraftPrefillChunk =
    "get_max_draft_prefill_chunk";
constexpr const char* kActivationDtype = "get_activation_dtype";
constexpr const char* kVisionHiddenSize = "get_vision_hidden_size";
constexpr const char* kMaxVisionPatches = "get_max_vision_patches";
constexpr int64_t kVisionDownsampleArea = 4;

const char* artifact_mode_name(MuseGlimmerArtifactMode mode) {
  switch (mode) {
    case MuseGlimmerArtifactMode::Auto:
      return "auto";
    case MuseGlimmerArtifactMode::Autoregressive:
      return "autoregressive";
    case MuseGlimmerArtifactMode::DFlash:
      return "dflash";
  }
  return "unknown";
}

Result<MuseGlimmerArtifactMode> resolve_artifact_mode(
    Module* module,
    MuseGlimmerArtifactMode requested_mode) {
  auto method_names_result = module->method_names();
  ET_CHECK_OK_OR_RETURN_ERROR(method_names_result.error());
  const auto& methods = method_names_result.get();
  const bool has_embeddings_method =
      methods.count(kTargetForwardFromEmbeddingsMethod) != 0;
  const bool has_embed_text_method = methods.count(kEmbedTextMethod) != 0;
  const bool has_vision_method =
      methods.count(kMuseGlimmerVisionEncoderMethod) != 0;
  const bool has_any_embeddings_target =
      has_embeddings_method || has_embed_text_method || has_vision_method;
  const bool has_embeddings_target =
      has_embeddings_method && has_embed_text_method;
  const bool has_draft = methods.count(kDraftForwardMethod) != 0;
  ET_CHECK_OR_RETURN_ERROR(
      !has_draft || !has_any_embeddings_target || has_embeddings_target,
      InvalidProgram,
      "DFlash artifact must expose target_forward_from_embeddings and "
      "embed_text together, plus vision_encoder when multimodal");
  const bool has_dflash = has_embeddings_target && has_draft;
#ifdef EXECUTORCH_BUILD_MLX
  const bool has_autoregressive = has_embed_text_method &&
      methods.count(kForwardFromEmbeddingsMethod) != 0 && !has_draft;
#else
  const bool has_autoregressive = has_embed_text_method &&
      methods.count(kForwardFromEmbeddingsMethod) != 0 &&
      methods.count(kDecodeFromEmbeddingMethod) != 0 && !has_draft;
#endif
  ET_CHECK_OR_RETURN_ERROR(
      has_autoregressive || has_dflash,
      InvalidProgram,
#ifdef EXECUTORCH_BUILD_MLX
      "Invalid Muse Glimmer artifact. MLX Solo expects embed_text and "
      "forward_from_embeddings. DFlash expects "
      "target_forward_from_embeddings, embed_text, and draft_forward. "
      "Re-export it.");
#else
      "Invalid Muse Glimmer artifact. Solo expects embed_text, "
      "forward_from_embeddings, and decode_from_embedding. DFlash expects a "
      "target_forward_from_embeddings, embed_text, and draft_forward. "
      "Re-export it.");
#endif

  ET_CHECK_OR_RETURN_ERROR(
      has_dflash != has_autoregressive,
      InvalidProgram,
      "Muse Glimmer artifact must expose exactly one complete execution contract: "
      "autoregressive=%d dflash=%d",
      static_cast<int>(has_autoregressive),
      static_cast<int>(has_dflash));
  const MuseGlimmerArtifactMode detected = has_dflash
      ? MuseGlimmerArtifactMode::DFlash
      : MuseGlimmerArtifactMode::Autoregressive;
  ET_CHECK_OR_RETURN_ERROR(
      requested_mode == MuseGlimmerArtifactMode::Auto ||
          requested_mode == detected,
      InvalidArgument,
      "Requested Muse Glimmer artifact mode %s does not match detected mode %s",
      artifact_mode_name(requested_mode),
      artifact_mode_name(detected));
  return detected;
}

Result<int64_t> get_required_int_metadata(Module* module, const char* name) {
  auto value = module->get(name);
  ET_CHECK_OK_OR_RETURN_ERROR(value.error());
  ET_CHECK_OR_RETURN_ERROR(
      value->isScalar(), InvalidProgram, "%s must return a scalar", name);
  return value->toScalar().to<int64_t>();
}

Result<std::string> get_required_string_metadata(
    Module* module,
    const char* name) {
  auto value = module->get(name);
  ET_CHECK_OK_OR_RETURN_ERROR(value.error());
  ET_CHECK_OR_RETURN_ERROR(
      value->isString(), InvalidProgram, "%s must return a string", name);
  return std::string(value->toString());
}

Result<uint64_t> read_sampled_token(
    const executorch::aten::Tensor& output,
    float temperature,
    bool use_sampling) {
#ifdef EXECUTORCH_BUILD_CUDA
  (void)temperature;
  (void)use_sampling;
  const void* ptr = output.const_data_ptr();
  cudaPointerAttributes attrs{};
  const bool on_device = cudaPointerGetAttributes(&attrs, ptr) == cudaSuccess &&
      attrs.type == cudaMemoryTypeDevice;

  auto copy_scalar = [&](void* dst, size_t nbytes) -> Error {
    if (on_device) {
      if (cudaMemcpy(dst, ptr, nbytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        return Error::Internal;
      }
    } else {
      std::memcpy(dst, ptr, nbytes);
    }
    return Error::Ok;
  };

  if (output.scalar_type() == executorch::aten::ScalarType::Long) {
    int64_t val = 0;
    if (copy_scalar(&val, sizeof(val)) != Error::Ok) {
      ET_LOG(Error, "read_sampled_token: cudaMemcpy D2H failed");
      return Error::Internal;
    }
    return static_cast<uint64_t>(val);
  }
  if (output.scalar_type() == executorch::aten::ScalarType::Float) {
    float val = 0.0f;
    if (copy_scalar(&val, sizeof(val)) != Error::Ok) {
      ET_LOG(Error, "read_sampled_token: cudaMemcpy D2H failed");
      return Error::Internal;
    }
    return static_cast<uint64_t>(llrintf(val));
  }
  ET_LOG(
      Error,
      "read_sampled_token: expected Long or Float scalar output, got %d",
      static_cast<int>(output.scalar_type()));
  return Error::InvalidArgument;
#else
  if (use_sampling) {
    ET_CHECK_OR_RETURN_ERROR(
        output.scalar_type() == executorch::aten::ScalarType::Long,
        InvalidProgram,
        "read_sampled_token: use_sampling set but forward output is not Long");
    return static_cast<uint64_t>(output.const_data_ptr<int64_t>()[0]);
  }
  return static_cast<uint64_t>(
      logits_to_token(output, temperature < 0.0f ? 0.0f : temperature));
#endif
}

Result<std::unique_ptr<Module>> build_muse_glimmer_module(
    const MuseGlimmerConfig& config,
    MuseGlimmerArtifactMode artifact_mode,
    bool has_vision,
    bool multi_session) {
  std::vector<std::string> data_files;
  if (!config.data_path.empty()) {
    data_files.push_back(config.data_path);
  }
  bool share_memory_arenas = true;
#ifdef EXECUTORCH_BUILD_CUDA
  share_memory_arenas = false;
#endif
  auto module = std::make_unique<Module>(
      config.model_path,
      data_files,
      Module::LoadMode::MmapUseMlockIgnoreErrors,
      /*event_tracer=*/nullptr,
      /*memory_allocator=*/nullptr,
      /*temp_allocator=*/nullptr,
      /*share_memory_arenas=*/share_memory_arenas);

#ifdef EXECUTORCH_BUILD_CUDA
  executorch::runtime::BackendOptions<3> cuda_opts;
  ET_CHECK_OK_OR_RETURN_ERROR(
      cuda_opts.set_option("use_shared_cuda_stream", true));
  ET_CHECK_OK_OR_RETURN_ERROR(
      cuda_opts.set_option("weight_sharing_across_methods", true));
  if (config.enable_cuda_graph) {
    const char* graph_methods = artifact_mode == MuseGlimmerArtifactMode::DFlash
        ? "target_forward_from_embeddings,draft_forward"
        : kDecodeFromEmbeddingMethod;
    ET_CHECK_OK_OR_RETURN_ERROR(
        cuda_opts.set_option("enable_cuda_graph_for_method", graph_methods));
    ET_LOG(Info, "MuseGlimmerEngine: CUDA graph enabled for %s", graph_methods);
  }
  ET_CHECK_OK_OR_RETURN_ERROR(
      executorch::runtime::set_option("CudaBackend", cuda_opts.view()));
#endif

  const executorch::runtime::LoadBackendOptionsMap* load_options = nullptr;
#ifdef EXECUTORCH_BUILD_MLX
  // Per-model MLX runtime specs, delivered to MLXBackend::init(). Must outlive
  // the load_method calls below (they read it during init).
  executorch::runtime::BackendOptions<2> mlx_opts;
  executorch::runtime::LoadBackendOptionsMap mlx_options_map;
  // Release MLX's cached buffer pool every N forward calls to bound memory
  // growth during long sessions.
  constexpr int kMLXClearCacheInterval = 1;
  ET_CHECK_OK_OR_RETURN_ERROR(mlx_opts.set_option(
      ::executorch::backends::mlx::kClearCacheIntervalKey,
      kMLXClearCacheInterval));
  ET_LOG(
      Info,
      "MuseGlimmerEngine: MLX clear_cache_interval=%d",
      kMLXClearCacheInterval);
  // skip_mutable_buffer_init must match the multi-session owner: it is only
  // safe when a load scope is active (the caller passes multi_session = "owner
  // exists"), since per-session buffers are then allocated by
  // mlx_mutable_state. The backend also defensively rejects the flag without an
  // active owner.
  if (multi_session) {
    ET_CHECK_OK_OR_RETURN_ERROR(mlx_opts.set_option(
        ::executorch::backends::mlx::kSkipMutableBufferInitKey, true));
    ET_LOG(Info, "MuseGlimmerEngine: MLX skip_mutable_buffer_init=true");
  }
  ET_CHECK_OK_OR_RETURN_ERROR(mlx_options_map.set_options(
      ::executorch::backends::mlx::kMLXBackendId, mlx_opts.view()));
  load_options = &mlx_options_map;
#endif

  if (artifact_mode == MuseGlimmerArtifactMode::DFlash) {
#ifdef EXECUTORCH_BUILD_CUDA
    auto method_names_result = module->method_names();
    ET_CHECK_OK_OR_RETURN_ERROR(method_names_result.error());
    if (method_names_result.get().count(kTargetPrefillFromEmbeddingsMethod) !=
        0) {
      ET_CHECK_OK_OR_RETURN_ERROR(module->load_method(
          kTargetPrefillFromEmbeddingsMethod, nullptr, nullptr, load_options));
    }
    if (method_names_result.get().count(kDraftPrefillMethod) != 0) {
      ET_CHECK_OK_OR_RETURN_ERROR(module->load_method(
          kDraftPrefillMethod, nullptr, nullptr, load_options));
    }
#endif
    ET_CHECK_OK_OR_RETURN_ERROR(module->load_method(
        kTargetForwardFromEmbeddingsMethod, nullptr, nullptr, load_options));
    ET_CHECK_OK_OR_RETURN_ERROR(
        module->load_method(kEmbedTextMethod, nullptr, nullptr, load_options));
    if (has_vision) {
      ET_CHECK_OK_OR_RETURN_ERROR(module->load_method(
          kMuseGlimmerVisionEncoderMethod, nullptr, nullptr, load_options));
    }
    ET_CHECK_OK_OR_RETURN_ERROR(module->load_method(
        kDraftForwardMethod, nullptr, nullptr, load_options));
  } else {
    ET_CHECK_OK_OR_RETURN_ERROR(
        module->load_method(kEmbedTextMethod, nullptr, nullptr, load_options));
    ET_CHECK_OK_OR_RETURN_ERROR(module->load_method(
        kForwardFromEmbeddingsMethod, nullptr, nullptr, load_options));
#ifndef EXECUTORCH_BUILD_MLX
    ET_CHECK_OK_OR_RETURN_ERROR(module->load_method(
        kDecodeFromEmbeddingMethod, nullptr, nullptr, load_options));
#endif
    if (has_vision) {
      ET_CHECK_OK_OR_RETURN_ERROR(module->load_method(
          kMuseGlimmerVisionEncoderMethod, nullptr, nullptr, load_options));
    }
  }
  return module;
}

std::string default_pos_embed_path(const std::string& model_path) {
  const auto separator = model_path.find_last_of("/\\");
  const std::string directory =
      separator == std::string::npos ? "." : model_path.substr(0, separator);
  return directory + "/pos_embed.bin";
}

void add_token_piece(
    ::tokenizers::Tokenizer* tokenizer,
    std::unordered_set<uint64_t>& ids,
    const char* piece) {
  if (auto id = tokenizer->piece_to_id(piece); id.ok()) {
    ids.insert(*id);
  }
}

#ifdef EXECUTORCH_BUILD_CUDA
Error register_mutable_fqns(
    Module* module,
    ::executorch::backends::cuda::MutableStateContextOwner& mutable_state) {
  auto res = module->execute("get_mutable_buffer_metadata");
  if (res.error() != Error::Ok) {
    ET_LOG(
        Info,
        "MuseGlimmerEngine: model has no get_mutable_buffer_metadata; "
        "multi-session disabled");
    return res.error();
  }
  const auto& outs = res.get();
  if (outs.empty() || !outs[0].isString()) {
    ET_LOG(Error, "get_mutable_buffer_metadata did not return a string");
    return Error::InvalidProgram;
  }
  std::string json_str(outs[0].toString());
  auto j = nlohmann::json::parse(json_str, nullptr, /*allow_exceptions=*/false);
  if (j.is_discarded() || !j.is_object()) {
    ET_LOG(Error, "get_mutable_buffer_metadata is not a valid JSON object");
    return Error::InvalidProgram;
  }
  if (!j.contains("version") || !j["version"].is_number_integer() ||
      j["version"].get<int>() != 1) {
    ET_LOG(Error, "get_mutable_buffer_metadata: unsupported/missing version");
    return Error::InvalidProgram;
  }
  if (!j.contains("mutable_buffers") || !j["mutable_buffers"].is_array() ||
      j["mutable_buffers"].empty()) {
    ET_LOG(
        Error,
        "get_mutable_buffer_metadata: mutable_buffers must be a non-empty array");
    return Error::InvalidProgram;
  }
  std::vector<std::string> fqns;
  for (const auto& f : j["mutable_buffers"]) {
    if (!f.is_string() || f.get<std::string>().empty()) {
      ET_LOG(
          Error,
          "get_mutable_buffer_metadata: every mutable_buffers entry must be a "
          "non-empty string");
      return Error::InvalidProgram;
    }
    fqns.push_back(f.get<std::string>());
  }
  mutable_state.register_fqns(fqns);
  return Error::Ok;
}

TensorPtr build_decode_pos_table(
    const std::unordered_map<std::string, int64_t>& metadata) {
  auto ctx_it = metadata.find(kMaxContextLen);
  if (ctx_it == metadata.end() || ctx_it->second <= 0) {
    return nullptr;
  }
  std::vector<int64_t> pos_data(ctx_it->second);
  for (int64_t i = 0; i < ctx_it->second; ++i) {
    pos_data[i] = i;
  }
  return clone_tensor_ptr_to(
      from_blob(
          pos_data.data(),
          {static_cast<SizesType>(pos_data.size())},
          executorch::aten::ScalarType::Long),
      executorch::aten::Device(executorch::aten::DeviceType::CUDA, 0));
}
#endif

class MuseGlimmerSession : public LLMSession,
                           public MuseGlimmerMultimodalSession {
 public:
  MuseGlimmerSession(
      Module* module,
      std::mutex* exec_mutex,
      std::atomic<int>* live_sessions,
      ::tokenizers::Tokenizer* tokenizer,
      std::unordered_map<std::string, int64_t> metadata,
      std::unordered_set<uint64_t> eos_ids,
      int64_t max_prefill_chunk,
      int64_t min_prefill_chunk,
      TensorPtr decode_pos_table_dev,
      MuseGlimmerMutableStateContextOwner* mutable_state,
      int session_token)
      : module_(module),
        exec_mutex_(exec_mutex),
        live_sessions_(live_sessions),
        tokenizer_(tokenizer),
        metadata_(std::move(metadata)),
        eos_ids_(std::move(eos_ids)),
        max_prefill_chunk_(max_prefill_chunk),
        min_prefill_chunk_(min_prefill_chunk),
#ifdef EXECUTORCH_BUILD_CUDA
        decode_pos_table_dev_(std::move(decode_pos_table_dev)),
#endif
        mutable_state_(mutable_state),
        session_token_(session_token) {
    if (auto it = metadata_.find(kUseSampling); it != metadata_.end()) {
      use_sampling_ = it->second != 0;
    }
    decode_tokens_ = from_blob(
        decode_token_data_, {1, 1}, executorch::aten::ScalarType::Long);
    decode_pos_ =
        from_blob(decode_pos_data_, {1}, executorch::aten::ScalarType::Long);
#ifdef EXECUTORCH_BUILD_CUDA
    decode_tokens_dev_ = clone_tensor_ptr_to(decode_tokens_, cuda_device_);
    decode_pos_dev_ = clone_tensor_ptr_to(decode_pos_, cuda_device_);
    auto temp_host =
        from_blob(&temp_val_, {1}, executorch::aten::ScalarType::Float);
    temp_tensor_dev_ = clone_tensor_ptr_to(temp_host, cuda_device_);
#endif
#ifdef EXECUTORCH_BUILD_MLX
    temp_tensor_mlx_ =
        from_blob(&temp_val_mlx_, {}, executorch::aten::ScalarType::Float);
    top_k_tensor_ =
        from_blob(&top_k_val_, {}, executorch::aten::ScalarType::Long);
    top_p_tensor_ =
        from_blob(&top_p_val_, {}, executorch::aten::ScalarType::Float);
    seed_tensor_ =
        from_blob(&seed_val_, {}, executorch::aten::ScalarType::Long);
#endif
  }

  ~MuseGlimmerSession() override {
    if (mutable_state_ != nullptr &&
        session_token_ != kMuseGlimmerNoMutableSession) {
      mutable_state_->destroy_session(session_token_);
    }
    if (live_sessions_ != nullptr) {
      live_sessions_->fetch_sub(1);
    }
  }

  Error prefill_tokens(
      const std::vector<uint64_t>& tokens,
      const SamplingConfig* initial_sampling) override {
    std::optional<PreparedMuseGlimmerImage> image = std::move(staged_image_);
    staged_image_.reset();
    preserve_staged_image_on_next_reset_ = false;
    if (tokens.empty()) {
      ET_LOG(Error, "prefill_tokens: empty token list");
      return Error::InvalidArgument;
    }
    int64_t next_image_row = 0;
    if (image.has_value()) {
      ET_CHECK_OR_RETURN_ERROR(
          pos_ == 0,
          InvalidState,
          "Image prefill requires a freshly reset Muse Glimmer session");
      ET_CHECK_OK_OR_RETURN_ERROR(validate_staged_image(tokens, *image));
    }
    float first_token_temp = temperature_;
    if (initial_sampling != nullptr) {
      if (!use_sampling_ &&
          (initial_sampling->top_k != 0 || initial_sampling->top_p != 1.0f ||
           initial_sampling->seed != 0)) {
        ET_LOG(
            Error,
            "prefill_tokens: top_k/top_p/seed require a sampling model "
            "(export with --sample); only temperature is supported otherwise");
        return Error::NotSupported;
      }
      first_token_temp = initial_sampling->temperature;
      if (use_sampling_) {
        if (!valid_top_p(initial_sampling->top_p)) {
          ET_LOG(Error, "prefill_tokens: top_p must be in (0, 1]");
          return Error::InvalidArgument;
        }
        top_k_ = initial_sampling->top_k;
        top_p_ = initial_sampling->top_p;
        seed_ = initial_sampling->seed;
      }
    }
    if (!valid_temperature(first_token_temp)) {
      ET_LOG(Error, "prefill_tokens: temperature must be -1 or in [0, 2]");
      return Error::InvalidArgument;
    }

    const int64_t T = static_cast<int64_t>(tokens.size());
    const auto ctx_it = metadata_.find(kMaxContextLen);
    if (ctx_it != metadata_.end() && pos_ + T >= ctx_it->second) {
      ET_LOG(
          Error,
          "prefill_tokens would leave no room to generate (pos %" PRId64
          " + %" PRId64 " >= max_context %" PRId64 ")",
          pos_,
          T,
          ctx_it->second);
      return Error::InvalidArgument;
    }

    stop_.store(false, std::memory_order_relaxed);
    int64_t offset = 0;
    while (offset < T) {
      int64_t chunk = T - offset;
      if (max_prefill_chunk_ > 0) {
        chunk = std::min(chunk, max_prefill_chunk_);
      }
#ifdef EXECUTORCH_BUILD_CUDA
      if (chunk > 1 && chunk < min_prefill_chunk_) {
        chunk = 1;
      }
#endif
      auto sampled = run_prefill_chunk(
          tokens.data() + offset,
          chunk,
          first_token_temp,
          image.has_value() ? &*image : nullptr,
          image.has_value() ? &next_image_row : nullptr);
      ET_CHECK_OK_OR_RETURN_ERROR(sampled.error());
      pending_ = sampled.get();
      pos_ += chunk;
      offset += chunk;
    }
    ET_CHECK_OR_RETURN_ERROR(
        !image.has_value() || next_image_row == image->num_soft_tokens,
        InvalidState,
        "Image prefill did not consume every image embedding row");
    prev_decode_token_ = tokens.back();
#ifdef EXECUTORCH_BUILD_MLX
    if (use_sampling_) {
      seed_ += 1;
    }
#endif
    return Error::Ok;
  }

  Result<DecodeResult> decode_one(const SamplingConfig& sampling) override {
    if (!use_sampling_ &&
        (sampling.top_k != 0 || sampling.top_p != 1.0f || sampling.seed != 0)) {
      ET_LOG(
          Error,
          "MuseGlimmerSession: top_k/top_p/seed require a sampling model "
          "(export with --sample); only temperature is supported otherwise");
      return Error::NotSupported;
    }
    if (!valid_temperature(sampling.temperature)) {
      ET_LOG(Error, "decode_one: temperature must be -1 or in [0, 2]");
      return Error::InvalidArgument;
    }
    ET_CHECK_OR_RETURN_ERROR(
        pending_.has_value(),
        InvalidState,
        "decode_one requires a pending token; call prefill_tokens() first");
    temperature_ = sampling.temperature;
    if (use_sampling_) {
      if (!valid_top_p(sampling.top_p)) {
        ET_LOG(Error, "decode_one: top_p must be in (0, 1]");
        return Error::InvalidArgument;
      }
      top_k_ = sampling.top_k;
      top_p_ = sampling.top_p;
    }

    if (stop_.load(std::memory_order_relaxed)) {
      return DecodeResult{0, "", /*is_eos=*/false, /*is_terminal=*/true};
    }

    const uint64_t token = pending_.value();
    const bool is_eos = eos_ids_.find(token) != eos_ids_.end();
    const uint64_t prev = prev_decode_token_.value_or(token);
    auto dec = tokenizer_->decode(prev, token);
    if (!dec.ok()) {
      ET_LOG(
          Error,
          "Tokenizers error code %d",
          static_cast<uint32_t>(dec.error()));
      return Error::InvalidArgument;
    }
    std::string text_piece = std::move(*dec);

    if (is_eos) {
      pending_.reset();
      return DecodeResult{
          token, std::move(text_piece), is_eos, /*is_terminal=*/true};
    }

    const auto ctx_it = metadata_.find(kMaxContextLen);
    if (ctx_it != metadata_.end()) {
      ET_CHECK_OR_RETURN_ERROR(
          pos_ < ctx_it->second,
          InvalidArgument,
          "decode_one would exceed context capacity: pos_ %" PRId64
          " >= max_context %" PRId64,
          pos_,
          ctx_it->second);
    }

    decode_token_data_[0] = static_cast<int64_t>(token);
    decode_pos_data_[0] = pos_;
    std::vector<EValue> inputs;
#ifdef EXECUTORCH_BUILD_CUDA
    ET_CHECK_OK_OR_RETURN_ERROR(update_decode_pos_on_cuda());
    ET_CHECK_OK_OR_RETURN_ERROR(set_temperature(temperature_));
    inputs.push_back(EValue(decode_tokens_dev_));
    inputs.push_back(EValue(decode_pos_dev_));
    inputs.push_back(EValue(temp_tensor_dev_));
#else
    inputs.push_back(EValue(decode_tokens_));
    inputs.push_back(EValue(decode_pos_));
#ifdef EXECUTORCH_BUILD_MLX
    if (use_sampling_) {
      set_sampling_inputs(temperature_, top_k_, top_p_, seed_);
      inputs.push_back(EValue(temp_tensor_mlx_));
      inputs.push_back(EValue(top_k_tensor_));
      inputs.push_back(EValue(top_p_tensor_));
      inputs.push_back(EValue(seed_tensor_));
    }
#endif
#endif
    // MLX must reuse the dynamic prefill method so decode observes the KV
    // cache written by prefill. Separate delegated methods do not share MLX
    // mutable state reliably even when their ExecuTorch arenas are shared.
#ifdef EXECUTORCH_BUILD_MLX
    constexpr const char* kDecodeMethod = kForwardFromEmbeddingsMethod;
#else
    constexpr const char* kDecodeMethod = kDecodeFromEmbeddingMethod;
#endif
    auto sampled = run_locked(
        kDecodeMethod,
        inputs,
        temperature_,
        /*sync_after=*/false,
        /*token_ids=*/nullptr,
        1,
        /*image=*/nullptr,
        /*next_image_row=*/nullptr);
    ET_CHECK_OK_OR_RETURN_ERROR(sampled.error());
    pending_ = sampled.get();
    prev_decode_token_ = token;
    pos_ += 1;
#ifdef EXECUTORCH_BUILD_MLX
    if (use_sampling_) {
      seed_ += 1;
    }
#endif
    return DecodeResult{
        token, std::move(text_piece), /*is_eos=*/false, /*is_terminal=*/false};
  }

  int64_t position() const override {
    return pos_;
  }

  Error reset() override {
    pos_ = 0;
    pending_.reset();
    prev_decode_token_.reset();
    if (preserve_staged_image_on_next_reset_) {
      preserve_staged_image_on_next_reset_ = false;
    } else {
      staged_image_.reset();
    }
    stop_.store(false, std::memory_order_relaxed);
    return Error::Ok;
  }

  void stop() override {
    stop_.store(true, std::memory_order_relaxed);
  }

  void stage_image_for_next_prefill(PreparedMuseGlimmerImage image) override {
    staged_image_ = std::move(image);
    preserve_staged_image_on_next_reset_ = true;
  }

  void clear_staged_image() override {
    staged_image_.reset();
    preserve_staged_image_on_next_reset_ = false;
  }

 private:
  Error validate_staged_image(
      const std::vector<uint64_t>& tokens,
      const PreparedMuseGlimmerImage& image) const {
    ET_CHECK_OR_RETURN_ERROR(
        image.num_soft_tokens > 0 && image.hidden_dim > 0 &&
            image.embeddings.size() ==
                static_cast<size_t>(image.num_soft_tokens * image.hidden_dim),
        InvalidArgument,
        "Staged image data does not match its dimensions");

    // A contiguous run of <|patch|> tokens identifies the image span.
    std::optional<int64_t> first;
    int64_t last = -1;
    int64_t patch_count = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(tokens.size()); ++i) {
      if (tokens[i] == kMuseGlimmerImagePatchTokenId) {
        if (!first.has_value()) {
          first = i;
        }
        last = i;
        ++patch_count;
      }
    }
    ET_CHECK_OR_RETURN_ERROR(
        patch_count == image.num_soft_tokens,
        InvalidArgument,
        "Image span patch count does not match staged embeddings");
    ET_CHECK_OR_RETURN_ERROR(
        first.has_value() && last - *first + 1 == patch_count,
        InvalidArgument,
        "Image span must be one contiguous run of patch tokens");
    return Error::Ok;
  }

  Error
  copy_bytes_to_host(void* destination, const void* source, size_t bytes) {
#ifdef EXECUTORCH_BUILD_CUDA
    cudaPointerAttributes attributes{};
    const bool on_device =
        cudaPointerGetAttributes(&attributes, source) == cudaSuccess &&
        attributes.type == cudaMemoryTypeDevice;
    if (on_device) {
      return cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost) ==
              cudaSuccess
          ? Error::Ok
          : Error::Internal;
    }
#endif
    std::memcpy(destination, source, bytes);
    return Error::Ok;
  }

  static bool valid_temperature(float temperature) {
    return temperature == -1.0f || (temperature >= 0.0f && temperature <= 2.0f);
  }

  static bool valid_top_p(float top_p) {
    return top_p > 0.0f && top_p <= 1.0f;
  }

#ifdef EXECUTORCH_BUILD_MLX
  void
  set_sampling_inputs(float temp, int64_t top_k, float top_p, uint64_t seed) {
    temp_val_mlx_ = (temp < 0.0f) ? 0.0f : temp;
    top_k_val_ = (top_k <= 0) ? INT64_MAX : top_k; // 0/neg = keep all
    top_p_val_ = top_p;
    seed_val_ = static_cast<int64_t>(seed);
  }
#endif

  Result<uint64_t> run_prefill_chunk(
      const uint64_t* tokens,
      int64_t T,
      float temperature,
      const PreparedMuseGlimmerImage* image,
      int64_t* next_image_row) {
    std::vector<int64_t> token_data(tokens, tokens + T);
    std::vector<int64_t> pos_data(T);
    for (int64_t i = 0; i < T; ++i) {
      pos_data[i] = pos_ + i;
    }
    auto tokens_tensor = from_blob(
        token_data.data(),
        {1, static_cast<SizesType>(T)},
        executorch::aten::ScalarType::Long);
    auto pos_tensor = from_blob(
        pos_data.data(),
        {static_cast<SizesType>(T)},
        executorch::aten::ScalarType::Long);
    std::vector<EValue> inputs;
    TensorPtr token_input = tokens_tensor;
    TensorPtr pos_input = pos_tensor;
#ifdef EXECUTORCH_BUILD_CUDA
    std::vector<TensorPtr> device_inputs;
    token_input = to_cuda(token_input, device_inputs);
    pos_input = to_cuda(pos_input, device_inputs);
    ET_CHECK_OK_OR_RETURN_ERROR(set_temperature(temperature));
#endif
    inputs.push_back(EValue(token_input));
    inputs.push_back(EValue(pos_input));
#ifdef EXECUTORCH_BUILD_CUDA
    inputs.push_back(EValue(temp_tensor_dev_));
#endif
#ifdef EXECUTORCH_BUILD_MLX
    constexpr const char* method = kForwardFromEmbeddingsMethod;
#else
    const char* method = T >= min_prefill_chunk_ ? kForwardFromEmbeddingsMethod
                                                 : kDecodeFromEmbeddingMethod;
#endif
#ifdef EXECUTORCH_BUILD_MLX
    if (use_sampling_) {
      set_sampling_inputs(temperature, top_k_, top_p_, seed_);
      inputs.push_back(EValue(temp_tensor_mlx_));
      inputs.push_back(EValue(top_k_tensor_));
      inputs.push_back(EValue(top_p_tensor_));
      inputs.push_back(EValue(seed_tensor_));
    }
#endif
    return run_locked(
        method,
        inputs,
        temperature,
        /*sync_after=*/true,
        tokens,
        T,
        image,
        next_image_row);
  }

#ifdef EXECUTORCH_BUILD_CUDA
  TensorPtr to_cuda(TensorPtr tensor, std::vector<TensorPtr>& keep_alive) {
    keep_alive.push_back(clone_tensor_ptr_to(tensor, cuda_device_));
    return keep_alive.back();
  }

  Error set_temperature(float temperature) {
    if (!valid_temperature(temperature)) {
      return Error::InvalidArgument;
    }
    temp_val_ = (temperature <= 0.0f) ? 1e-6f : temperature;
    if (cudaMemcpy(
            temp_tensor_dev_->mutable_data_ptr(),
            &temp_val_,
            sizeof(float),
            cudaMemcpyHostToDevice) != cudaSuccess) {
      ET_LOG(Error, "set_temperature: cudaMemcpy H2D failed");
      return Error::Internal;
    }
    return Error::Ok;
  }

  Error copy_decode_token_to_cuda(uint64_t token) {
    const int64_t token_value = static_cast<int64_t>(token);
    if (cudaMemcpy(
            decode_tokens_dev_->mutable_data_ptr(),
            &token_value,
            sizeof(int64_t),
            cudaMemcpyHostToDevice) != cudaSuccess) {
      ET_LOG(Error, "copy_decode_token_to_cuda: token H2D failed");
      return Error::Internal;
    }
    return Error::Ok;
  }

  Error stage_next_decode_token_on_cuda(
      const executorch::aten::Tensor& out_tensor,
      uint64_t token) {
    if (out_tensor.scalar_type() == executorch::aten::ScalarType::Long) {
      const void* ptr = out_tensor.const_data_ptr();
      cudaPointerAttributes attrs{};
      const bool on_device =
          cudaPointerGetAttributes(&attrs, ptr) == cudaSuccess &&
          attrs.type == cudaMemoryTypeDevice;
      if (cudaMemcpy(
              decode_tokens_dev_->mutable_data_ptr(),
              ptr,
              sizeof(int64_t),
              on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice) !=
          cudaSuccess) {
        ET_LOG(Error, "stage_next_decode_token_on_cuda: token copy failed");
        return Error::Internal;
      }
      return Error::Ok;
    }
    return copy_decode_token_to_cuda(token);
  }

  Error update_decode_pos_on_cuda() {
    if (decode_pos_table_dev_ != nullptr) {
      auto* pos_table =
          static_cast<int64_t*>(decode_pos_table_dev_->mutable_data_ptr());
      auto* pos_slot =
          static_cast<int64_t*>(decode_pos_dev_->mutable_data_ptr());
      if (cudaMemcpy(
              pos_slot,
              pos_table + pos_,
              sizeof(int64_t),
              cudaMemcpyDeviceToDevice) != cudaSuccess) {
        ET_LOG(Error, "update_decode_pos_on_cuda: position D2D failed");
        return Error::Internal;
      }
      return Error::Ok;
    }
    if (cudaMemcpy(
            decode_pos_dev_->mutable_data_ptr(),
            decode_pos_data_,
            sizeof(int64_t),
            cudaMemcpyHostToDevice) != cudaSuccess) {
      ET_LOG(Error, "update_decode_pos_on_cuda: position H2D failed");
      return Error::Internal;
    }
    return Error::Ok;
  }
#endif

  Result<uint64_t> run_locked(
      const char* method,
      std::vector<EValue>& inputs,
      float temperature,
      bool sync_after,
      const uint64_t* token_ids,
      int64_t token_count,
      const PreparedMuseGlimmerImage* image,
      int64_t* next_image_row) {
    std::lock_guard<std::mutex> guard(*exec_mutex_);
    auto execute_contract = [&]() -> Result<std::vector<EValue>> {
      auto embed_outputs = module_->execute(kEmbedTextMethod, {inputs[0]});
      ET_CHECK_OK_OR_RETURN_ERROR(embed_outputs.error());
      ET_CHECK_OR_RETURN_ERROR(
          embed_outputs->size() == 1 && (*embed_outputs)[0].isTensor(),
          InvalidProgram,
          "embed_text must return one tensor");
      const auto& embed_tensor = (*embed_outputs)[0].toTensor();
      ET_CHECK_OR_RETURN_ERROR(
          embed_tensor.dim() == 3 && embed_tensor.size(0) == 1 &&
              embed_tensor.size(1) == token_count &&
              (embed_tensor.scalar_type() ==
                   executorch::aten::ScalarType::BFloat16 ||
               embed_tensor.scalar_type() ==
                   executorch::aten::ScalarType::Half),
          InvalidProgram,
          "embed_text must return [1, T, H] in the activation dtype");

      std::vector<uint16_t> spliced_embeddings;
      TensorPtr spliced_tensor;
      std::vector<EValue> forward_inputs = inputs;
      if (image != nullptr) {
        ET_CHECK_OR_RETURN_ERROR(
            token_ids != nullptr && next_image_row != nullptr &&
                embed_tensor.size(2) == image->hidden_dim,
            InvalidArgument,
            "Image embedding dimension does not match embed_text");
        const int64_t hidden_dim = embed_tensor.size(2);
        spliced_embeddings.resize(token_count * hidden_dim);
        ET_CHECK_OK_OR_RETURN_ERROR(copy_bytes_to_host(
            spliced_embeddings.data(),
            embed_tensor.const_data_ptr(),
            spliced_embeddings.size() * sizeof(uint16_t)));
        for (int64_t row = 0; row < token_count; ++row) {
          if (token_ids[row] != kMuseGlimmerImagePatchTokenId) {
            continue;
          }
          ET_CHECK_OR_RETURN_ERROR(
              *next_image_row < image->num_soft_tokens,
              InvalidArgument,
              "Image prefill has more patch tokens than image rows");
          std::memcpy(
              spliced_embeddings.data() + row * hidden_dim,
              image->embeddings.data() + *next_image_row * hidden_dim,
              hidden_dim * sizeof(uint16_t));
          ++*next_image_row;
        }
        spliced_tensor = from_blob(
            spliced_embeddings.data(),
            {1,
             static_cast<SizesType>(token_count),
             static_cast<SizesType>(hidden_dim)},
            embed_tensor.scalar_type());
        forward_inputs[0] = EValue(spliced_tensor);
      } else {
        forward_inputs[0] = (*embed_outputs)[0];
      }
      return module_->execute(method, forward_inputs);
    };
    auto res = mutable_state_ != nullptr
        ? mutable_state_->with_active_session(session_token_, execute_contract)
        : execute_contract();
    ET_CHECK_OK_OR_RETURN_ERROR(res.error());
    const auto& out_tensor = res.get()[0].toTensor();
    auto sampled = read_sampled_token(out_tensor, temperature, use_sampling_);
    ET_CHECK_OK_OR_RETURN_ERROR(sampled.error());
#ifdef EXECUTORCH_BUILD_CUDA
    ET_CHECK_OK_OR_RETURN_ERROR(
        stage_next_decode_token_on_cuda(out_tensor, sampled.get()));
#endif
#ifdef EXECUTORCH_BUILD_CUDA
    if (sync_after && cudaDeviceSynchronize() != cudaSuccess) {
      ET_LOG(Error, "run_locked: cudaDeviceSynchronize failed");
      return Error::Internal;
    }
#else
    (void)sync_after;
#endif
    return sampled.get();
  }

  Module* module_;
  std::mutex* exec_mutex_;
  std::atomic<int>* live_sessions_;
  ::tokenizers::Tokenizer* tokenizer_;
  std::unordered_map<std::string, int64_t> metadata_;
  std::unordered_set<uint64_t> eos_ids_;
  int64_t max_prefill_chunk_;
  int64_t min_prefill_chunk_;

  int64_t pos_ = 0;
  std::optional<uint64_t> pending_;
  std::optional<uint64_t> prev_decode_token_;
  float temperature_ = -1.0f;
  std::atomic<bool> stop_{false};
  std::optional<PreparedMuseGlimmerImage> staged_image_;
  bool preserve_staged_image_on_next_reset_ = false;

  bool use_sampling_ = false;
  int64_t top_k_ = 0; // 0 = off (keep all); mapped to INT64_MAX on-device
  float top_p_ = 1.0f;
  uint64_t seed_ = 0;

  int64_t decode_token_data_[1] = {0};
  int64_t decode_pos_data_[1] = {0};
  TensorPtr decode_tokens_;
  TensorPtr decode_pos_;
#ifdef EXECUTORCH_BUILD_CUDA
  TensorPtr decode_pos_table_dev_;
#endif
  MuseGlimmerMutableStateContextOwner* mutable_state_ = nullptr;
  int session_token_ = kMuseGlimmerNoMutableSession;
#ifdef EXECUTORCH_BUILD_CUDA
  float temp_val_ = 1e-6f;
  executorch::aten::Device cuda_device_ =
      executorch::aten::Device(executorch::aten::DeviceType::CUDA, 0);
  TensorPtr decode_tokens_dev_;
  TensorPtr decode_pos_dev_;
  TensorPtr temp_tensor_dev_;
#endif
#ifdef EXECUTORCH_BUILD_MLX
  float temp_val_mlx_ = 0.0f;
  int64_t top_k_val_ = INT64_MAX;
  float top_p_val_ = 1.0f;
  int64_t seed_val_ = 0;
  TensorPtr temp_tensor_mlx_;
  TensorPtr top_k_tensor_;
  TensorPtr top_p_tensor_;
  TensorPtr seed_tensor_;
#endif
};

} // namespace

Result<std::unique_ptr<MuseGlimmerEngine>> MuseGlimmerEngine::create(
    const MuseGlimmerConfig& config) {
  if (config.model_path.empty() || config.tokenizer_path.empty()) {
    ET_LOG(
        Error, "MuseGlimmerEngine: model_path and tokenizer_path are required");
    return Error::InvalidArgument;
  }

  auto tokenizer = std::make_unique<::tokenizers::HFTokenizer>();
  if (tokenizer->load(config.tokenizer_path) != ::tokenizers::Error::Ok) {
    ET_LOG(Error, "MuseGlimmerEngine: failed to load tokenizer");
    return Error::InvalidArgument;
  }

  std::vector<std::string> data_files;
  if (!config.data_path.empty()) {
    data_files.push_back(config.data_path);
  }
  auto meta_module = std::make_unique<Module>(
      config.model_path, data_files, Module::LoadMode::File);
  auto artifact_mode_result =
      resolve_artifact_mode(meta_module.get(), config.artifact_mode);
  if (artifact_mode_result.error() != Error::Ok) {
    ET_LOG(Error, "MuseGlimmerEngine: failed to resolve artifact mode");
    return artifact_mode_result.error();
  }
  const MuseGlimmerArtifactMode artifact_mode = artifact_mode_result.get();
  ET_LOG(
      Info,
      "MuseGlimmerEngine: detected %s artifact",
      artifact_mode_name(artifact_mode));

  auto method_names_result = meta_module->method_names();
  ET_CHECK_OK_OR_RETURN_ERROR(method_names_result.error());
  const auto& method_names = method_names_result.get();
  const bool has_vision =
      method_names.count(kMuseGlimmerVisionEncoderMethod) != 0;
#ifdef EXECUTORCH_BUILD_CUDA
  const bool has_dflash_target_prefill =
      artifact_mode == MuseGlimmerArtifactMode::DFlash &&
      method_names.count(kTargetPrefillFromEmbeddingsMethod) != 0;
#endif
  if (has_vision && artifact_mode == MuseGlimmerArtifactMode::DFlash) {
    ET_CHECK_OR_RETURN_ERROR(
        method_names.count(kEmbedTextMethod) != 0 &&
            method_names.count(kTargetForwardFromEmbeddingsMethod) != 0,
        InvalidProgram,
        "Vision DFlash artifact must expose embed_text and "
        "target_forward_from_embeddings");
  }
#ifdef EXECUTORCH_BUILD_CUDA
  ET_CHECK_OR_RETURN_ERROR(
      artifact_mode != MuseGlimmerArtifactMode::DFlash ||
          has_dflash_target_prefill,
      InvalidProgram,
      "CUDA DFlash artifact must expose target_prefill_from_embeddings");
#endif

  auto metadata_result = get_llm_metadata(tokenizer.get(), meta_module.get());
  if (metadata_result.error() != Error::Ok) {
    ET_LOG(Error, "MuseGlimmerEngine: failed to read metadata");
    return metadata_result.error();
  }

  // Validate the patch token only for vision artifacts; text-only tokenizers
  // may omit it.
  if (has_vision) {
    auto patch_id = tokenizer->piece_to_id("<|patch|>");
    ET_CHECK_OR_RETURN_ERROR(
        patch_id.ok(),
        InvalidArgument,
        "Vision artifact needs a tokenizer with a <|patch|> token");
    ET_CHECK_OR_RETURN_ERROR(
        *patch_id == kMuseGlimmerImagePatchTokenId,
        InvalidArgument,
        "Tokenizer maps <|patch|> to %" PRIu64 ", expected %" PRIu64,
        static_cast<uint64_t>(*patch_id),
        kMuseGlimmerImagePatchTokenId);
  }

  auto eos_ids = get_eos_ids(tokenizer.get(), meta_module.get());
  eos_ids.insert(static_cast<uint64_t>(config.eos_id));
  // <|eom|> continues the assistant turn. Add only turn-ending Harmony tokens
  // here so a reasoning message can precede a tool call.
  add_token_piece(tokenizer.get(), eos_ids, "<|eot|>");
  add_token_piece(tokenizer.get(), eos_ids, "<|end_of_text|>");

  auto metadata = metadata_result.get();
  int64_t max_prefill_chunk = 1;
  auto max_ctx_it = metadata.find(kMaxContextLen);
  if (max_ctx_it != metadata.end() && max_ctx_it->second > 1) {
    max_prefill_chunk = max_ctx_it->second - 1;
  }
  if (auto get_result = meta_module->get(kMaxPrefillChunk); get_result.ok()) {
    max_prefill_chunk = get_result->toScalar().to<int64_t>();
    metadata[kMaxPrefillChunk] = max_prefill_chunk;
  }

  if (auto get_result = meta_module->get(kUseSampling); get_result.ok()) {
    metadata[kUseSampling] = get_result->toScalar().to<int64_t>();
  }

  int64_t min_prefill_chunk = 1;
#ifdef EXECUTORCH_BUILD_CUDA
  if (artifact_mode == MuseGlimmerArtifactMode::Autoregressive) {
    min_prefill_chunk = 5;
    if (auto get_result = meta_module->get(kMinPrefillChunk); get_result.ok()) {
      min_prefill_chunk = get_result->toScalar().to<int64_t>();
    }
    metadata[kMinPrefillChunk] = min_prefill_chunk;
  }
#endif

  int64_t dflash_block_size = 0;
  int64_t dflash_mask_token_id = 0;
  int64_t dflash_n_target_layers = 0;
  int64_t dflash_sliding_window = 0;
  int64_t dflash_min_target_prefill_chunk = 0;
  int64_t dflash_max_target_prefill_chunk = 0;
  int64_t dflash_min_draft_prefill_chunk = 0;
  int64_t dflash_max_draft_prefill_chunk = 0;
  int64_t vision_hidden_size = 0;
  int64_t max_vision_patches = 0;
  auto activation_dtype_result =
      get_required_string_metadata(meta_module.get(), kActivationDtype);
  ET_CHECK_OK_OR_RETURN_ERROR(activation_dtype_result.error());
  std::string activation_dtype = std::move(activation_dtype_result.get());
  ET_CHECK_OR_RETURN_ERROR(
      activation_dtype == "bfloat16" || activation_dtype == "float16" ||
          activation_dtype == "float32",
      InvalidProgram,
      "Unsupported Muse Glimmer activation dtype %s",
      activation_dtype.c_str());
  ET_CHECK_OR_RETURN_ERROR(
      (!has_vision && artifact_mode != MuseGlimmerArtifactMode::DFlash) ||
          activation_dtype != "float32",
      InvalidProgram,
      "Vision and DFlash require bfloat16 or float16 activations");
  if (artifact_mode == MuseGlimmerArtifactMode::DFlash) {
    auto block_size =
        get_required_int_metadata(meta_module.get(), kDFlashBlockSize);
    auto mask_token =
        get_required_int_metadata(meta_module.get(), kDFlashMaskTokenId);
    auto target_layers =
        get_required_int_metadata(meta_module.get(), kDFlashTargetLayers);
    auto sliding_window =
        get_required_int_metadata(meta_module.get(), kDFlashSlidingWindow);
    ET_CHECK_OK_OR_RETURN_ERROR(block_size.error());
    ET_CHECK_OK_OR_RETURN_ERROR(mask_token.error());
    ET_CHECK_OK_OR_RETURN_ERROR(target_layers.error());
    ET_CHECK_OK_OR_RETURN_ERROR(sliding_window.error());
    dflash_block_size = block_size.get();
    dflash_mask_token_id = mask_token.get();
    dflash_n_target_layers = target_layers.get();
    dflash_sliding_window = sliding_window.get();
#ifdef EXECUTORCH_BUILD_CUDA
    if (has_dflash_target_prefill) {
      auto min_target_prefill_chunk = get_required_int_metadata(
          meta_module.get(), kDFlashMinTargetPrefillChunk);
      auto max_target_prefill_chunk = get_required_int_metadata(
          meta_module.get(), kDFlashMaxTargetPrefillChunk);
      ET_CHECK_OK_OR_RETURN_ERROR(min_target_prefill_chunk.error());
      ET_CHECK_OK_OR_RETURN_ERROR(max_target_prefill_chunk.error());
      dflash_min_target_prefill_chunk = min_target_prefill_chunk.get();
      dflash_max_target_prefill_chunk = max_target_prefill_chunk.get();
      ET_CHECK_OR_RETURN_ERROR(
          dflash_min_target_prefill_chunk > kCudaDFlashHiddenRows &&
              dflash_max_target_prefill_chunk >=
                  dflash_min_target_prefill_chunk,
          InvalidProgram,
          "Invalid DFlash target prefill chunk interval [%" PRId64 ", %" PRId64
          "]",
          dflash_min_target_prefill_chunk,
          dflash_max_target_prefill_chunk);
    }
    const bool has_draft_prefill = method_names.count(kDraftPrefillMethod) != 0;
    if (has_draft_prefill) {
      auto min_draft_prefill_chunk = get_required_int_metadata(
          meta_module.get(), kDFlashMinDraftPrefillChunk);
      auto max_draft_prefill_chunk = get_required_int_metadata(
          meta_module.get(), kDFlashMaxDraftPrefillChunk);
      ET_CHECK_OK_OR_RETURN_ERROR(min_draft_prefill_chunk.error());
      ET_CHECK_OK_OR_RETURN_ERROR(max_draft_prefill_chunk.error());
      dflash_min_draft_prefill_chunk = min_draft_prefill_chunk.get();
      dflash_max_draft_prefill_chunk = max_draft_prefill_chunk.get();
      ET_CHECK_OR_RETURN_ERROR(
          dflash_min_draft_prefill_chunk > kCudaDFlashHiddenRows &&
              dflash_max_draft_prefill_chunk >= dflash_min_draft_prefill_chunk,
          InvalidProgram,
          "Invalid DFlash draft prefill chunk interval [%" PRId64 ", %" PRId64
          "]",
          dflash_min_draft_prefill_chunk,
          dflash_max_draft_prefill_chunk);
    }
#endif
    ET_CHECK_OR_RETURN_ERROR(
        dflash_block_size >= 2,
        InvalidProgram,
        "DFlash block size must be at least 2, got %" PRId64,
        dflash_block_size);
    ET_CHECK_OR_RETURN_ERROR(
        config.dflash_block_length == 0 ||
            (config.dflash_block_length >= 2 &&
             config.dflash_block_length <= dflash_block_size),
        InvalidArgument,
        "DFlash block length must be in [2, %" PRId64 "]",
        dflash_block_size);
    const int64_t block_length = config.dflash_block_length > 0
        ? config.dflash_block_length
        : dflash_block_size;
    ET_CHECK_OR_RETURN_ERROR(
        config.dflash_n_draft == 0 ||
            (config.dflash_n_draft > 0 && config.dflash_n_draft < block_length),
        InvalidArgument,
        "DFlash n_draft must be in [1, block_length - 1]");
#ifdef EXECUTORCH_BUILD_CUDA
    // A decode cycle retains up to n_draft + 1 hidden rows, which the next
    // draft call must feed into the fixed-size CUDA hidden input.
    const int64_t n_draft =
        config.dflash_n_draft > 0 ? config.dflash_n_draft : block_length - 1;
    ET_CHECK_OR_RETURN_ERROR(
        n_draft + 1 <= kCudaDFlashHiddenRows,
        InvalidArgument,
        "CUDA DFlash n_draft must be at most %" PRId64 ", got %" PRId64,
        kCudaDFlashHiddenRows - 1,
        n_draft);
#endif
    metadata[kDFlashBlockSize] = dflash_block_size;
    metadata[kDFlashMaskTokenId] = dflash_mask_token_id;
    metadata[kDFlashTargetLayers] = dflash_n_target_layers;
    metadata[kDFlashSlidingWindow] = dflash_sliding_window;
#ifdef EXECUTORCH_BUILD_CUDA
    metadata[kDFlashMinTargetPrefillChunk] = dflash_min_target_prefill_chunk;
    metadata[kDFlashMaxTargetPrefillChunk] = dflash_max_target_prefill_chunk;
    metadata[kDFlashMinDraftPrefillChunk] = dflash_min_draft_prefill_chunk;
    metadata[kDFlashMaxDraftPrefillChunk] = dflash_max_draft_prefill_chunk;
#endif
  }
  if (has_vision) {
    auto hidden_size =
        get_required_int_metadata(meta_module.get(), kVisionHiddenSize);
    auto patch_limit =
        get_required_int_metadata(meta_module.get(), kMaxVisionPatches);
    ET_CHECK_OK_OR_RETURN_ERROR(hidden_size.error());
    ET_CHECK_OK_OR_RETURN_ERROR(patch_limit.error());
    vision_hidden_size = hidden_size.get();
    max_vision_patches = patch_limit.get();
    ET_CHECK_OR_RETURN_ERROR(
        vision_hidden_size > 0,
        InvalidProgram,
        "Vision hidden size must be positive");
    ET_CHECK_OR_RETURN_ERROR(
        max_vision_patches > 0 &&
            max_vision_patches % kVisionDownsampleArea == 0,
        InvalidProgram,
        "Max vision patches must be positive and divisible by %" PRId64,
        kVisionDownsampleArea);
    metadata[kVisionHiddenSize] = vision_hidden_size;
    metadata[kMaxVisionPatches] = max_vision_patches;
  }

  std::unique_ptr<MuseGlimmerMutableStateContextOwner> mutable_state;
#ifdef EXECUTORCH_BUILD_CUDA
  if (config.enable_cuda_graph) {
    ET_LOG(
        Info,
        "MuseGlimmerEngine: CUDA graph requested; per-session rebinding "
        "disabled and serving capacity clamped to 1 session.");
  } else {
    auto candidate = std::make_unique<MuseGlimmerMutableStateContextOwner>();
    if (Error e = register_mutable_fqns(meta_module.get(), *candidate);
        e == Error::Ok) {
      mutable_state = std::move(candidate);
    } else {
      ET_LOG(
          Info,
          "MuseGlimmerEngine: mutable-buffer metadata unavailable or invalid; "
          "serving capacity clamped to 1 session.");
    }
  }
#elif defined(EXECUTORCH_BUILD_MLX)
  // Only enable the per-session mutable-buffer path when actually serving more
  // than one session. For a single session (the CLI runner) the rebind would
  // allocate a second copy of the KV-cache buffers on top of the program's
  // default buffers — doubling KV-cache memory and adding a one-time
  // session-buffer allocation during the first prefill — for no isolation
  // benefit. Leaving mutable_state null keeps the program's default buffers.
  if (config.max_sessions > 1) {
    mutable_state = std::make_unique<MuseGlimmerMutableStateContextOwner>();
  }
#endif

  // Pass whether a multi-session owner exists as the single source of truth for
  // skip_mutable_buffer_init, so the skip flag can never diverge from the
  // owner.
  const bool multi_session = mutable_state != nullptr;
  auto module_res = multi_session
      ? mutable_state->with_load_scope([&]() {
          return build_muse_glimmer_module(
              config, artifact_mode, has_vision, multi_session);
        })
      : build_muse_glimmer_module(
            config, artifact_mode, has_vision, multi_session);
  if (module_res.error() != Error::Ok) {
    return module_res.error();
  }
  std::unique_ptr<Module> shared_module = std::move(module_res.get());

  bool rebind_available = false;
  rebind_available = mutable_state != nullptr && mutable_state->available();
  if (rebind_available && mutable_state->validate_coverage() != Error::Ok) {
    ET_LOG(
        Error,
        "MuseGlimmerEngine: mutable-buffer coverage check failed; disabling "
        "multi-session (capacity clamped to 1).");
    rebind_available = false;
  }
  if (!rebind_available) {
    ET_LOG(
        Info,
        "MuseGlimmerEngine: per-session rebinding unavailable; serving capacity "
        "clamped to 1 session.");
  }

  TensorPtr decode_pos_table_dev;
#ifdef EXECUTORCH_BUILD_CUDA
  decode_pos_table_dev = build_decode_pos_table(metadata);
#endif

  ET_CHECK_OR_RETURN_ERROR(
      !has_vision || config.max_image_bytes > 0,
      InvalidArgument,
      "max_image_bytes must be positive");
  const auto vision_activation_dtype = activation_dtype == "bfloat16"
      ? executorch::aten::ScalarType::BFloat16
      : executorch::aten::ScalarType::Half;
  auto engine = std::unique_ptr<MuseGlimmerEngine>(new MuseGlimmerEngine(
      config,
      std::move(tokenizer),
      std::move(metadata),
      std::move(eos_ids),
      std::move(shared_module),
      artifact_mode,
      max_prefill_chunk,
      min_prefill_chunk,
      dflash_block_size,
      dflash_mask_token_id,
      dflash_n_target_layers,
      dflash_sliding_window,
      dflash_min_target_prefill_chunk,
      dflash_max_target_prefill_chunk,
      std::move(activation_dtype),
      std::move(decode_pos_table_dev),
      /*vision_runtime=*/nullptr,
      rebind_available,
      std::move(mutable_state)));
  if (has_vision) {
    MuseGlimmerVisionRuntimeConfig vision_config;
    vision_config.module = engine->shared_module_.get();
    vision_config.execution_mutex = &engine->exec_mutex_;
    vision_config.pos_embed_path = config.pos_embed_path.empty()
        ? default_pos_embed_path(config.model_path)
        : config.pos_embed_path;
    vision_config.activation_dtype = vision_activation_dtype;
    vision_config.expected_hidden_dim = vision_hidden_size;
    vision_config.max_image_tokens = max_vision_patches / kVisionDownsampleArea;
    vision_config.max_encoded_bytes = config.max_image_bytes;
    try {
      engine->vision_runtime_ =
          std::make_unique<MuseGlimmerVisionRuntime>(std::move(vision_config));
    } catch (const std::exception& exception) {
      ET_LOG(
          Error,
          "MuseGlimmerEngine: failed to initialize vision runtime: %s",
          exception.what());
      return Error::InvalidExternalData;
    }
  }
  return engine;
}

MuseGlimmerEngine::~MuseGlimmerEngine() = default;

Result<PreparedMuseGlimmerImage> MuseGlimmerEngine::prepare_image_from_file(
    const std::string& image_path) const {
  ET_CHECK_OR_RETURN_ERROR(
      vision_runtime_ != nullptr,
      NotSupported,
      "Muse Glimmer artifact does not support image input");
  return vision_runtime_->prepare_image_from_file(image_path);
}

Result<PreparedMuseGlimmerImage> MuseGlimmerEngine::prepare_image_from_bytes(
    ::executorch::runtime::Span<const uint8_t> encoded_image) const {
  ET_CHECK_OR_RETURN_ERROR(
      vision_runtime_ != nullptr,
      NotSupported,
      "Muse Glimmer artifact does not support image input");
  return vision_runtime_->prepare_image_from_bytes(encoded_image);
}

Result<std::unique_ptr<LLMSession>> MuseGlimmerEngine::create_session() {
  const int cap =
      serving_capacity().max_physical_sessions_without_weight_duplication;
  {
    std::lock_guard<std::mutex> g(exec_mutex_);
    if (live_sessions_.load() >= cap) {
      ET_LOG(
          Error,
          "MuseGlimmerEngine: at session capacity (%d); refusing create_session",
          cap);
      return Error::InvalidState;
    }
    live_sessions_.fetch_add(1);
  }

  int token = -1;
  if (rebind_available_) {
    auto t = mutable_state_->create_session();
    if (t.error() != Error::Ok) {
      live_sessions_.fetch_sub(1);
      return t.error();
    }
    token = t.get();
  }

  if (artifact_mode_ == MuseGlimmerArtifactMode::DFlash) {
    const int64_t block_length = config_.dflash_block_length > 0
        ? config_.dflash_block_length
        : dflash_block_size_;
    const int64_t n_draft =
        config_.dflash_n_draft > 0 ? config_.dflash_n_draft : block_length - 1;
    DFlashSessionConfig session_config;
    session_config.module = shared_module_.get();
    session_config.exec_mutex = &exec_mutex_;
    session_config.live_sessions = &live_sessions_;
    session_config.tokenizer = tokenizer_.get();
    session_config.metadata = metadata_;
    session_config.eos_ids = eos_ids_;
    session_config.mutable_state =
        rebind_available_ ? mutable_state_.get() : nullptr;
    session_config.session_token = token;
    session_config.max_prefill_chunk = max_prefill_chunk_;
    session_config.block_size = dflash_block_size_;
    session_config.block_length = block_length;
    session_config.n_draft = n_draft;
    session_config.mask_token_id = dflash_mask_token_id_;
    session_config.n_target_layers = dflash_n_target_layers_;
    session_config.draft_sliding_window = dflash_sliding_window_;
    session_config.min_target_prefill_chunk = dflash_min_target_prefill_chunk_;
    session_config.max_target_prefill_chunk = dflash_max_target_prefill_chunk_;
    session_config.min_draft_prefill_chunk =
        metadata_.count(kDFlashMinDraftPrefillChunk) != 0
        ? metadata_.at(kDFlashMinDraftPrefillChunk)
        : 0;
    session_config.max_draft_prefill_chunk =
        metadata_.count(kDFlashMaxDraftPrefillChunk) != 0
        ? metadata_.at(kDFlashMaxDraftPrefillChunk)
        : 0;
    session_config.has_draft_prefill =
        session_config.min_draft_prefill_chunk > 0;
    session_config.activation_dtype = activation_dtype_;
    session_config.draft_argmax = config_.dflash_draft_argmax;
    session_config.ignore_eos = config_.dflash_ignore_eos;
    session_config.timing = config_.dflash_timing;
    auto session = create_dflash_session(std::move(session_config));
    if (session.error() != Error::Ok) {
      if (rebind_available_) {
        mutable_state_->destroy_session(token);
      }
      live_sessions_.fetch_sub(1);
    }
    return session;
  }

  return std::unique_ptr<LLMSession>(new MuseGlimmerSession(
      shared_module_.get(),
      &exec_mutex_,
      &live_sessions_,
      tokenizer_.get(),
      metadata_,
      eos_ids_,
      max_prefill_chunk_,
      min_prefill_chunk_,
      decode_pos_table_dev_,
      rebind_available_ ? mutable_state_.get() : nullptr,
      token));
}

LLMServingCapacity MuseGlimmerEngine::serving_capacity() const {
  LLMServingCapacity cap;
  if (rebind_available_) {
    cap.max_physical_sessions_without_weight_duplication =
        config_.max_sessions > 1 ? config_.max_sessions : 1;
    cap.estimated_bytes_per_session = mutable_state_->bytes_per_session();
  }
  return cap;
}

} // namespace executorch::extension::llm
