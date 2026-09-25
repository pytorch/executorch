/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Muse Glimmer runner for the ExecuTorch CUDA and MLX backends.
// Text and vision generation use MuseGlimmerEngine. Vision inputs are
// host-preprocessed and spliced into the prompt at <|patch|> positions.

#include <gflags/gflags.h>

#include <executorch/examples/models/muse-glimmer/runtime/engine/muse_glimmer_engine.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/platform/log.h>
#include <pytorch/tokenizers/hf_tokenizer.h>

#include <executorch/examples/models/muse-glimmer/runtime/engine/sampling.h>

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/types.h>
extern "C" void et_pal_emit_log_message(
    ET_UNUSED et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    ET_UNUSED const char* function,
    size_t line,
    const char* message,
    ET_UNUSED size_t length) {
  if (level == 'D' || level == 'I') {
    return;
  }
  fprintf(stderr, "%c [%s:%zu] %s\n", (char)level, filename, line, message);
}

#ifdef EXECUTORCH_BUILD_CUDA
#include <cuda_runtime.h>
#endif

// The vision runtime translation unit owns the stb implementations; this file
// uses declarations for its direct vision path.
#include <stb_image.h>

#include <executorch/examples/models/muse-glimmer/vision/preprocess.h>

#ifdef EXECUTORCH_BUILD_MLX
// MLX build: option keys for MLXBackend::init (clear_cache_interval).
#include <executorch/backends/mlx/runtime/backend_options.h>
#endif

DEFINE_string(model_path, "", "Model .pte file path.");
DEFINE_string(data_path, "", "Data file (.ptd), if the export produced one.");
DEFINE_string(
    tokenizer_path,
    "",
    "Hugging Face tokenizer.json path. Must encode the Harmony control "
    "tokens; a raw BPE vocabulary will not.");
DEFINE_string(prompt, "The meaning of life is", "Prompt text.");
DEFINE_string(
    prompt_file,
    "",
    "Path to file containing prompt text (overrides --prompt).");
DEFINE_string(
    prompt_tokens_file,
    "",
    "int64 LE pre-tokenized prompt IDs; bypasses encode in the timed path.");
DEFINE_bool(
    tokens_have_bos,
    false,
    "If true the file already includes BOS; else BOS is prepended.");
DEFINE_double(temperature, 0.8, "Sampling temperature (0 = near-greedy).");
DEFINE_double(
    top_p,
    1.0,
    "Top-p (nucleus) host sampling for MLX/CPU. 1.0 = disabled. Not supported "
    "on CUDA, which samples on-device.");
DEFINE_int32(
    top_k,
    0,
    "Top-k host sampling for MLX/CPU. 0 = disabled. Not supported on CUDA, "
    "which samples on-device.");
DEFINE_int64(
    seed,
    0,
    "RNG seed for host sampling (MLX/CPU) and the engine path. <=0 = "
    "nondeterministic. Applies to both the direct-Module and engine generation "
    "paths so a given seed reproduces the same sequence.");
DEFINE_int32(max_new_tokens, 128, "Maximum tokens to generate.");
DEFINE_bool(
    ignore_eos,
    false,
    "Ignore EOS and emit exactly max_new_tokens tokens for benchmarking.");
DEFINE_string(
    generated_tokens_file,
    "",
    "Optional path to write logical continuation token IDs as native int64 "
    "values.");
DEFINE_int32(bos_id, 200000, "BOS token id.");
DEFINE_int32(eos_id, 200001, "EOS token id.");
DEFINE_bool(
    cuda_graph,
    true,
    "Enable CUDA graph capture for the decode method. CUDA only; default on.");
DEFINE_string(
    nll_tokens_file,
    "",
    "Path to binary file of int64 token IDs for NLL validation. "
    "Runs chunked prefill, dumps logits to --nll_output_file, then exits.");
DEFINE_string(
    nll_output_file,
    "",
    "Path to write binary logits (float32, [T, V]) for NLL validation.");
DEFINE_bool(
    nll_use_prefill,
    false,
    "Use prefill method (multi-token) instead of decode (single-token) for NLL.");
DEFINE_string(
    nll_image_manifest,
    "",
    "Path to a newline-delimited, ordered list of image paths for multimodal "
    "NLL. When set together with --nll_tokens_file (CUDA vision --logits .pte), "
    "the runner encodes each image with vision_encoder, splices the resulting "
    "soft-token features into the <|patch|> runs of the token sequence in "
    "order, prefills on embeddings, and writes (T-1) float32 per-position "
    "next-token log-probs to --nll_output_file (position i scores tokens[i+1]).");
DEFINE_int32(
    max_prefill_chunk,
    0,
    "Override the prefill chunk size. Must be <= the model's exported "
    "get_max_prefill_chunk (the baked dynamic seq_len bound); values above it "
    "are ignored. 0 = use the model's value. Smaller chunks trade prefill "
    "throughput for lower peak memory.");
DEFINE_string(
    method,
    "forward",
    "Forward method to invoke, for whichever path runs. Text single-method "
    "(token input): default 'forward'; use 'target_forward' to run a "
    "DFlash-exported target autoregressively (non-draft). MLX vision (embeds "
    "input, after splicing image soft-tokens): the default 'forward' maps to "
    "'forward_from_embeddings' (solo vision export); use "
    "'target_forward_from_embeddings' for a DFlash-vision .pte. Only "
    "output[0] logits are used; any extra hidden output is ignored.");
DEFINE_int32(
    vision_max_prefill_chunk,
    3072,
    "Maximum prefill chunk for vision prompts. 0 disables this "
    "vision-specific cap.");
DEFINE_string(
    image_path,
    "",
    "Optional path to an image (JPEG/PNG). When set (CUDA or MLX build with a "
    "vision-enabled .pte), the runner runs the vision_encoder, splices the "
    "image soft-tokens at the prompt's single <img> marker, and prefills on "
    "embeds.");
DEFINE_string(
    pos_embed_path,
    "",
    "Path to pos_embed.bin (vision positional-embedding table) written next to "
    "model.pte by export.py. Defaults to <model_dir>/pos_embed.bin.");

namespace llm = ::executorch::extension::llm;
using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;

using SizesType = executorch::aten::SizesType;

// Whether the runner invokes a single exported method (--method) instead of the
// standard prefill/decode pair. Derived from the loaded .pte's exported methods
// (see main), not the build backend: a model exporting both `prefill` and
// `decode` uses the two-method path; a single-`forward` export or a DFlash
// target-only method selected via --method uses single-method execution.
// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
static bool kSingleMethod = false;

// Method to invoke for a chunk of `chunk_len` tokens.
static const char* method_for(int64_t chunk_len) {
  if (kSingleMethod) {
    return FLAGS_method.c_str();
  }
  return (chunk_len == 1) ? "decode" : "prefill";
}

// Read a sampled token ID from a scalar float output (on-device sampling).
// Used by the CUDA sampling export (sample=True), whose methods return one
// Gumbel-max token id as a float scalar instead of full logits.
[[maybe_unused]] static uint64_t read_token(
    const executorch::aten::Tensor& output) {
  const void* ptr = output.const_data_ptr();
  float val = 0.0f;
#ifdef EXECUTORCH_BUILD_CUDA
  cudaPointerAttributes attrs{};
  bool on_device = cudaPointerGetAttributes(&attrs, ptr) == cudaSuccess &&
      attrs.type == cudaMemoryTypeDevice;
  if (on_device) {
    cudaMemcpy(&val, ptr, sizeof(float), cudaMemcpyDeviceToHost);
  } else {
    memcpy(&val, ptr, sizeof(float));
  }
#else
  memcpy(&val, ptr, sizeof(float));
#endif
  return static_cast<uint64_t>(llrintf(val));
}

// Copy CUDA or MLX tensor storage to the host for embedding splices.
static Error
copy_to_host(const executorch::aten::Tensor& src, void* dst, size_t num_bytes) {
  if (num_bytes > src.nbytes()) {
    ET_LOG(
        Error,
        "copy_to_host requested %zu bytes from a %zu-byte tensor",
        num_bytes,
        src.nbytes());
    return Error::InvalidArgument;
  }
  const void* src_ptr = src.const_data_ptr();
#ifdef EXECUTORCH_BUILD_CUDA
  cudaPointerAttributes attrs{};
  bool on_device = cudaPointerGetAttributes(&attrs, src_ptr) == cudaSuccess &&
      attrs.type == cudaMemoryTypeDevice;
  if (on_device) {
    auto err = cudaMemcpy(dst, src_ptr, num_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      ET_LOG(Error, "copy_to_host D2H failed: %s", cudaGetErrorString(err));
      return Error::Internal;
    }
  } else
#endif
  {
    memcpy(dst, src_ptr, num_bytes);
  }
  return Error::Ok;
}

// Muse Glimmer image special-token ids (from the tokenizer /
// MuseGlimmerConfig).

// Preprocess one image file, run the exported vision_encoder, and copy the
// resulting soft-token embeddings ([num_soft_tokens, hidden] bf16) to host.
// Used by the multimodal NLL path to score the image/perception reference.
static Error encode_image_nll(
    Module& module,
    const std::string& image_path,
    const std::vector<float>& pos_table,
    std::vector<uint16_t>& out_embeds,
    int64_t& out_num_soft_tokens,
    int64_t& out_hidden) {
  namespace gv = ::executorch::examples::muse_glimmer_vision;
  int img_w = 0, img_h = 0, img_c = 0;
  unsigned char* img_data =
      stbi_load(image_path.c_str(), &img_w, &img_h, &img_c, 3);
  if (img_data == nullptr) {
    ET_LOG(Error, "Failed to load image: %s", image_path.c_str());
    return Error::Internal;
  }
  gv::VisionInputs vis;
  try {
    vis = gv::preprocess_image(img_data, img_w, img_h, pos_table);
  } catch (const std::exception& e) {
    ET_LOG(
        Error,
        "preprocess_image failed for %s: %s",
        image_path.c_str(),
        e.what());
    stbi_image_free(img_data);
    return Error::Internal;
  }
  stbi_image_free(img_data);

  std::vector<EValue> ve_inputs;
  ve_inputs.push_back(EValue(vis.patches));
  ve_inputs.push_back(EValue(vis.pos_emb));
  ve_inputs.push_back(EValue(vis.cos_2d));
  ve_inputs.push_back(EValue(vis.sin_2d));
  ve_inputs.push_back(EValue(vis.sparse_perm));
  ve_inputs.push_back(EValue(vis.inv_perm));
  ve_inputs.push_back(EValue(vis.global_mask));
  ve_inputs.push_back(EValue(vis.sparse_mask));
  ve_inputs.push_back(EValue(vis.pixel_perm));

  auto ve_result = module.execute("vision_encoder", ve_inputs);
  if (ve_result.error() != Error::Ok) {
    ET_LOG(Error, "vision_encoder failed for %s", image_path.c_str());
    return Error::Internal;
  }
  const auto& image_embeds = ve_result.get()[0].toTensor();
  if (image_embeds.dim() != 3 || image_embeds.size(0) != 1) {
    ET_LOG(Error, "vision_encoder output must be [1, N, hidden]");
    return Error::Internal;
  }
  out_num_soft_tokens = image_embeds.size(1);
  out_hidden = image_embeds.size(2);
  out_embeds.assign(static_cast<size_t>(out_num_soft_tokens * out_hidden), 0);
  return copy_to_host(
      image_embeds,
      out_embeds.data(),
      static_cast<size_t>(out_num_soft_tokens * out_hidden) * sizeof(uint16_t));
}

// Apply the optional --max_prefill_chunk override to the model's exported chunk
// size. The override may only lower the chunk, never exceed the exported bound
// (which is baked into the .pte as the dynamic seq_len limit).
static int64_t apply_prefill_chunk_override(
    int64_t model_chunk,
    bool is_vision = false) {
  if (is_vision && FLAGS_vision_max_prefill_chunk > 0) {
    model_chunk =
        std::min<int64_t>(model_chunk, FLAGS_vision_max_prefill_chunk);
  }
  if (FLAGS_max_prefill_chunk > 0 && FLAGS_max_prefill_chunk < model_chunk) {
    return FLAGS_max_prefill_chunk;
  }
  return model_chunk;
}

#ifdef EXECUTORCH_BUILD_MLX
static Error read_activation_dtype(
    Module& module,
    executorch::aten::ScalarType& dtype) {
  auto result = module.get("get_activation_dtype");
  if (!result.ok() || !result->isString()) {
    ET_LOG(Error, "Model .pte is missing get_activation_dtype metadata");
    return Error::InvalidProgram;
  }
  const auto tag = result->toString();
  if (tag == "bfloat16") {
    dtype = executorch::aten::ScalarType::BFloat16;
  } else if (tag == "float16") {
    dtype = executorch::aten::ScalarType::Half;
  } else {
    ET_LOG(
        Error,
        "Unsupported activation dtype '%.*s' in .pte metadata",
        static_cast<int>(tag.size()),
        tag.data());
    return Error::InvalidProgram;
  }
  return Error::Ok;
}
#endif

// Host-side sampling over full logits. Used by the --logits (NLL) export mode
// and the MLX `forward` export, which return full logits instead of a token.
[[maybe_unused]] static uint64_t sample_from_logits(
    const executorch::aten::Tensor& logits,
    int64_t seq_pos,
    double temperature) {
  // logits: (B, T, V) or (B, 1, V) float32. Sampling math is shared with the
  // DFlash/engine path via sampling.h (muse_glimmer::). --top_p/--top_k are
  // honored here (MLX/CPU); main() rejects them on CUDA, which samples
  // on-device.
  const int64_t vocab_size = logits.size(logits.dim() - 1);
  const int64_t offset = seq_pos * vocab_size;

  // Copy logits for the target position to host (source may be device on CUDA).
  std::vector<float> host_logits(vocab_size);
  const void* src = static_cast<const char*>(logits.const_data_ptr()) +
      offset * sizeof(float);

#ifdef EXECUTORCH_BUILD_CUDA
  cudaPointerAttributes attrs{};
  bool on_device = cudaPointerGetAttributes(&attrs, src) == cudaSuccess &&
      attrs.type == cudaMemoryTypeDevice;
  if (on_device) {
    cudaMemcpy(
        host_logits.data(),
        src,
        vocab_size * sizeof(float),
        cudaMemcpyDeviceToHost);
  } else {
    memcpy(host_logits.data(), src, vocab_size * sizeof(float));
  }
#else
  memcpy(host_logits.data(), src, vocab_size * sizeof(float));
#endif

  if (temperature <= 0.0) {
    return muse_glimmer::argmax_index(host_logits.data(), vocab_size);
  }

  static std::mt19937 rng(
      FLAGS_seed > 0 ? static_cast<std::mt19937::result_type>(FLAGS_seed)
                     : std::random_device{}());
  static muse_glimmer::SamplingWorkspace workspace;
  muse_glimmer::fill_sampling_probabilities(
      host_logits.data(),
      vocab_size,
      temperature,
      FLAGS_top_k,
      FLAGS_top_p,
      workspace);
  return muse_glimmer::categorical_sample(
      rng, workspace.probabilities.data(), vocab_size);
}

// Unified autoregressive generation through the shared
// MuseGlimmerEngine/LLMSession stack used by muse_glimmer_worker and
// dflash_runner.
static int run_engine_generation(llm::Stats& stats) {
  llm::MuseGlimmerConfig config;
  config.model_path = FLAGS_model_path;
  config.data_path = FLAGS_data_path;
  config.tokenizer_path = FLAGS_tokenizer_path;
  config.pos_embed_path = FLAGS_pos_embed_path;
  config.max_sessions = 1;
  config.eos_id = FLAGS_eos_id;
  config.enable_cuda_graph = FLAGS_cuda_graph;
  config.artifact_mode = llm::MuseGlimmerArtifactMode::Auto;

  auto engine_result = llm::MuseGlimmerEngine::create(config);
  if (engine_result.error() != Error::Ok) {
    ET_LOG(Error, "Failed to create MuseGlimmerEngine");
    return 1;
  }
  auto engine = std::move(engine_result.get());
  stats.model_load_end_ms = llm::time_in_ms();

#ifdef EXECUTORCH_BUILD_CUDA
  {
    size_t gpu_free_bytes = 0, gpu_total_bytes = 0;
    cudaMemGetInfo(&gpu_free_bytes, &gpu_total_bytes);
    stats.gpu_free_after_load_bytes = gpu_free_bytes;
  }
#endif

  auto session_result = engine->create_session();
  if (session_result.error() != Error::Ok) {
    ET_LOG(Error, "Failed to create LLMSession");
    return 1;
  }
  auto session = std::move(session_result.get());

  const bool has_image = !FLAGS_image_path.empty();
  int64_t num_soft_tokens = 0;
  if (has_image) {
    if (!FLAGS_prompt_tokens_file.empty()) {
      ET_LOG(Error, "--image_path is not supported with --prompt_tokens_file");
      return 1;
    }
    if (!engine->has_vision()) {
      ET_LOG(Error, "--image_path requires an artifact with vision_encoder");
      return 1;
    }
    auto prepared = engine->prepare_image_from_file(FLAGS_image_path);
    if (prepared.error() != Error::Ok) {
      ET_LOG(Error, "Failed to prepare image");
      return 1;
    }
    num_soft_tokens = prepared->num_soft_tokens;
    auto* multimodal = llm::as_muse_glimmer_multimodal_session(session.get());
    if (multimodal == nullptr) {
      ET_LOG(Error, "Muse Glimmer session does not support image input");
      return 1;
    }
    multimodal->stage_image_for_next_prefill(std::move(prepared.get()));
  }

  // Build prompt tokens, preserving the runner's input options (mirrors the
  // non-image prompt handling in main()).
  std::vector<uint64_t> prompt_tokens;
  if (!FLAGS_prompt_tokens_file.empty()) {
    std::ifstream tf(FLAGS_prompt_tokens_file, std::ios::binary);
    if (!tf.is_open()) {
      ET_LOG(
          Error,
          "Cannot open --prompt_tokens_file: %s",
          FLAGS_prompt_tokens_file.c_str());
      return 1;
    }
    tf.seekg(0, std::ios::end);
    const std::streamoff nbytes = tf.tellg();
    tf.seekg(0, std::ios::beg);
    if (nbytes < 0 ||
        nbytes % static_cast<std::streamoff>(sizeof(int64_t)) != 0) {
      ET_LOG(
          Error,
          "--prompt_tokens_file size (%lld bytes) is not a positive multiple "
          "of %zu",
          static_cast<long long>(nbytes),
          sizeof(int64_t));
      return 1;
    }
    const int64_t num_file_tokens =
        static_cast<int64_t>(nbytes) / static_cast<int64_t>(sizeof(int64_t));
    std::vector<int64_t> file_tokens(num_file_tokens);
    tf.read(reinterpret_cast<char*>(file_tokens.data()), nbytes);
    if (tf.gcount() != nbytes) {
      ET_LOG(Error, "Failed to read all of --prompt_tokens_file");
      return 1;
    }
    tf.close();
    if (!FLAGS_tokens_have_bos) {
      prompt_tokens.push_back(static_cast<uint64_t>(FLAGS_bos_id));
    }
    prompt_tokens.insert(
        prompt_tokens.end(), file_tokens.begin(), file_tokens.end());
  } else {
    std::string prompt_text = FLAGS_prompt;
    if (!FLAGS_prompt_file.empty()) {
      std::ifstream f(FLAGS_prompt_file);
      if (!f.is_open()) {
        ET_LOG(
            Error, "Failed to open prompt file: %s", FLAGS_prompt_file.c_str());
        return 1;
      }
      prompt_text = std::string(
          (std::istreambuf_iterator<char>(f)),
          std::istreambuf_iterator<char>());
    }
    auto tokenized = llm::tokenize_muse_glimmer_prompt(
        *engine->tokenizer(),
        prompt_text,
        static_cast<uint64_t>(FLAGS_bos_id),
        has_image ? std::optional<int64_t>(num_soft_tokens) : std::nullopt);
    if (!tokenized.ok()) {
      ET_LOG(
          Error,
          "Failed to tokenize prompt; --image_path requires exactly one "
          "<img> marker");
      return 1;
    }
    prompt_tokens = std::move(*tokenized);
  }

  const int64_t num_prompt_tokens = static_cast<int64_t>(prompt_tokens.size());
  printf("Prompt tokens: %" PRId64 "\n", num_prompt_tokens);
  stats.num_prompt_tokens = num_prompt_tokens;

  llm::SamplingConfig sampling;
  sampling.temperature = static_cast<float>(FLAGS_temperature);
  sampling.top_p = static_cast<float>(FLAGS_top_p);
  sampling.top_k = FLAGS_top_k;
  sampling.seed = FLAGS_seed > 0 ? static_cast<uint64_t>(FLAGS_seed) : 0;

  // Tokenization is excluded from the prefill timer to match main()'s timing.
  stats.inference_start_ms = llm::time_in_ms();
  if (session->prefill_tokens(prompt_tokens, &sampling) != Error::Ok) {
    ET_LOG(Error, "prefill failed");
    return 1;
  }
  stats.prompt_eval_end_ms = llm::time_in_ms();
  const double prefill_ms =
      static_cast<double>(stats.prompt_eval_end_ms - stats.inference_start_ms);
  printf(
      "Prefill: %" PRId64 " tokens in %.1f ms (%.1f tok/s)\n",
      num_prompt_tokens,
      prefill_ms,
      prefill_ms > 0.0 ? num_prompt_tokens * 1000.0 / prefill_ms : 0.0);

  int64_t num_generated = 0;
  for (int32_t step = 0; step < FLAGS_max_new_tokens; ++step) {
    auto decode_result = session->decode_one(sampling);
    if (decode_result.error() != Error::Ok) {
      ET_LOG(Error, "Decode step %d failed", step);
      return 1;
    }
    const auto& decoded = decode_result.get();
    if (decoded.is_terminal) {
      break;
    }
    if (num_generated == 0) {
      stats.first_token_ms = llm::time_in_ms();
    }
    if (!decoded.text_piece.empty()) {
      fwrite(decoded.text_piece.data(), 1, decoded.text_piece.size(), stdout);
      fflush(stdout);
    }
    ++num_generated;
  }

  stats.inference_end_ms = llm::time_in_ms();
  stats.num_generated_tokens = num_generated;
  printf("\n");
  const double decode_ms =
      static_cast<double>(stats.inference_end_ms - stats.prompt_eval_end_ms);
  printf(
      "Decode: %" PRId64 " tokens in %.1f ms (%.1f tok/s)\n",
      num_generated,
      decode_ms,
      decode_ms > 0.0 ? num_generated * 1000.0 / decode_ms : 0.0);

#ifdef EXECUTORCH_BUILD_CUDA
  {
    size_t gpu_free_bytes = 0, gpu_total_bytes = 0;
    cudaMemGetInfo(&gpu_free_bytes, &gpu_total_bytes);
    stats.gpu_free_after_generate_bytes = gpu_free_bytes;
    stats.gpu_peak_usage_mb =
        (stats.gpu_total_bytes - gpu_free_bytes) / 1024.0 / 1024.0;
  }
#endif

  llm::print_report(stats);
  return 0;
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_model_path.empty()) {
    ET_LOG(Error, "Must specify --model_path");
    return 1;
  }
  if (FLAGS_tokenizer_path.empty()) {
    ET_LOG(Error, "Must specify --tokenizer_path");
    return 1;
  }

#ifdef EXECUTORCH_BUILD_CUDA
  if (FLAGS_top_p != 1.0 || FLAGS_top_k != 0) {
    ET_LOG(
        Error,
        "--top_p/--top_k are not supported on CUDA (sampling runs on-device); "
        "omit them to use the on-device sampler.");
    return 1;
  }
#endif

  llm::Stats stats;

#ifdef EXECUTORCH_BUILD_CUDA
  size_t gpu_free_bytes = 0, gpu_total_bytes = 0;
  cudaMemGetInfo(&gpu_free_bytes, &gpu_total_bytes);
  stats.gpu_total_bytes = gpu_total_bytes;
  stats.gpu_free_before_load_bytes = gpu_free_bytes;
#endif

  stats.model_load_start_ms = llm::time_in_ms();

  // CUDA uses backend cross-method weight sharing and file loading; its tensors
  // cannot use core shared memory arenas. MLX shares arenas for KV state and
  // mmaps model weights.
  std::vector<std::string> data_files;
  if (!FLAGS_data_path.empty()) {
    data_files.push_back(FLAGS_data_path);
  }
#ifdef EXECUTORCH_BUILD_CUDA
  constexpr auto kLoadMode = Module::LoadMode::File;
  constexpr bool kShareMemoryArenas = false;
#else
  constexpr auto kLoadMode = Module::LoadMode::MmapUseMlockIgnoreErrors;
  constexpr bool kShareMemoryArenas = true;
#endif
  auto module = std::make_unique<Module>(
      FLAGS_model_path,
      data_files,
      kLoadMode,
      /*event_tracer=*/nullptr,
      /*memory_allocator=*/nullptr,
      /*temp_allocator=*/nullptr,
      /*share_memory_arenas=*/kShareMemoryArenas);

  // Derive single-method mode from the artifact's unified Solo contract.
  {
    const auto method_names = module->method_names();
    bool has_solo_contract = method_names.ok() &&
        method_names->count("embed_text") != 0 &&
        method_names->count("forward_from_embeddings") != 0;
#ifndef EXECUTORCH_BUILD_MLX
    has_solo_contract =
        has_solo_contract && method_names->count("decode_from_embedding") != 0;
#endif
    kSingleMethod = !has_solo_contract;
  }

  // Generation uses the same engine/session path for CUDA, MLX, text, and
  // vision. NLL and explicit single-method benches use the direct path below.
  {
    const auto engine_methods = module->method_names();
    bool has_solo_contract = engine_methods.ok() &&
        engine_methods->count("embed_text") != 0 &&
        engine_methods->count("forward_from_embeddings") != 0;
#ifndef EXECUTORCH_BUILD_MLX
    has_solo_contract = has_solo_contract &&
        engine_methods->count("decode_from_embedding") != 0;
#endif
    const bool is_dflash =
        engine_methods.ok() && engine_methods->count("draft_forward") != 0;
    const bool generation =
        FLAGS_nll_tokens_file.empty() && FLAGS_method == "forward";
    if (generation && has_solo_contract && !is_dflash) {
      // Release the probe module before MuseGlimmerEngine reloads the .pte,
      // which builds its own tokenizer. The direct path below loads its own.
      module.reset();
      return run_engine_generation(stats);
    }
  }

  // Same tokenizer the engine path builds. This used to be a Tiktoken whose
  // only real special tokens were <|begin_of_text|> and <|end_of_text|>, so
  // <|start|>, <|message|> and <|eot|> byte-pair encoded into ordinary text and
  // every prompt measured here was longer than the one generation runs.
  auto tokenizer = std::make_unique<::tokenizers::HFTokenizer>();
  if (tokenizer->load(FLAGS_tokenizer_path) != tokenizers::Error::Ok) {
    ET_LOG(
        Error,
        "Failed to load tokenizer from %s",
        FLAGS_tokenizer_path.c_str());
    return 1;
  }

  // Get metadata
  auto metadata_result = llm::get_llm_metadata(tokenizer.get(), module.get());
  if (metadata_result.error() != Error::Ok) {
    ET_LOG(Error, "Failed to read model metadata");
    return 1;
  }

#ifdef EXECUTORCH_BUILD_CUDA
  if (FLAGS_cuda_graph) {
    executorch::runtime::BackendOptions<2> cuda_opts;
    cuda_opts.set_option(
        "enable_cuda_graph_for_method", "decode_from_embedding");
    executorch::runtime::set_option("CudaBackend", cuda_opts.view());
    printf("CUDA graph enabled for decode_from_embedding method\n");
  }

  // Cross-method per-FQN weight sharing keeps both decoder methods on one set
  // of weights and KV-cache buffers.
  {
    executorch::runtime::BackendOptions<1> backend_options;
    auto set_err =
        backend_options.set_option("weight_sharing_across_methods", true);
    if (set_err != Error::Ok) {
      ET_LOG(
          Error,
          "Failed to construct weight_sharing_across_methods option: %d",
          static_cast<int>(set_err));
      return 1;
    }
    auto opt_err =
        executorch::runtime::set_option("CudaBackend", backend_options.view());
    if (opt_err != Error::Ok) {
      ET_LOG(
          Error,
          "Failed to enable weight_sharing_across_methods: %d",
          static_cast<int>(opt_err));
      return 1;
    }
  }
#else
  if (FLAGS_cuda_graph) {
    ET_LOG(Info, "--cuda_graph ignored on non-CUDA build");
  }
#endif

  printf("Loading methods...\n");
  const bool has_image = !FLAGS_image_path.empty();
#ifdef EXECUTORCH_BUILD_CUDA
  // On CUDA, kSingleMethod means a token single-method (--method) bench, which
  // cannot splice an image. On MLX the vision path is selected by the
  // artifact's vision_encoder (mlx_multimodal), and has_image is validated in
  // that loader below, so this guard is CUDA-only.
  if (has_image && kSingleMethod) {
    ET_LOG(Error, "--image_path is not supported with --method");
    return 1;
  }
#endif
#ifdef EXECUTORCH_BUILD_CUDA
  // Two CUDA export flavors:
  //  * --logits (NLL) .pte: token-input `prefill`/`decode` returning logits.
  //  * sampling .pte (default): `embed_text` + embeds-input `prefill` +
  //    token-input `decode` (+ optional `vision_encoder`), returning a sampled
  //    token id. Multimodal NLL reuses its embeds-input short forward because
  //    that method returns logits for every position.
  const bool nll_mode = !FLAGS_nll_tokens_file.empty();
  const bool nll_vision = nll_mode && !FLAGS_nll_image_manifest.empty();
  std::string nll_vision_forward_method = "prefill";
  bool nll_vision_uses_short_forward = false;
  if (nll_vision) {
    const auto method_names = module->method_names();
    if (method_names.ok() &&
        method_names->count("target_forward_from_embeddings") != 0) {
      // DFlash long prefill intentionally computes lm_head for only the last
      // position. NLL needs one logits row per input position, which the short
      // verifier method already returns.
      nll_vision_forward_method = "target_forward_from_embeddings";
      nll_vision_uses_short_forward = true;
    }
  }
  // embed_text is needed for generation (splice image/text embeds) and for the
  // multimodal NLL path; the text-only NLL path uses token-input
  // prefill/decode.
  if ((!nll_mode && !kSingleMethod) || nll_vision) {
    if (module->load_method("embed_text") != Error::Ok) {
      ET_LOG(Error, "Failed to load embed_text method");
      return 1;
    }
  }
  if (nll_vision) {
    if (module->load_method(nll_vision_forward_method.c_str()) != Error::Ok) {
      ET_LOG(
          Error,
          "Failed to load multimodal NLL forward method %s",
          nll_vision_forward_method.c_str());
      return 1;
    }
  } else if (kSingleMethod) {
    if (module->load_method(FLAGS_method.c_str()) != Error::Ok) {
      ET_LOG(Error, "Failed to load %s method", FLAGS_method.c_str());
      return 1;
    }
  } else {
    if (module->load_method("prefill") != Error::Ok) {
      ET_LOG(Error, "Failed to load prefill method");
      return 1;
    }
    // The multimodal NLL path scores whole docs via embeds-input `prefill`
    // only; it never samples, so skip loading `decode` (saves its planned
    // memory, which matters for large vision models near single-GPU capacity).
    if (!nll_vision) {
      if (module->load_method("decode") != Error::Ok) {
        ET_LOG(Error, "Failed to load decode method");
        return 1;
      }
    }
  }
  if (nll_mode && has_image && !nll_vision) {
    ET_LOG(
        Error,
        "--image_path is not supported in NLL mode; pass an ordered image "
        "list via --nll_image_manifest for multimodal NLL.");
    return 1;
  }
  if (has_image && !FLAGS_prompt_tokens_file.empty()) {
    ET_LOG(Error, "--image_path is not supported with --prompt_tokens_file");
    return 1;
  }
  if (has_image || nll_vision) {
    if (module->load_method("vision_encoder") != Error::Ok) {
      ET_LOG(
          Error,
          "Failed to load vision_encoder — was the model exported with "
          "--mmproj?");
      return 1;
    }
  }
#elif defined(EXECUTORCH_BUILD_MLX)
  // MLX supports both a text-only `forward` .pte and a
  // vision .pte (embed_text + embeds-input `forward_from_embeddings` +
  // `vision_encoder`), detected by the method set. That embeds-forward is the
  // only KV-writing method; decode reuses it. Host-side sampling.
  bool mlx_multimodal = false;
  {
    auto result = module->get("has_vision_encoder");
    if (result.ok()) {
      mlx_multimodal = result->toScalar().to<bool>();
    }
  }
  // One --method flag selects the forward for whichever path runs. The
  // embeds-input vision path maps the default "forward" to
  // "forward_from_embeddings" (its natural name) and otherwise uses the
  // override (e.g. target_forward_from_embeddings for a DFlash-vision .pte).
  const std::string vision_method = FLAGS_method == "forward"
      ? std::string("forward_from_embeddings")
      : FLAGS_method;
  executorch::aten::ScalarType activation_dtype =
      executorch::aten::ScalarType::BFloat16;
  // MLX: release MLX's cached buffer pool every N calls to bound memory growth
  // during long sessions (mirrors the gemma4 runner).
  constexpr int kMLXClearCacheInterval = 1;
  executorch::runtime::BackendOptions<1> mlx_opts;
  executorch::runtime::LoadBackendOptionsMap mlx_options_map;
  if (mlx_opts.set_option(
          executorch::backends::mlx::kClearCacheIntervalKey,
          kMLXClearCacheInterval) != Error::Ok ||
      mlx_options_map.set_options(
          executorch::backends::mlx::kMLXBackendId, mlx_opts.view()) !=
          Error::Ok) {
    ET_LOG(Error, "Failed to set MLX clear_cache_interval option");
    return 1;
  }
  ET_LOG(Info, "MLX clear_cache_interval=%d", kMLXClearCacheInterval);
  executorch::runtime::LoadBackendOptionsMap* mlx_opts_ptr = &mlx_options_map;
  if (mlx_multimodal) {
    if (read_activation_dtype(*module, activation_dtype) != Error::Ok) {
      return 1;
    }
    if (module->load_method("embed_text", nullptr, nullptr, mlx_opts_ptr) !=
        Error::Ok) {
      ET_LOG(Error, "Failed to load embed_text method");
      return 1;
    }
    if (module->load_method(
            vision_method.c_str(), nullptr, nullptr, mlx_opts_ptr) !=
        Error::Ok) {
      ET_LOG(Error, "Failed to load %s method", vision_method.c_str());
      return 1;
    }
    if (has_image &&
        module->load_method("vision_encoder", nullptr, nullptr, mlx_opts_ptr) !=
            Error::Ok) {
      ET_LOG(
          Error,
          "Failed to load vision_encoder — was the model exported with "
          "--mmproj?");
      return 1;
    }
  } else {
    if (has_image) {
      ET_LOG(
          Error,
          "--image_path requires a vision-enabled .pte (export with --mmproj)");
      return 1;
    }
    if (module->load_method(
            FLAGS_method.c_str(), nullptr, nullptr, mlx_opts_ptr) !=
        Error::Ok) {
      ET_LOG(Error, "Failed to load %s method", FLAGS_method.c_str());
      return 1;
    }
  }
#endif
  stats.model_load_end_ms = llm::time_in_ms();

#ifdef EXECUTORCH_BUILD_CUDA
  cudaMemGetInfo(&gpu_free_bytes, &gpu_total_bytes);
  stats.gpu_free_after_load_bytes = gpu_free_bytes;
#endif

  // ---------------------------------------------------------------
  // NLL validation mode: read tokens, dump logits, exit
  // ---------------------------------------------------------------
  if (!FLAGS_nll_tokens_file.empty()) {
    // Read token IDs from binary file (int64_t[])
    std::ifstream tf(FLAGS_nll_tokens_file, std::ios::binary);
    if (!tf.is_open()) {
      ET_LOG(
          Error,
          "Cannot open --nll_tokens_file: %s",
          FLAGS_nll_tokens_file.c_str());
      return 1;
    }
    tf.seekg(0, std::ios::end);
    int64_t nbytes = tf.tellg();
    tf.seekg(0, std::ios::beg);
    int64_t num_tokens = nbytes / sizeof(int64_t);
    std::vector<int64_t> tokens(num_tokens);
    tf.read(reinterpret_cast<char*>(tokens.data()), nbytes);
    tf.close();

    printf("NLL mode: %lld tokens\n", (long long)num_tokens);

#ifdef EXECUTORCH_BUILD_CUDA
    if (nll_vision) {
      // -------------------------------------------------------------
      // Multimodal NLL: encode ordered images, splice their soft-token
      // features into the <|patch|> runs of the exact reference tokens,
      // prefill on embeddings, and write (T-1) per-position next-token
      // log-probs. The Python driver applies the patch/terminal mask and
      // compares to the perception reference.
      // -------------------------------------------------------------
      namespace gv = ::executorch::examples::muse_glimmer_vision;
      // Positional-embedding table (written next to model.pte by export.py).
      std::string pe_path = FLAGS_pos_embed_path;
      if (pe_path.empty()) {
        auto slash = FLAGS_model_path.find_last_of('/');
        std::string dir = (slash == std::string::npos)
            ? "."
            : FLAGS_model_path.substr(0, slash);
        pe_path = dir + "/pos_embed.bin";
      }
      std::vector<float> pos_table;
      try {
        pos_table = gv::load_pos_embed_table(pe_path);
      } catch (const std::exception& e) {
        ET_LOG(Error, "Failed to load pos_embed.bin: %s", e.what());
        return 1;
      }

      // Read the ordered image manifest (one path per line).
      std::vector<std::string> image_paths;
      {
        std::ifstream mf(FLAGS_nll_image_manifest);
        if (!mf.is_open()) {
          ET_LOG(
              Error,
              "Cannot open --nll_image_manifest: %s",
              FLAGS_nll_image_manifest.c_str());
          return 1;
        }
        std::string line;
        while (std::getline(mf, line)) {
          while (!line.empty() &&
                 (line.back() == '\n' || line.back() == '\r' ||
                  line.back() == ' ' || line.back() == '\t')) {
            line.pop_back();
          }
          if (!line.empty()) {
            image_paths.push_back(line);
          }
        }
      }
      if (image_paths.empty()) {
        ET_LOG(Error, "--nll_image_manifest is empty");
        return 1;
      }

      // Encode each image (in order) to host soft-token embeddings.
      std::vector<std::vector<uint16_t>> img_embeds(image_paths.size());
      std::vector<int64_t> img_counts(image_paths.size(), 0);
      int64_t vision_hidden = 0;
      int64_t total_soft_tokens = 0;
      for (size_t k = 0; k < image_paths.size(); ++k) {
        int64_t n = 0, h = 0;
        if (encode_image_nll(
                *module, image_paths[k], pos_table, img_embeds[k], n, h) !=
            Error::Ok) {
          return 1;
        }
        if (vision_hidden == 0) {
          vision_hidden = h;
        } else if (h != vision_hidden) {
          ET_LOG(Error, "inconsistent vision hidden dim across images");
          return 1;
        }
        img_counts[k] = n;
        total_soft_tokens += n;
        printf(
            "  image %zu: %s -> %lld soft tokens\n",
            k,
            image_paths[k].c_str(),
            (long long)n);
      }

      // Fail if total <|patch|> tokens != total vision features.
      int64_t patch_in_tokens = 0;
      for (int64_t t : tokens) {
        if (t == llm::kMuseGlimmerImagePatchTokenId) {
          ++patch_in_tokens;
        }
      }
      if (patch_in_tokens != total_soft_tokens) {
        ET_LOG(
            Error,
            "patch-token count %lld != total vision features %lld",
            (long long)patch_in_tokens,
            (long long)total_soft_tokens);
        return 1;
      }

      // Embed using the large prefill bound. The logits-producing forward has
      // its own bound below; long prefill cannot be used because it returns
      // only its final logits row.
      int64_t model_chunk = (*metadata_result)[llm::kMaxSeqLen] - 1;
      {
        auto gr = module->get("get_max_prefill_chunk");
        if (gr.ok()) {
          model_chunk = gr->toScalar().to<int64_t>();
        }
      }
      const int64_t embed_chunk_sz =
          apply_prefill_chunk_override(model_chunk, /*is_vision=*/true);
      int64_t forward_chunk_sz = embed_chunk_sz;
      if (nll_vision_uses_short_forward) {
        int64_t short_forward_bound = 4;
        auto gr = module->get("get_block_size");
        if (gr.ok()) {
          short_forward_bound = gr->toScalar().to<int64_t>();
        }
        forward_chunk_sz =
            std::min<int64_t>(forward_chunk_sz, short_forward_bound);
      } else if (forward_chunk_sz < 5) {
        forward_chunk_sz = 5;
      }

      // 1) embed_text over the whole doc (chunked) -> host embeds [T, hidden].
      int64_t hidden = 0;
      std::vector<uint16_t> embeds_host;
      for (int64_t ep = 0; ep < num_tokens; ep += embed_chunk_sz) {
        const int64_t elen = std::min(num_tokens - ep, embed_chunk_sz);
        std::vector<int64_t> tok_chunk(
            tokens.begin() + ep, tokens.begin() + ep + elen);
        auto tok_t = from_blob(
            tok_chunk.data(),
            {1, static_cast<SizesType>(elen)},
            executorch::aten::ScalarType::Long);
        auto et_res = module->execute("embed_text", {EValue(tok_t)});
        if (et_res.error() != Error::Ok) {
          ET_LOG(Error, "embed_text failed at %lld", (long long)ep);
          return 1;
        }
        const auto& te = et_res.get()[0].toTensor();
        if (te.dim() != 3 || te.size(1) != elen) {
          ET_LOG(Error, "embed_text returned unexpected shape");
          return 1;
        }
        if (hidden == 0) {
          hidden = te.size(2);
          if (hidden != vision_hidden) {
            ET_LOG(
                Error,
                "text hidden %lld != vision hidden %lld",
                (long long)hidden,
                (long long)vision_hidden);
            return 1;
          }
          embeds_host.resize(static_cast<size_t>(num_tokens * hidden));
        }
        if (copy_to_host(
                te,
                embeds_host.data() + ep * hidden,
                static_cast<size_t>(elen * hidden) * sizeof(uint16_t)) !=
            Error::Ok) {
          return 1;
        }
      }

      // 2) Splice vision features into the <|patch|> runs, in image order.
      {
        size_t cur_img = 0;
        int64_t within = 0; // soft-token offset inside the current image
        for (int64_t i = 0; i < num_tokens; ++i) {
          if (tokens[i] != llm::kMuseGlimmerImagePatchTokenId) {
            if (within != 0) {
              if (within != img_counts[cur_img]) {
                ET_LOG(
                    Error,
                    "patch run for image %zu has %lld tokens, expected %lld",
                    cur_img,
                    (long long)within,
                    (long long)img_counts[cur_img]);
                return 1;
              }
              ++cur_img;
              within = 0;
            }
            continue;
          }
          if (cur_img >= image_paths.size()) {
            ET_LOG(Error, "more <|patch|> runs than images");
            return 1;
          }
          if (within >= img_counts[cur_img]) {
            ET_LOG(Error, "patch run longer than image %zu features", cur_img);
            return 1;
          }
          std::memcpy(
              embeds_host.data() + i * hidden,
              img_embeds[cur_img].data() + within * hidden,
              static_cast<size_t>(hidden) * sizeof(uint16_t));
          ++within;
        }
        if (within != 0) { // trailing run at end of sequence
          if (within != img_counts[cur_img]) {
            ET_LOG(Error, "final patch run mismatch for image %zu", cur_img);
            return 1;
          }
          ++cur_img;
        }
        if (cur_img != image_paths.size()) {
          ET_LOG(
              Error, "spliced %zu of %zu images", cur_img, image_paths.size());
          return 1;
        }
      }

      // 3) Chunked forward on embeds; write per-position next-token log-prob.
      std::ofstream out(FLAGS_nll_output_file, std::ios::binary);
      if (!out.is_open()) {
        ET_LOG(
            Error,
            "Cannot open --nll_output_file: %s",
            FLAGS_nll_output_file.c_str());
        return 1;
      }
      int64_t pos = 0;
      while (pos < num_tokens) {
        int64_t clen = std::min(num_tokens - pos, forward_chunk_sz);
        const int64_t tail = num_tokens - pos - clen;
        if (!nll_vision_uses_short_forward && tail > 0 && tail < 5) {
          clen -= (5 - tail);
        }
        std::vector<int64_t> pos_data(clen);
        for (int64_t i = 0; i < clen; ++i) {
          pos_data[i] = pos + i;
        }
        uint16_t* chunk_ptr = embeds_host.data() + pos * hidden;
        auto chunk_embeds = from_blob(
            chunk_ptr,
            {1, static_cast<SizesType>(clen), static_cast<SizesType>(hidden)},
            executorch::aten::ScalarType::BFloat16);
        auto pos_t = from_blob(
            pos_data.data(),
            {static_cast<SizesType>(clen)},
            executorch::aten::ScalarType::Long);
        auto res = module->execute(
            nll_vision_forward_method, {EValue(chunk_embeds), EValue(pos_t)});
        if (res.error() != Error::Ok) {
          ET_LOG(Error, "NLL forward(embeds) failed at %lld", (long long)pos);
          return 1;
        }
        const auto& logits = res.get()[0].toTensor();
        const int64_t vocab_size = logits.size(logits.dim() - 1);
        const int64_t logits_rows = logits.size(logits.dim() - 2);
        // CUDA outputs use their bounded capacity (four rows) even when the
        // logical tail chunk has fewer rows. The valid logits are the leading
        // `clen` rows, matching DFlashSession::process_target_outputs().
        if (logits_rows < clen) {
          ET_LOG(
              Error,
              "%s returned only %lld logits rows for a %lld-token NLL chunk",
              nll_vision_forward_method.c_str(),
              (long long)logits_rows,
              (long long)clen);
          return 1;
        }
        std::vector<float> host_logits(static_cast<size_t>(clen) * vocab_size);
        if (copy_to_host(
                logits,
                host_logits.data(),
                static_cast<size_t>(clen) * vocab_size * sizeof(float)) !=
            Error::Ok) {
          return 1;
        }
        std::vector<float> lps;
        lps.reserve(clen);
        for (int64_t r = 0; r < clen; ++r) {
          const int64_t g = pos + r;
          if (g >= num_tokens - 1) {
            break; // last position has no next-token target
          }
          const float* row =
              host_logits.data() + static_cast<size_t>(r) * vocab_size;
          float m = row[0];
          for (int64_t v = 1; v < vocab_size; ++v) {
            if (row[v] > m) {
              m = row[v];
            }
          }
          double sum_exp = 0.0;
          for (int64_t v = 0; v < vocab_size; ++v) {
            sum_exp += std::exp(static_cast<double>(row[v]) - m);
          }
          const int64_t target = tokens[g + 1];
          const double lp =
              static_cast<double>(row[target]) - m - std::log(sum_exp);
          lps.push_back(static_cast<float>(lp));
        }
        out.write(
            reinterpret_cast<const char*>(lps.data()),
            lps.size() * sizeof(float));
        printf(
            "  nll chunk %lld-%lld done\n",
            (long long)pos,
            (long long)(pos + clen - 1));
        pos += clen;
      }
      out.close();
      printf(
          "Wrote %lld logprobs to %s\n",
          (long long)(num_tokens - 1),
          FLAGS_nll_output_file.c_str());
      return 0;
    }
#endif // EXECUTORCH_BUILD_CUDA

    if (FLAGS_nll_use_prefill && kSingleMethod) {
      ET_LOG(
          Error,
          "MLX forward returns only last-token logits; "
          "--nll_use_prefill requires the CUDA build.");
      return 1;
    }

    auto S = [](int64_t v) -> SizesType { return static_cast<SizesType>(v); };

    int64_t max_prefill_chunk = (*metadata_result)[llm::kMaxSeqLen] - 1;
    {
      auto get_result = module->get("get_max_prefill_chunk");
      if (get_result.ok()) {
        max_prefill_chunk = get_result->toScalar().to<int64_t>();
      }
    }
    max_prefill_chunk = apply_prefill_chunk_override(max_prefill_chunk);

    // Open output file
    std::ofstream out(FLAGS_nll_output_file, std::ios::binary);
    if (!out.is_open()) {
      ET_LOG(
          Error,
          "Cannot open --nll_output_file: %s",
          FLAGS_nll_output_file.c_str());
      return 1;
    }

    if (FLAGS_nll_use_prefill) {
      // Multi-token prefill: chunked, dump all logits (T, V) per chunk
      int64_t pos = 0;
      while (pos < num_tokens) {
        int64_t chunk_len = std::min(num_tokens - pos, max_prefill_chunk);
        const int64_t tail_len = num_tokens - pos - chunk_len;
        constexpr int64_t kMinPrefillBucket = 5;
        if (tail_len > 0 && tail_len < kMinPrefillBucket) {
          chunk_len =
              std::max<int64_t>(1, chunk_len - (kMinPrefillBucket - tail_len));
        }
        std::string method = method_for(chunk_len);

        std::vector<int64_t> chunk_tokens(
            tokens.begin() + pos, tokens.begin() + pos + chunk_len);
        std::vector<int64_t> pos_data(chunk_len);
        for (int64_t i = 0; i < chunk_len; i++)
          pos_data[i] = pos + i;

        auto tok_t = from_blob(
            chunk_tokens.data(),
            {1, S(chunk_len)},
            executorch::aten::ScalarType::Long);
        auto pos_t = from_blob(
            pos_data.data(),
            {S(chunk_len)},
            executorch::aten::ScalarType::Long);

        std::vector<EValue> inputs;
        inputs.push_back(EValue(tok_t));
        inputs.push_back(EValue(pos_t));

        auto result = module->execute(method, inputs);
        if (result.error() != Error::Ok) {
          ET_LOG(
              Error, "%s failed at pos %lld", method.c_str(), (long long)pos);
          return 1;
        }

        const auto& logits = result.get()[0].toTensor();
        int64_t vocab_size = logits.size(logits.dim() - 1);
        int64_t total_floats = chunk_len * vocab_size;
        std::vector<float> host_logits(total_floats);
        const void* src = logits.const_data_ptr();

#ifdef EXECUTORCH_BUILD_CUDA
        cudaPointerAttributes attrs{};
        bool on_device = cudaPointerGetAttributes(&attrs, src) == cudaSuccess &&
            attrs.type == cudaMemoryTypeDevice;
        if (on_device) {
          cudaMemcpy(
              host_logits.data(),
              src,
              total_floats * sizeof(float),
              cudaMemcpyDeviceToHost);
        } else {
          memcpy(host_logits.data(), src, total_floats * sizeof(float));
        }
#else
        memcpy(host_logits.data(), src, total_floats * sizeof(float));
#endif

        out.write(
            reinterpret_cast<const char*>(host_logits.data()),
            total_floats * sizeof(float));
        printf(
            "  prefill chunk %lld-%lld done\n",
            (long long)pos,
            (long long)(pos + chunk_len - 1));
        pos += chunk_len;
      }
    } else {
      // Single-token loop via decode method
      for (int64_t pos = 0; pos < num_tokens; pos++) {
        std::vector<int64_t> tok_data = {tokens[pos]};
        std::vector<int64_t> pos_data = {pos};

        auto tok_t = from_blob(
            tok_data.data(), {1, 1}, executorch::aten::ScalarType::Long);
        auto pos_t =
            from_blob(pos_data.data(), {1}, executorch::aten::ScalarType::Long);

        std::vector<EValue> inputs;
        inputs.push_back(EValue(tok_t));
        inputs.push_back(EValue(pos_t));

        auto result = module->execute(method_for(1), inputs);
        if (result.error() != Error::Ok) {
          ET_LOG(Error, "decode failed at pos %lld", (long long)pos);
          return 1;
        }

        const auto& logits = result.get()[0].toTensor();
        int64_t vocab_size = logits.size(logits.dim() - 1);
        std::vector<float> host_logits(vocab_size);
        const void* src = logits.const_data_ptr();

#ifdef EXECUTORCH_BUILD_CUDA
        cudaPointerAttributes attrs{};
        bool on_device = cudaPointerGetAttributes(&attrs, src) == cudaSuccess &&
            attrs.type == cudaMemoryTypeDevice;
        if (on_device) {
          cudaMemcpy(
              host_logits.data(),
              src,
              vocab_size * sizeof(float),
              cudaMemcpyDeviceToHost);
        } else {
          memcpy(host_logits.data(), src, vocab_size * sizeof(float));
        }
#else
        memcpy(host_logits.data(), src, vocab_size * sizeof(float));
#endif

        out.write(
            reinterpret_cast<const char*>(host_logits.data()),
            vocab_size * sizeof(float));

        if ((pos + 1) % 100 == 0) {
          printf(
              "  %lld / %lld tokens\n",
              (long long)(pos + 1),
              (long long)num_tokens);
        }
      }
    }

    out.close();
    printf("Wrote logits to %s\n", FLAGS_nll_output_file.c_str());
    return 0;
  }

  // EOS token ids
  auto eos_ids = llm::get_eos_ids(tokenizer.get(), module.get());
  eos_ids.insert(static_cast<uint64_t>(FLAGS_eos_id));

  auto S = [](int64_t v) -> SizesType { return static_cast<SizesType>(v); };

  // Vision: preprocess image on host, run vision_encoder, pull soft-token
  // embeddings back to host for splicing. Only when --image_path is set.
  int64_t num_soft_tokens = 0;
  int64_t vision_hidden = 0;
  std::vector<uint16_t> image_host; // [N, hidden] bf16
  namespace gv = ::executorch::examples::muse_glimmer_vision;
  if (has_image) {
    // Positional-embedding table (written next to model.pte by export.py).
    std::string pe_path = FLAGS_pos_embed_path;
    if (pe_path.empty()) {
      auto slash = FLAGS_model_path.find_last_of('/');
      std::string dir = (slash == std::string::npos)
          ? "."
          : FLAGS_model_path.substr(0, slash);
      pe_path = dir + "/pos_embed.bin";
    }
    std::vector<float> pos_table;
    try {
      pos_table = gv::load_pos_embed_table(pe_path);
    } catch (const std::exception& e) {
      ET_LOG(Error, "Failed to load pos_embed.bin: %s", e.what());
      return 1;
    }

    int img_w = 0, img_h = 0, img_c = 0;
    unsigned char* img_data =
        stbi_load(FLAGS_image_path.c_str(), &img_w, &img_h, &img_c, 3);
    if (img_data == nullptr) {
      ET_LOG(Error, "Failed to load image: %s", FLAGS_image_path.c_str());
      return 1;
    }
    gv::VisionInputs vis;
    try {
      vis = gv::preprocess_image(img_data, img_w, img_h, pos_table);
    } catch (const std::exception& e) {
      ET_LOG(Error, "preprocess_image failed: %s", e.what());
      stbi_image_free(img_data);
      return 1;
    }
    stbi_image_free(img_data);

    printf(
        "Image: %dx%d -> %" PRId64 " patches -> %" PRId64 " soft tokens\n",
        img_w,
        img_h,
        vis.num_patches,
        vis.num_soft_tokens);

    std::vector<EValue> ve_inputs;
    ve_inputs.push_back(EValue(vis.patches));
    ve_inputs.push_back(EValue(vis.pos_emb));
    ve_inputs.push_back(EValue(vis.cos_2d));
    ve_inputs.push_back(EValue(vis.sin_2d));
    ve_inputs.push_back(EValue(vis.sparse_perm));
    ve_inputs.push_back(EValue(vis.inv_perm));
    ve_inputs.push_back(EValue(vis.global_mask));
    ve_inputs.push_back(EValue(vis.sparse_mask));
    ve_inputs.push_back(EValue(vis.pixel_perm));

    auto ve_result = module->execute("vision_encoder", ve_inputs);
    if (ve_result.error() != Error::Ok) {
      ET_LOG(Error, "vision_encoder failed");
      return 1;
    }
    const auto& image_embeds = ve_result.get()[0].toTensor();
    if (image_embeds.dim() != 3 || image_embeds.size(0) != 1) {
      ET_LOG(Error, "vision_encoder output must be [1, N, hidden]");
      return 1;
    }
    num_soft_tokens = image_embeds.size(1);
    vision_hidden = image_embeds.size(2);
    image_host.assign(static_cast<size_t>(num_soft_tokens * vision_hidden), 0);
    if (copy_to_host(
            image_embeds,
            image_host.data(),
            static_cast<size_t>(num_soft_tokens * vision_hidden) *
                sizeof(uint16_t)) != Error::Ok) {
      return 1;
    }
  }

  std::vector<uint64_t> prompt_tokens;
  if (!FLAGS_prompt_tokens_file.empty()) {
    // Pre-tokenized path: read int64 LE token IDs (same format as the nll
    // path) and bypass encoding, which is slow for big prompts.
    std::ifstream tf(FLAGS_prompt_tokens_file, std::ios::binary);
    if (!tf.is_open()) {
      ET_LOG(
          Error,
          "Cannot open --prompt_tokens_file: %s",
          FLAGS_prompt_tokens_file.c_str());
      return 1;
    }
    tf.seekg(0, std::ios::end);
    int64_t nbytes = tf.tellg();
    tf.seekg(0, std::ios::beg);
    int64_t num_tokens = nbytes / sizeof(int64_t);
    std::vector<int64_t> file_tokens(num_tokens);
    tf.read(reinterpret_cast<char*>(file_tokens.data()), nbytes);
    tf.close();

    prompt_tokens.assign(file_tokens.begin(), file_tokens.end());
    if (!FLAGS_tokens_have_bos) {
      prompt_tokens.insert(
          prompt_tokens.begin(), static_cast<uint64_t>(FLAGS_bos_id));
    }
  } else {
    // Read prompt from file or flag
    std::string prompt_text = FLAGS_prompt;
    if (!FLAGS_prompt_file.empty()) {
      std::ifstream f(FLAGS_prompt_file);
      if (!f.is_open()) {
        ET_LOG(
            Error, "Failed to open prompt file: %s", FLAGS_prompt_file.c_str());
        return 1;
      }
      prompt_text = std::string(
          (std::istreambuf_iterator<char>(f)),
          std::istreambuf_iterator<char>());
    }

    auto tokenized = llm::tokenize_muse_glimmer_prompt(
        *tokenizer,
        prompt_text,
        static_cast<uint64_t>(FLAGS_bos_id),
        has_image ? std::optional<int64_t>(num_soft_tokens) : std::nullopt);
    if (!tokenized.ok()) {
      ET_LOG(
          Error,
          "Failed to tokenize prompt; --image_path requires exactly one "
          "<img> marker");
      return 1;
    }
    prompt_tokens = std::move(*tokenized);
  }

  int64_t num_prompt_tokens = static_cast<int64_t>(prompt_tokens.size());
  printf("Prompt tokens: %" PRId64 "\n", num_prompt_tokens);
  stats.num_prompt_tokens = num_prompt_tokens;

  stats.inference_start_ms = llm::time_in_ms();

  // On-device Gumbel sampling: temperature input (0 -> ~greedy via clamp).
  // Methods return one sampled token; read_token() copies the 4-byte scalar.
  // Unused on the MLX build (kSingleMethod), which samples on the host.
  [[maybe_unused]] float temp_val =
      FLAGS_temperature <= 0.0 ? 1e-6f : static_cast<float>(FLAGS_temperature);
  [[maybe_unused]] auto temp_tensor =
      from_blob(&temp_val, {1}, executorch::aten::ScalarType::Float);

  // Prefill (chunked to respect ring-buffer KV cache limit)
  int64_t max_prefill_chunk = (*metadata_result)[llm::kMaxSeqLen] - 1;
  {
    auto get_result = module->get("get_max_prefill_chunk");
    if (get_result.ok()) {
      max_prefill_chunk = get_result->toScalar().to<int64_t>();
    }
  }
  max_prefill_chunk =
      apply_prefill_chunk_override(max_prefill_chunk, has_image);

  uint64_t cur_token = 0;

#ifdef EXECUTORCH_BUILD_CUDA
  if (kSingleMethod) {
    int64_t prefill_pos = 0;
    while (prefill_pos < num_prompt_tokens) {
      int64_t chunk_len =
          std::min(num_prompt_tokens - prefill_pos, max_prefill_chunk);
      if (FLAGS_method == "target_forward") {
        chunk_len = std::min<int64_t>(chunk_len, 4);
      }
      std::vector<int64_t> token_data(
          prompt_tokens.begin() + prefill_pos,
          prompt_tokens.begin() + prefill_pos + chunk_len);
      std::vector<int64_t> pos_data(chunk_len);
      for (int64_t i = 0; i < chunk_len; i++) {
        pos_data[i] = prefill_pos + i;
      }
      auto tokens_tensor = from_blob(
          token_data.data(),
          {1, S(chunk_len)},
          executorch::aten::ScalarType::Long);
      auto pos_tensor = from_blob(
          pos_data.data(), {S(chunk_len)}, executorch::aten::ScalarType::Long);

      auto result = module->execute(
          FLAGS_method, {EValue(tokens_tensor), EValue(pos_tensor)});
      if (result.error() != Error::Ok) {
        ET_LOG(
            Error,
            "%s failed at pos %" PRId64,
            FLAGS_method.c_str(),
            prefill_pos);
        return 1;
      }
      const auto& logits = result.get()[0].toTensor();
      const int64_t last_pos =
          (logits.dim() >= 3) ? logits.size(logits.dim() - 2) - 1 : 0;
      cur_token = sample_from_logits(logits, last_pos, FLAGS_temperature);
      prefill_pos += chunk_len;
    }
  } else {
    // CUDA path: embed_text -> (splice image rows) -> chunked prefill on
    // embeds. Build the full [T, hidden] bf16 embedding on host, splice, then
    // chunk.
    int64_t hidden_size = 0;
    std::vector<uint16_t> embeds_host;
    for (int64_t ep = 0; ep < num_prompt_tokens; ep += max_prefill_chunk) {
      const int64_t elen = std::min(num_prompt_tokens - ep, max_prefill_chunk);
      std::vector<int64_t> token_chunk(
          prompt_tokens.begin() + ep, prompt_tokens.begin() + ep + elen);
      auto tokens_tensor = from_blob(
          token_chunk.data(), {1, S(elen)}, executorch::aten::ScalarType::Long);
      auto et_result = module->execute("embed_text", {EValue(tokens_tensor)});
      if (et_result.error() != Error::Ok) {
        ET_LOG(Error, "embed_text failed at position %" PRId64, ep);
        return 1;
      }
      const auto& text_embeds = et_result.get()[0].toTensor();
      if (text_embeds.dim() != 3 || text_embeds.size(1) != elen) {
        ET_LOG(Error, "embed_text returned unexpected shape");
        return 1;
      }
      if (hidden_size == 0) {
        hidden_size = text_embeds.size(2);
        embeds_host.resize(
            static_cast<size_t>(num_prompt_tokens * hidden_size));
      }
      if (text_embeds.size(2) != hidden_size ||
          copy_to_host(
              text_embeds,
              embeds_host.data() + ep * hidden_size,
              static_cast<size_t>(elen * hidden_size) * sizeof(uint16_t)) !=
              Error::Ok) {
        return 1;
      }
    }

    if (has_image) {
      if (vision_hidden != hidden_size) {
        ET_LOG(
            Error,
            "vision hidden %" PRId64 " != text hidden %" PRId64,
            vision_hidden,
            hidden_size);
        return 1;
      }
      int64_t img_idx = 0;
      for (int64_t i = 0; i < num_prompt_tokens; ++i) {
        if (prompt_tokens[i] !=
            static_cast<uint64_t>(llm::kMuseGlimmerImagePatchTokenId)) {
          continue;
        }
        if (img_idx >= num_soft_tokens) {
          ET_LOG(Error, "more <|patch|> tokens than vision soft tokens");
          return 1;
        }
        std::memcpy(
            embeds_host.data() + i * hidden_size,
            image_host.data() + img_idx * hidden_size,
            hidden_size * sizeof(uint16_t));
        ++img_idx;
      }
      if (img_idx != num_soft_tokens) {
        ET_LOG(
            Error,
            "spliced %" PRId64 " of %" PRId64 " vision soft tokens",
            img_idx,
            num_soft_tokens);
        return 1;
      }
    }

    int64_t prefill_pos = 0;
    while (prefill_pos < num_prompt_tokens) {
      int64_t chunk_len =
          std::min(num_prompt_tokens - prefill_pos, max_prefill_chunk);
      std::vector<int64_t> pos_data(chunk_len);
      for (int64_t i = 0; i < chunk_len; i++) {
        pos_data[i] = prefill_pos + i;
      }
      auto pos_tensor = from_blob(
          pos_data.data(), {S(chunk_len)}, executorch::aten::ScalarType::Long);
      // Single-token text chunks can use the faster token `decode` graph; image
      // patch rows and multi-token chunks go through `prefill` on the embeds.
      const bool is_patch = has_image &&
          prompt_tokens[prefill_pos] ==
              static_cast<uint64_t>(llm::kMuseGlimmerImagePatchTokenId);
      if (chunk_len == 1 && !is_patch) {
        int64_t tok_val = static_cast<int64_t>(prompt_tokens[prefill_pos]);
        auto tok_t =
            from_blob(&tok_val, {1, 1}, executorch::aten::ScalarType::Long);
        auto result = module->execute(
            "decode", {EValue(tok_t), EValue(pos_tensor), EValue(temp_tensor)});
        if (result.error() != Error::Ok) {
          ET_LOG(
              Error,
              "decode (prefill chunk=1) failed at %" PRId64,
              prefill_pos);
          return 1;
        }
        cur_token = read_token(result.get()[0].toTensor());
      } else {
        uint16_t* chunk_ptr = embeds_host.data() + prefill_pos * hidden_size;
        auto chunk_embeds = from_blob(
            chunk_ptr,
            {1, S(chunk_len), S(hidden_size)},
            executorch::aten::ScalarType::BFloat16);
        auto result = module->execute(
            "prefill",
            {EValue(chunk_embeds), EValue(pos_tensor), EValue(temp_tensor)});
        if (result.error() != Error::Ok) {
          ET_LOG(Error, "prefill failed at %" PRId64, prefill_pos);
          return 1;
        }
        cur_token = read_token(result.get()[0].toTensor());
      }
      prefill_pos += chunk_len;
    }
  }
#elif defined(EXECUTORCH_BUILD_MLX)
  // MLX path. Vision-capable .pte: embed_text -> (splice image rows) ->
  // chunked prefill on embeds. Text-only .pte: single token-input `forward`.
  // Host sampling in both cases.
  int64_t prefill_pos = 0;
  if (mlx_multimodal) {
    int64_t hidden_size = 0;
    std::vector<uint16_t> embeds_host;
    for (int64_t ep = 0; ep < num_prompt_tokens; ep += max_prefill_chunk) {
      const int64_t embed_len =
          std::min(num_prompt_tokens - ep, max_prefill_chunk);
      std::vector<int64_t> embed_tokens(
          prompt_tokens.begin() + ep, prompt_tokens.begin() + ep + embed_len);
      auto tokens_tensor = from_blob(
          embed_tokens.data(),
          {1, S(embed_len)},
          executorch::aten::ScalarType::Long);
      auto embed_result =
          module->execute("embed_text", {EValue(tokens_tensor)});
      if (embed_result.error() != Error::Ok) {
        ET_LOG(Error, "embed_text failed at position %" PRId64, ep);
        return 1;
      }
      const auto& text_embeds = embed_result.get()[0].toTensor();
      if (text_embeds.dim() != 3 || text_embeds.size(1) != embed_len) {
        ET_LOG(Error, "embed_text returned unexpected shape");
        return 1;
      }
      if (hidden_size == 0) {
        hidden_size = text_embeds.size(2);
        embeds_host.resize(
            static_cast<size_t>(num_prompt_tokens * hidden_size));
      }
      if (text_embeds.size(2) != hidden_size ||
          copy_to_host(
              text_embeds,
              embeds_host.data() + ep * hidden_size,
              static_cast<size_t>(embed_len * hidden_size) *
                  sizeof(uint16_t)) != Error::Ok) {
        return 1;
      }
    }

    if (has_image) {
      if (vision_hidden != hidden_size) {
        ET_LOG(
            Error,
            "vision hidden %" PRId64 " != text hidden %" PRId64,
            vision_hidden,
            hidden_size);
        return 1;
      }
      int64_t img_idx = 0;
      for (int64_t i = 0; i < num_prompt_tokens; ++i) {
        if (prompt_tokens[i] !=
            static_cast<uint64_t>(llm::kMuseGlimmerImagePatchTokenId)) {
          continue;
        }
        if (img_idx >= num_soft_tokens) {
          ET_LOG(Error, "more <|patch|> tokens than vision soft tokens");
          return 1;
        }
        std::memcpy(
            embeds_host.data() + i * hidden_size,
            image_host.data() + img_idx * hidden_size,
            hidden_size * sizeof(uint16_t));
        ++img_idx;
      }
      if (img_idx != num_soft_tokens) {
        ET_LOG(
            Error,
            "spliced %" PRId64 " of %" PRId64 " vision soft tokens",
            img_idx,
            num_soft_tokens);
        return 1;
      }
    }

    while (prefill_pos < num_prompt_tokens) {
      int64_t chunk_len =
          std::min(num_prompt_tokens - prefill_pos, max_prefill_chunk);
      std::vector<int64_t> pos_data(chunk_len);
      for (int64_t i = 0; i < chunk_len; i++) {
        pos_data[i] = prefill_pos + i;
      }
      auto pos_tensor = from_blob(
          pos_data.data(), {S(chunk_len)}, executorch::aten::ScalarType::Long);
      uint16_t* chunk_ptr = embeds_host.data() + prefill_pos * hidden_size;
      auto chunk_embeds = from_blob(
          chunk_ptr, {1, S(chunk_len), S(hidden_size)}, activation_dtype);
      auto result = module->execute(
          vision_method, {EValue(chunk_embeds), EValue(pos_tensor)});
      if (result.error() != Error::Ok) {
        ET_LOG(
            Error, "%s failed at %" PRId64, vision_method.c_str(), prefill_pos);
        return 1;
      }
      // `prefill` returns last-token logits [1,1,V]; a DFlash
      // `target_forward_from_embeddings` returns full [1,T,V] (with hidden at
      // output[1], ignored). Sampling the last row is correct for both.
      const auto& vlogits = result.get()[0].toTensor();
      const int64_t vlast =
          vlogits.dim() >= 2 ? vlogits.size(vlogits.dim() - 2) - 1 : 0;
      cur_token = sample_from_logits(vlogits, vlast, FLAGS_temperature);
      prefill_pos += chunk_len;
    }
  } else {
    while (prefill_pos < num_prompt_tokens) {
      int64_t chunk_len =
          std::min(num_prompt_tokens - prefill_pos, max_prefill_chunk);
      std::vector<int64_t> token_data(
          prompt_tokens.begin() + prefill_pos,
          prompt_tokens.begin() + prefill_pos + chunk_len);
      std::vector<int64_t> pos_data(chunk_len);
      for (int64_t i = 0; i < chunk_len; i++) {
        pos_data[i] = prefill_pos + i;
      }
      auto tokens_tensor = from_blob(
          token_data.data(),
          {1, S(chunk_len)},
          executorch::aten::ScalarType::Long);
      auto pos_tensor = from_blob(
          pos_data.data(), {S(chunk_len)}, executorch::aten::ScalarType::Long);

      auto result = module->execute(
          FLAGS_method, {EValue(tokens_tensor), EValue(pos_tensor)});
      if (result.error() != Error::Ok) {
        ET_LOG(
            Error,
            "%s failed at pos %" PRId64,
            FLAGS_method.c_str(),
            prefill_pos);
        return 1;
      }
      const auto& logits = result.get()[0].toTensor();
      const int64_t last_pos =
          logits.dim() >= 3 ? logits.size(logits.dim() - 2) - 1 : 0;
      cur_token = sample_from_logits(logits, last_pos, FLAGS_temperature);
      prefill_pos += chunk_len;
    }
  }
#endif // EXECUTORCH_BUILD_CUDA

  stats.prompt_eval_end_ms = llm::time_in_ms();
  double prefill_ms =
      static_cast<double>(stats.prompt_eval_end_ms - stats.inference_start_ms);
  printf(
      "Prefill: %" PRId64 " tokens in %.1f ms (%.1f tok/s)\n",
      num_prompt_tokens,
      prefill_ms,
      num_prompt_tokens * 1000.0 / prefill_ms);

#ifdef EXECUTORCH_BUILD_CUDA
  cudaDeviceSynchronize();
#endif

  // Decode loop
  int64_t pos = num_prompt_tokens;
  std::vector<int64_t> decode_token_data = {static_cast<int64_t>(cur_token)};
  std::vector<int64_t> decode_pos_data = {pos};
  auto decode_tokens = from_blob(
      decode_token_data.data(), {1, 1}, executorch::aten::ScalarType::Long);
  auto decode_pos = from_blob(
      decode_pos_data.data(), {1}, executorch::aten::ScalarType::Long);

  uint64_t prev_token = cur_token;
  std::vector<int64_t> generated_tokens;
  if (!FLAGS_generated_tokens_file.empty() && FLAGS_max_new_tokens > 0) {
    generated_tokens.reserve(FLAGS_max_new_tokens);
    generated_tokens.push_back(static_cast<int64_t>(cur_token));
  }
  for (int32_t step = 0; step < FLAGS_max_new_tokens; step++) {
    decode_token_data[0] = static_cast<int64_t>(cur_token);
    decode_pos_data[0] = pos;

#ifdef EXECUTORCH_BUILD_CUDA
    // CUDA: token-input `decode`, on-device sampling.
    std::vector<EValue> decode_inputs;
    decode_inputs.push_back(EValue(decode_tokens));
    decode_inputs.push_back(EValue(decode_pos));
    decode_inputs.push_back(EValue(temp_tensor));
    auto decode_result = module->execute("decode", decode_inputs);
    if (decode_result.error() != Error::Ok) {
      ET_LOG(Error, "Decode step %d failed", step);
      return 1;
    }
    prev_token = cur_token;
    cur_token = read_token(decode_result.get()[0].toTensor());
#elif defined(EXECUTORCH_BUILD_MLX)
    prev_token = cur_token;
    if (mlx_multimodal) {
      // MLX vision contract: decode = embed_text(token) -> prefill(embeds),
      // host sampling (prefill is the single KV-writing method).
      auto embed_result =
          module->execute("embed_text", {EValue(decode_tokens)});
      if (embed_result.error() != Error::Ok) {
        ET_LOG(Error, "embed_text failed at decode step %d", step);
        return 1;
      }
      auto decode_embed = embed_result.get()[0].toTensor();
      if (decode_embed.dim() != 3 || decode_embed.size(0) != 1 ||
          decode_embed.size(1) != 1) {
        ET_LOG(Error, "embed_text returned unexpected decode shape");
        return 1;
      }
      auto decode_result = module->execute(
          vision_method, {EValue(decode_embed), EValue(decode_pos)});
      if (decode_result.error() != Error::Ok) {
        ET_LOG(Error, "Decode step %d failed", step);
        return 1;
      }
      const auto& vlogits = decode_result.get()[0].toTensor();
      const int64_t vlast =
          vlogits.dim() >= 2 ? vlogits.size(vlogits.dim() - 2) - 1 : 0;
      cur_token = sample_from_logits(vlogits, vlast, FLAGS_temperature);
    } else {
      // MLX text contract: token-input `forward` (or `--method`), host
      // sampling.
      auto decode_result = module->execute(
          FLAGS_method, {EValue(decode_tokens), EValue(decode_pos)});
      if (decode_result.error() != Error::Ok) {
        ET_LOG(Error, "Decode step %d failed", step);
        return 1;
      }
      cur_token = sample_from_logits(
          decode_result.get()[0].toTensor(), 0, FLAGS_temperature);
    }
#endif

    if (step == 0) {
      stats.first_token_ms = llm::time_in_ms();
    }
    pos++;
    if (!FLAGS_generated_tokens_file.empty() &&
        generated_tokens.size() < static_cast<size_t>(FLAGS_max_new_tokens)) {
      generated_tokens.push_back(static_cast<int64_t>(cur_token));
    }

    auto decode_str = tokenizer->decode(prev_token, cur_token);
    if (decode_str.ok()) {
      printf("%s", decode_str->c_str());
      fflush(stdout);
    }

    if (!FLAGS_ignore_eos && eos_ids.find(cur_token) != eos_ids.end()) {
      printf("\n");
      break;
    }
  }

  stats.inference_end_ms = llm::time_in_ms();
  printf("\n");

  const int64_t num_generated = pos - num_prompt_tokens;
  stats.num_generated_tokens = num_generated;
  if (FLAGS_ignore_eos && num_generated != FLAGS_max_new_tokens) {
    ET_LOG(
        Error,
        "Exact-output mode emitted %" PRId64 " tokens, expected %d",
        num_generated,
        FLAGS_max_new_tokens);
    return 1;
  }
  if (!FLAGS_generated_tokens_file.empty()) {
    if (generated_tokens.size() != static_cast<size_t>(FLAGS_max_new_tokens)) {
      ET_LOG(Error, "Failed to capture the requested logical continuation");
      return 1;
    }
    std::ofstream out(FLAGS_generated_tokens_file, std::ios::binary);
    out.write(
        reinterpret_cast<const char*>(generated_tokens.data()),
        generated_tokens.size() * sizeof(int64_t));
    if (!out) {
      ET_LOG(Error, "Failed to write --generated_tokens_file");
      return 1;
    }
  }
  double decode_ms =
      static_cast<double>(stats.inference_end_ms - stats.prompt_eval_end_ms);
  printf(
      "Decode: %" PRId64 " tokens in %.1f ms (%.1f tok/s)\n",
      num_generated,
      decode_ms,
      num_generated * 1000.0 / decode_ms);
  printf("Prompt tokens: %" PRId64 "\n", num_prompt_tokens);

#ifdef EXECUTORCH_BUILD_CUDA
  cudaMemGetInfo(&gpu_free_bytes, &gpu_total_bytes);
  stats.gpu_free_after_generate_bytes = gpu_free_bytes;
  stats.gpu_peak_usage_mb =
      (stats.gpu_total_bytes - gpu_free_bytes) / 1024.0 / 1024.0;
#endif

  llm::print_report(stats);
  return 0;
}
