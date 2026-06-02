/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Gemma 4 31B-IT runner for ExecuTorch. Supports two backends:
//   CUDA  — exports embed_text + vision_encoder + prefill + decode methods
//           under the unified-prefill contract (Pin #4):
//             embed_text(tokens [1,T] i64)              -> embeds [1,T,5376]
//             bf16 vision_encoder(pixels, pixel_position_ids) -> (image_embeds,
//             mask) prefill(inputs_embeds [1,T,5376] bf16,
//                     input_pos [T] i64,
//                     temperature [1] f32)              -> sampled [1,1] f32
//             decode(tokens [1,1] i64,
//                    input_pos [1] i64,
//                    temperature [1] f32)               -> sampled [1,1] f32
//           The runner ALWAYS uses embed_text → prefill (no token-based prefill
//           path); when --image_path is supplied it additionally runs
//           vision_encoder and splices image rows into the embeds at
//           <image_token> placeholders. The exported model performs Gumbel-max
//           sampling on-device and returns a single float token ID per call.
//   MLX   — exports the same four methods, but prefill/decode return
//           last-token logits; the runner samples on the host via
//           ``llm::logits_to_token`` with the same temperature semantics.

#include <gflags/gflags.h>

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/llm/sampler/util.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/platform/log.h>
#include <pytorch/tokenizers/hf_tokenizer.h>

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
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

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

// image_utils.h transitively pulls in stb_image_resize.h, so we define
// STB_IMAGE_RESIZE_IMPLEMENTATION here (exactly once, in this TU) before
// including image_utils.h. The deprecated stb_image_resize.h does NOT
// guard its implementation block, so including it twice in the same TU
// with the impl macro defined produces redefinition errors.
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <executorch/examples/models/gemma4/image_utils.h>

// Shared chat-template token IDs + build_vision_input_ids (single source of
// truth; mirrors the Python examples/models/gemma4/chat_template.py).
#include <executorch/examples/models/gemma4/runner/chat_template.h>

DEFINE_string(model_path, "", "Model .pte file path.");
DEFINE_string(data_path, "", "Data file (.ptd) for CUDA backend.");
DEFINE_string(tokenizer_path, "", "HuggingFace tokenizer.json path.");
DEFINE_string(prompt, "Hello", "Prompt text.");
DEFINE_string(
    prompt_file,
    "",
    "Path to file containing prompt text (overrides --prompt).");
DEFINE_double(temperature, 0.8, "Sampling temperature (0 = near-greedy).");
DEFINE_int32(max_new_tokens, 128, "Maximum tokens to generate.");
DEFINE_int32(bos_id, 2, "BOS token id to prepend (Gemma convention: 2).");
DEFINE_int32(eos_id, 1, "EOS token id (Gemma convention: 1).");
DEFINE_bool(
    raw_prompt,
    false,
    "Skip chat-template wrapping (use if the prompt is already formatted).");
DEFINE_bool(
    cuda_graph,
    false,
    "Enable CUDA graph capture for the decode method. CUDA only.");
DEFINE_string(
    image_path,
    "",
    "Optional: path to an image file (JPEG/PNG). When set, the runner uses "
    "the multimodal prefill path with the exported vision_encoder + "
    "embed_text + prefill_image methods.");
DEFINE_int32(
    max_vision_soft_tokens,
    280,
    "Maximum number of vision soft tokens (post-pooling image embedding rows) "
    "the runner asks the vision encoder for.");

namespace llm = ::executorch::extension::llm;
using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;

using SizesType = executorch::aten::SizesType;

// Chat-template token IDs + build_vision_input_ids come from the shared
// chat_template.h above (single source of truth, mirrored by the E2B/E4B
// gemma4_runner and the Python chat_template.py).
namespace g4 = ::executorch::examples::gemma4;

// Read a sampled token ID from a scalar float output (CUDA path).
static uint64_t read_token(const executorch::aten::Tensor& output) {
  const void* ptr = output.const_data_ptr();
  float val = 0.0f;

#ifdef EXECUTORCH_BUILD_CUDA
  cudaPointerAttributes attrs{};
  bool on_device = cudaPointerGetAttributes(&attrs, ptr) == cudaSuccess &&
      attrs.type == cudaMemoryTypeDevice;
  if (on_device) {
    cudaError_t err =
        cudaMemcpy(&val, ptr, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      ET_LOG(
          Error,
          "read_token: cudaMemcpy D2H failed: %s",
          cudaGetErrorString(err));
      return 0;
    }
  } else {
    memcpy(&val, ptr, sizeof(float));
  }
#else
  memcpy(&val, ptr, sizeof(float));
#endif

  return static_cast<uint64_t>(llrintf(val));
}

// Copy ``num_bytes`` from a runtime tensor's storage into ``dst`` on the
// host, regardless of whether the storage lives on host or device. Used to
// pull bf16 embedding rows back to host so we can splice them.
static Error
copy_to_host(const executorch::aten::Tensor& src, void* dst, size_t num_bytes) {
  const void* src_ptr = src.const_data_ptr();
#ifdef EXECUTORCH_BUILD_CUDA
  cudaPointerAttributes attrs{};
  bool on_device = cudaPointerGetAttributes(&attrs, src_ptr) == cudaSuccess &&
      attrs.type == cudaMemoryTypeDevice;
  if (on_device) {
    auto err = cudaMemcpy(dst, src_ptr, num_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      ET_LOG(
          Error,
          "copy_to_host: cudaMemcpy D2H failed: %s",
          cudaGetErrorString(err));
      return Error::Internal;
    }
  } else
#endif
  {
    memcpy(dst, src_ptr, num_bytes);
  }
  return Error::Ok;
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

  llm::Stats stats;

#ifdef EXECUTORCH_BUILD_CUDA
  size_t gpu_free_bytes = 0, gpu_total_bytes = 0;
  cudaMemGetInfo(&gpu_free_bytes, &gpu_total_bytes);
  stats.gpu_total_bytes = gpu_total_bytes;
  stats.gpu_free_before_load_bytes = gpu_free_bytes;
#endif

  stats.model_load_start_ms = llm::time_in_ms();

  // Tokenizer
  auto tokenizer = std::make_unique<tokenizers::HFTokenizer>();
  if (tokenizer->load(FLAGS_tokenizer_path) != tokenizers::Error::Ok) {
    ET_LOG(
        Error,
        "Failed to load tokenizer from %s",
        FLAGS_tokenizer_path.c_str());
    return 1;
  }

  // Module
  std::vector<std::string> data_files;
  if (!FLAGS_data_path.empty()) {
    data_files.push_back(FLAGS_data_path);
  }
  auto module = std::make_unique<Module>(
      FLAGS_model_path,
      data_files,
      Module::LoadMode::MmapUseMlockIgnoreErrors,
      /*event_tracer=*/nullptr,
      /*memory_allocator=*/nullptr,
      /*temp_allocator=*/nullptr,
      /*share_memory_arenas=*/true);

  // Get metadata
  auto metadata_result = llm::get_llm_metadata(tokenizer.get(), module.get());
  if (metadata_result.error() != Error::Ok) {
    ET_LOG(Error, "Failed to read model metadata");
    return 1;
  }

  int64_t max_prefill_chunk = (*metadata_result)[llm::kMaxSeqLen] - 1;
  {
    auto get_result = module->get("get_max_prefill_chunk");
    if (get_result.ok()) {
      max_prefill_chunk = get_result->toScalar().to<int64_t>();
    }
  }

  auto S = [](int64_t v) -> SizesType { return static_cast<SizesType>(v); };

  float temp_val =
      FLAGS_temperature <= 0.0 ? 1e-6f : static_cast<float>(FLAGS_temperature);

#ifdef EXECUTORCH_BUILD_CUDA
  if (FLAGS_cuda_graph) {
    executorch::runtime::BackendOptions<2> cuda_opts;
    cuda_opts.set_option("enable_cuda_graph_for_method", "decode");
    executorch::runtime::set_option("CudaBackend", cuda_opts.view());
    printf("CUDA graph enabled for decode method\n");
  }
  {
    executorch::runtime::BackendOptions<1> backend_options;
    auto set_err =
        backend_options.set_option("weight_sharing_across_methods", true);
    if (set_err != Error::Ok) {
      ET_LOG(
          Error,
          "Failed to set weight_sharing_across_methods: %d",
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
  if (module->load_method("embed_text") != Error::Ok) {
    ET_LOG(Error, "Failed to load embed_text method");
    return 1;
  }
  if (module->load_method("prefill") != Error::Ok) {
    ET_LOG(Error, "Failed to load prefill method");
    return 1;
  }
#ifdef EXECUTORCH_BUILD_CUDA
  if (module->load_method("decode") != Error::Ok) {
    ET_LOG(Error, "Failed to load decode method");
    return 1;
  }
  auto temp_tensor =
      from_blob(&temp_val, {1}, executorch::aten::ScalarType::Float);
#endif

  const bool has_image = !FLAGS_image_path.empty();
  if (has_image) {
    if (module->load_method("vision_encoder") != Error::Ok) {
      ET_LOG(
          Error,
          "Failed to load vision_encoder method — was the model exported "
          "with vision support?");
      return 1;
    }
  }

  stats.model_load_end_ms = llm::time_in_ms();

#ifdef EXECUTORCH_BUILD_CUDA
  cudaMemGetInfo(&gpu_free_bytes, &gpu_total_bytes);
  stats.gpu_free_after_load_bytes = gpu_free_bytes;
#endif

  auto eos_ids = llm::get_eos_ids(tokenizer.get(), module.get());
  eos_ids.insert(static_cast<uint64_t>(FLAGS_eos_id));
  auto turn_ids = tokenizer->encode("<turn|>", /*bos=*/0, /*eos=*/0);
  if (turn_ids.ok() && turn_ids->size() == 1) {
    eos_ids.insert(turn_ids.get()[0]);
  }

  // Read prompt
  std::string prompt_text = FLAGS_prompt;
  if (!FLAGS_prompt_file.empty()) {
    std::ifstream f(FLAGS_prompt_file);
    if (!f.is_open()) {
      ET_LOG(
          Error, "Failed to open prompt file: %s", FLAGS_prompt_file.c_str());
      return 1;
    }
    prompt_text = std::string(
        (std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  }

  // Wrap with Gemma 4 IT chat template unless --raw_prompt is set.
  // BOS is prepended separately later; this adds the turn structure and the
  // empty thought block required by the instruction-tuned model. The actual
  // encoding happens later, inside the has_image / else branch.
  if (!FLAGS_raw_prompt) {
    prompt_text = "<|turn>user\n" + prompt_text +
        "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
  }

  std::vector<int64_t> prompt_tokens;
  int64_t num_prompt_tokens = 0;

  stats.inference_start_ms = llm::time_in_ms();

  uint64_t cur_token = 0;
  int64_t prefill_pos = 0;
  // bf16 host buffer holding the full inputs_embeds tensor for prefill.
  // Sized to num_prompt_tokens * hidden_size after we know the shape.
  std::vector<uint16_t> inputs_embeds_host_storage;
  int64_t hidden_size = 0;

  // ===========================================================
  // 1. Build the prompt token sequence.
  //    - Image+text: chat template with BOI + image_token*N + EOI.
  //    - Text-only: existing behavior (BOS prepended to encoded prompt).
  // ===========================================================
  int64_t num_soft_tokens = 0; // vision_encoder output length N (image-only)
  int64_t num_valid_soft_tokens = 0; // rows with pooler_mask == true
  std::vector<uint8_t> pooler_mask_host; // [N] (1 = valid soft token)

  if (has_image) {
    // ---------- Load + patchify the image ----------
    int img_w = 0, img_h = 0, img_c = 0;
    unsigned char* img_data =
        stbi_load(FLAGS_image_path.c_str(), &img_w, &img_h, &img_c, 3);
    if (img_data == nullptr) {
      ET_LOG(Error, "Failed to load image: %s", FLAGS_image_path.c_str());
      return 1;
    }

    executorch::examples::gemma4::ImageData image_data;
    try {
      image_data = executorch::examples::gemma4::patchify_rgb_image(
          img_data, img_w, img_h, FLAGS_max_vision_soft_tokens);
    } catch (const std::exception& e) {
      ET_LOG(Error, "patchify_rgb_image failed: %s", e.what());
      stbi_image_free(img_data);
      return 1;
    }
    stbi_image_free(img_data);

    printf(
        "Image: %dx%d -> %" PRId64 " patches (max=%d)\n",
        img_w,
        img_h,
        image_data.num_valid_patches,
        FLAGS_max_vision_soft_tokens *
            executorch::examples::gemma4::kPoolingKernel *
            executorch::examples::gemma4::kPoolingKernel);

    // ---------- Run vision_encoder ----------
    auto ve_result = module->execute(
        "vision_encoder",
        {EValue(image_data.pixel_values),
         EValue(image_data.pixel_position_ids)});
    if (ve_result.error() != Error::Ok) {
      ET_LOG(Error, "vision_encoder failed");
      return 1;
    }
    auto& ve_outputs = ve_result.get();
    if (ve_outputs.size() < 2 || !ve_outputs[0].isTensor() ||
        !ve_outputs[1].isTensor()) {
      ET_LOG(
          Error,
          "vision_encoder must return (image_embeds, pooler_mask) tensors");
      return 1;
    }
    auto image_embeds_tensor = ve_outputs[0].toTensor();
    auto pooler_mask_tensor = ve_outputs[1].toTensor();

    if (image_embeds_tensor.dim() != 3 || image_embeds_tensor.size(0) != 1) {
      ET_LOG(
          Error,
          "Unexpected vision_encoder output shape (dim=%zu)",
          (size_t)image_embeds_tensor.dim());
      return 1;
    }
    num_soft_tokens = image_embeds_tensor.size(1);
    hidden_size = image_embeds_tensor.size(2);
    if (image_embeds_tensor.scalar_type() !=
        executorch::aten::ScalarType::BFloat16) {
      ET_LOG(
          Error,
          "vision_encoder must return BFloat16 (got dtype=%d)",
          (int)image_embeds_tensor.scalar_type());
      return 1;
    }

    // Pull the pooler mask to host and count valid soft tokens. Padded /
    // invalid rows (mask == false) must be skipped during the splice so the
    // C++ runner matches the Python inference path exactly.
    pooler_mask_host.assign(static_cast<size_t>(num_soft_tokens), 0);
    if (copy_to_host(
            pooler_mask_tensor,
            pooler_mask_host.data(),
            static_cast<size_t>(num_soft_tokens) * sizeof(uint8_t)) !=
        Error::Ok) {
      return 1;
    }
    for (int64_t i = 0; i < num_soft_tokens; ++i) {
      if (pooler_mask_host[i]) {
        ++num_valid_soft_tokens;
      }
    }

    printf(
        "Vision encoder: %" PRId64 " soft tokens (%" PRId64
        " valid), hidden_size=%" PRId64 "\n",
        num_soft_tokens,
        num_valid_soft_tokens,
        hidden_size);

    // Pull image embeddings (all N rows) to a temporary host buffer for
    // splicing later. (Done now so we don't have to keep the device-side
    // output alive across the embed_text call below.)
    inputs_embeds_host_storage.assign(
        static_cast<size_t>(num_soft_tokens * hidden_size), 0);
    if (copy_to_host(
            image_embeds_tensor,
            inputs_embeds_host_storage.data(),
            num_soft_tokens * hidden_size * sizeof(uint16_t)) != Error::Ok) {
      return 1;
    }
  }

  // ---------- Build the input_ids sequence ----------
  if (has_image) {
    prompt_tokens = g4::build_vision_input_ids(
        tokenizer.get(),
        prompt_text,
        num_valid_soft_tokens,
        static_cast<int64_t>(FLAGS_bos_id));
  } else {
    auto encode_result = tokenizer->encode(prompt_text);
    if (!encode_result.ok()) {
      ET_LOG(Error, "Failed to encode prompt");
      return 1;
    }
    auto token_vec = std::move(*encode_result);
    // Gemma models require BOS at the start of the sequence.
    token_vec.insert(token_vec.begin(), static_cast<uint64_t>(FLAGS_bos_id));
    prompt_tokens.assign(token_vec.begin(), token_vec.end());
  }
  num_prompt_tokens = static_cast<int64_t>(prompt_tokens.size());
  printf(
      "Prompt tokens%s: %" PRId64 "\n",
      has_image ? " (image+text)" : "",
      num_prompt_tokens);
  stats.num_prompt_tokens = num_prompt_tokens;

  // ---------- Run embed_text(tokens) -> embeds [1,T,5376] bf16 ----------
  auto tokens_tensor = from_blob(
      prompt_tokens.data(),
      {1, S(num_prompt_tokens)},
      executorch::aten::ScalarType::Long);

  auto et_result = module->execute("embed_text", {EValue(tokens_tensor)});
  if (et_result.error() != Error::Ok) {
    ET_LOG(Error, "embed_text failed");
    return 1;
  }
  auto& et_outputs = et_result.get();
  if (et_outputs.empty() || !et_outputs[0].isTensor()) {
    ET_LOG(Error, "embed_text produced no tensor output");
    return 1;
  }
  auto text_embeds_tensor = et_outputs[0].toTensor();
  if (text_embeds_tensor.dim() != 3 || text_embeds_tensor.size(0) != 1 ||
      text_embeds_tensor.size(1) != num_prompt_tokens) {
    ET_LOG(
        Error,
        "embed_text returned unexpected shape (T=%" PRId64 ")",
        num_prompt_tokens);
    return 1;
  }
  if (text_embeds_tensor.scalar_type() !=
      executorch::aten::ScalarType::BFloat16) {
    ET_LOG(
        Error,
        "embed_text must return BFloat16 (got dtype=%d)",
        (int)text_embeds_tensor.scalar_type());
    return 1;
  }
  if (hidden_size == 0) {
    hidden_size = text_embeds_tensor.size(2);
  } else if (hidden_size != text_embeds_tensor.size(2)) {
    ET_LOG(
        Error,
        "hidden_size mismatch: vision=%" PRId64 " vs embed_text=%" PRId64,
        hidden_size,
        static_cast<int64_t>(text_embeds_tensor.size(2)));
    return 1;
  }

  // ---------- Splice (host) ----------
  // For text-only: just copy text embeds into the host buffer as-is.
  // For image+text: stash text embeds first, then overwrite rows where
  // tokens[i] == g4::kImageTokenId with the image rows we already pulled.
  int64_t total_elems = num_prompt_tokens * hidden_size;
  std::vector<uint16_t> image_host;
  if (has_image) {
    // Move the image embeds we previously copied into a separate vector,
    // because we need inputs_embeds_host_storage for the full T-row
    // tensor.
    image_host = std::move(inputs_embeds_host_storage);
  }
  inputs_embeds_host_storage.assign(static_cast<size_t>(total_elems), 0);

  // Pull text embeddings to host.
  if (copy_to_host(
          text_embeds_tensor,
          inputs_embeds_host_storage.data(),
          total_elems * sizeof(uint16_t)) != Error::Ok) {
    return 1;
  }

  if (has_image) {
    // For every <image> placeholder, advance to the next VALID image row
    // (skipping rows whose pooler_mask is false) and overwrite the text row.
    // Mirrors inference.py so padded/invalid soft tokens never get spliced.
    int64_t image_idx = 0;
    int64_t spliced = 0;
    for (int64_t i = 0; i < num_prompt_tokens; ++i) {
      if (prompt_tokens[i] != g4::kImageTokenId) {
        continue;
      }
      while (image_idx < num_soft_tokens && !pooler_mask_host[image_idx]) {
        ++image_idx;
      }
      if (image_idx >= num_soft_tokens) {
        ET_LOG(
            Error,
            "Ran out of valid vision soft tokens at text position %" PRId64
            " (spliced %" PRId64 " of %" PRId64 ")",
            i,
            spliced,
            num_valid_soft_tokens);
        return 1;
      }
      std::memcpy(
          inputs_embeds_host_storage.data() + i * hidden_size,
          image_host.data() + image_idx * hidden_size,
          hidden_size * sizeof(uint16_t));
      ++image_idx;
      ++spliced;
    }
    if (spliced != num_valid_soft_tokens) {
      ET_LOG(
          Error,
          "Image-token / soft-token mismatch: spliced %" PRId64 " of %" PRId64,
          spliced,
          num_valid_soft_tokens);
      return 1;
    }
  }

  // ===========================================================
  // 2. Chunked prefill on the embeds tensor.
  //    Sliding KV layers use a ring buffer sized 2*sliding_window. A single
  //    prefill call must not exceed that, so we chunk on the seq dim.
  //    For chunk_len == 1 we fall back to `decode` (token-based) since
  //    `decode`'s graph is the same shape and runs faster than dispatching
  //    the dense prefill graph for a degenerate one-row input.
  // ===========================================================
  while (prefill_pos < num_prompt_tokens) {
    int64_t chunk_len =
        std::min(num_prompt_tokens - prefill_pos, max_prefill_chunk);

    std::vector<int64_t> pos_data(chunk_len);
    for (int64_t i = 0; i < chunk_len; ++i) {
      pos_data[i] = prefill_pos + i;
    }
    auto pos_tensor = from_blob(
        pos_data.data(), {S(chunk_len)}, executorch::aten::ScalarType::Long);

#ifdef EXECUTORCH_BUILD_CUDA
    const bool use_decode_fast_path = chunk_len == 1 &&
        !(has_image && prompt_tokens[prefill_pos] == g4::kImageTokenId);
#else
    const bool use_decode_fast_path = false;
#endif
    if (use_decode_fast_path) {
      // CUDA token decode fast path for single-row text chunks. MLX keeps all
      // KV-cache updates in prefill because MLX delegate state is method-local.
      int64_t tok_val = prompt_tokens[prefill_pos];
      auto chunk_tokens =
          from_blob(&tok_val, {1, 1}, executorch::aten::ScalarType::Long);

      std::vector<EValue> decode_inputs;
      decode_inputs.push_back(EValue(chunk_tokens));
      decode_inputs.push_back(EValue(pos_tensor));
#ifdef EXECUTORCH_BUILD_CUDA
      decode_inputs.push_back(EValue(temp_tensor));
#endif

      auto decode_result = module->execute("decode", decode_inputs);
      if (decode_result.error() != Error::Ok) {
        ET_LOG(
            Error, "decode (chunk_len=1) failed at pos %" PRId64, prefill_pos);
        return 1;
      }
#ifdef EXECUTORCH_BUILD_CUDA
      cur_token = read_token(decode_result.get()[0].toTensor());
#else
      cur_token = static_cast<uint64_t>(
          llm::logits_to_token(decode_result.get()[0].toTensor(), temp_val));
#endif
    } else {
      uint16_t* chunk_embeds_ptr =
          inputs_embeds_host_storage.data() + prefill_pos * hidden_size;
      auto chunk_embeds_tensor = from_blob(
          chunk_embeds_ptr,
          {1, S(chunk_len), S(hidden_size)},
          executorch::aten::ScalarType::BFloat16);

      std::vector<EValue> prefill_inputs;
      prefill_inputs.push_back(EValue(chunk_embeds_tensor));
      prefill_inputs.push_back(EValue(pos_tensor));
#ifdef EXECUTORCH_BUILD_CUDA
      prefill_inputs.push_back(EValue(temp_tensor));
#endif

      auto prefill_result = module->execute("prefill", prefill_inputs);
      if (prefill_result.error() != Error::Ok) {
        ET_LOG(Error, "prefill failed at pos %" PRId64, prefill_pos);
        return 1;
      }
#ifdef EXECUTORCH_BUILD_CUDA
      cur_token = read_token(prefill_result.get()[0].toTensor());
#else
      cur_token = static_cast<uint64_t>(
          llm::logits_to_token(prefill_result.get()[0].toTensor(), temp_val));
#endif
    }
    prefill_pos += chunk_len;
  }

  stats.prompt_eval_end_ms = llm::time_in_ms();
  // First generated token came from the last prefill chunk; TTFT is prefill.
  stats.first_token_ms = stats.prompt_eval_end_ms;

#ifdef EXECUTORCH_BUILD_CUDA
  cudaDeviceSynchronize();
#endif

  // Print the first generated token (from the last prefill chunk).
  // Use the last prompt token as the streaming-decode prefix so any BPE
  // partial-character handling stays correct.
  {
    auto first_str = tokenizer->decode(prompt_tokens.back(), cur_token);
    if (first_str.ok()) {
      printf("%s", first_str->c_str());
      fflush(stdout);
    }
  }

  // ---------------------------------------------------------------
  // Decode loop
  // ---------------------------------------------------------------
  int64_t pos = num_prompt_tokens;
  std::vector<int64_t> decode_token_data = {static_cast<int64_t>(cur_token)};
  std::vector<int64_t> decode_pos_data = {pos};
  auto decode_tokens = from_blob(
      decode_token_data.data(), {1, 1}, executorch::aten::ScalarType::Long);
  auto decode_pos = from_blob(
      decode_pos_data.data(), {1}, executorch::aten::ScalarType::Long);

  uint64_t prev_token = cur_token;
  bool hit_eos = eos_ids.find(cur_token) != eos_ids.end();
  for (int32_t step = 0; step < FLAGS_max_new_tokens && !hit_eos; step++) {
    decode_token_data[0] = static_cast<int64_t>(cur_token);
    decode_pos_data[0] = pos;

#ifdef EXECUTORCH_BUILD_CUDA
    std::vector<EValue> inputs;
    inputs.push_back(EValue(decode_tokens));
    inputs.push_back(EValue(decode_pos));
    inputs.push_back(EValue(temp_tensor));
    auto result = module->execute("decode", inputs);
#else
    auto embed_result = module->execute("embed_text", {EValue(decode_tokens)});
    if (embed_result.error() != Error::Ok) {
      ET_LOG(Error, "embed_text failed at decode step %d", step);
      return 1;
    }
    auto& embed_outputs = embed_result.get();
    if (embed_outputs.empty() || !embed_outputs[0].isTensor()) {
      ET_LOG(Error, "embed_text produced no tensor at decode step %d", step);
      return 1;
    }
    auto decode_embed = embed_outputs[0].toTensor();
    std::vector<EValue> inputs;
    inputs.push_back(EValue(decode_embed));
    inputs.push_back(EValue(decode_pos));
    auto result = module->execute("prefill", inputs);
#endif

    if (result.error() != Error::Ok) {
      ET_LOG(Error, "Decode step %d failed", step);
      return 1;
    }

    prev_token = cur_token;
#ifdef EXECUTORCH_BUILD_CUDA
    cur_token = read_token(result.get()[0].toTensor());
#else
    cur_token = static_cast<uint64_t>(
        llm::logits_to_token(result.get()[0].toTensor(), temp_val));
#endif
    pos++;

    auto decode_str = tokenizer->decode(prev_token, cur_token);
    if (decode_str.ok()) {
      printf("%s", decode_str->c_str());
      fflush(stdout);
    }

    hit_eos = eos_ids.find(cur_token) != eos_ids.end();
  }
  printf("\n");

  stats.inference_end_ms = llm::time_in_ms();
  stats.num_generated_tokens = pos - num_prompt_tokens;

#ifdef EXECUTORCH_BUILD_CUDA
  cudaMemGetInfo(&gpu_free_bytes, &gpu_total_bytes);
  stats.gpu_free_after_generate_bytes = gpu_free_bytes;
  stats.gpu_peak_usage_mb =
      (stats.gpu_total_bytes - gpu_free_bytes) / 1024.0 / 1024.0;
#endif

  llm::print_report(stats);
  return 0;
}
