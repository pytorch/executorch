/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Gemma 4 31B-IT runner for ExecuTorch.
//
//   CUDA  — vision (4-method) contract:
//             embed_text(tokens [1,T] i64)                    -> embeds
//             [1,T,5376] bf16 vision_encoder(pixels, pixel_position_ids) ->
//             (image_embeds [1,N,5376]
//                                                                 bf16,
//                                                                 pooler_mask
//                                                                 [1,N] bool)
//             prefill(inputs_embeds [1,T,5376] bf16,
//                     input_pos [T] i64, temperature [1] f32) -> sampled [1,1]
//                     f32
//             decode(tokens [1,1] i64,
//                    input_pos [1] i64, temperature [1] f32)  -> sampled [1,1]
//                    f32
//           The CUDA runner uses embed_text -> prefill; when --image_path is
//           supplied it additionally runs vision_encoder and splices the valid
//           image rows (respecting pooler_mask) into the embeds at
//           <image_token> placeholders. Sampling is on-device (Gumbel-max), one
//           float id/call.
//
//   MLX   — text-only this branch: a single token-input ``forward`` method with
//           dynamic seq_len; the runner samples on the host via
//           ``llm::logits_to_token``. The image path is CUDA-only here; MLX
//           vision is added in the g4-vision-mlx branch.

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

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

// image_utils.h transitively pulls in stb_image_resize.h, so we define
// STB_IMAGE_RESIZE_IMPLEMENTATION here (exactly once, in this TU) before
// including image_utils.h. The deprecated stb_image_resize.h does NOT
// guard its implementation block, so including it twice in the same TU
// with the impl macro defined produces redefinition errors.
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <executorch/examples/models/gemma4/image_utils.h>

// Shared chat-template token IDs + build_vision_input_ids (mirrors the Python
// examples/models/gemma4/chat_template.py).
#include <executorch/examples/models/gemma4/runner/chat_template.h>
#endif // EXECUTORCH_BUILD_CUDA

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
    "embed_text + prefill methods. CUDA only.");
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

#ifdef EXECUTORCH_BUILD_CUDA
namespace g4 = ::executorch::examples::gemma4;

// Read a sampled token ID from a scalar float output (CUDA on-device sampling).
static uint64_t read_token(const executorch::aten::Tensor& output) {
  const void* ptr = output.const_data_ptr();
  float val = 0.0f;

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

  return static_cast<uint64_t>(llrintf(val));
}

// Copy ``num_bytes`` from a runtime tensor's storage into ``dst`` on the host,
// regardless of whether the storage lives on host or device. Used to pull bf16
// embedding rows (and the bool pooler mask) back to host so we can splice.
static Error
copy_to_host(const executorch::aten::Tensor& src, void* dst, size_t num_bytes) {
  const void* src_ptr = src.const_data_ptr();
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
  } else {
    memcpy(dst, src_ptr, num_bytes);
  }
  return Error::Ok;
}
#endif // EXECUTORCH_BUILD_CUDA

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

  // Common: EOS ids + chat-template-wrapped prompt text (BOS prepended later).
  auto eos_ids = llm::get_eos_ids(tokenizer.get(), module.get());
  eos_ids.insert(static_cast<uint64_t>(FLAGS_eos_id));
  {
    auto turn_ids = tokenizer->encode("<turn|>", /*bos=*/0, /*eos=*/0);
    if (turn_ids.ok() && turn_ids->size() == 1) {
      eos_ids.insert(turn_ids.get()[0]);
    }
  }

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
  if (!FLAGS_raw_prompt) {
    prompt_text = "<|turn>user\n" + prompt_text +
        "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
  }

#ifdef EXECUTORCH_BUILD_CUDA
  // =====================================================================
  // CUDA backend — embeddings-based 4-method vision contract.
  // =====================================================================
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

  printf("Loading methods...\n");
  if (module->load_method("embed_text") != Error::Ok) {
    ET_LOG(Error, "Failed to load embed_text method");
    return 1;
  }
  if (module->load_method("prefill") != Error::Ok) {
    ET_LOG(Error, "Failed to load prefill method");
    return 1;
  }
  if (module->load_method("decode") != Error::Ok) {
    ET_LOG(Error, "Failed to load decode method");
    return 1;
  }
  auto temp_tensor =
      from_blob(&temp_val, {1}, executorch::aten::ScalarType::Float);

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
  cudaMemGetInfo(&gpu_free_bytes, &gpu_total_bytes);
  stats.gpu_free_after_load_bytes = gpu_free_bytes;

  std::vector<int64_t> prompt_tokens;
  int64_t num_prompt_tokens = 0;

  stats.inference_start_ms = llm::time_in_ms();

  uint64_t cur_token = 0;
  int64_t prefill_pos = 0;
  std::vector<uint16_t> inputs_embeds_host_storage; // [T, hidden] bf16
  int64_t hidden_size = 0;

  // ---- Vision: patchify image, run vision_encoder, pull rows to host ----
  int64_t num_image_rows = 0; // vision_encoder output length N
  int64_t num_valid_soft_tokens = 0; // rows with pooler_mask == true
  std::vector<uint16_t> image_host; // [N, hidden] bf16
  std::vector<uint8_t> pooler_mask_host; // [N] (1 = valid soft token)

  if (has_image) {
    int img_w = 0, img_h = 0, img_c = 0;
    unsigned char* img_data =
        stbi_load(FLAGS_image_path.c_str(), &img_w, &img_h, &img_c, 3);
    if (img_data == nullptr) {
      ET_LOG(Error, "Failed to load image: %s", FLAGS_image_path.c_str());
      return 1;
    }

    g4::ImageData image_data;
    try {
      image_data = g4::patchify_rgb_image(
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
        FLAGS_max_vision_soft_tokens * g4::kPoolingKernel * g4::kPoolingKernel);

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
    num_image_rows = image_embeds_tensor.size(1);
    hidden_size = image_embeds_tensor.size(2);
    if (image_embeds_tensor.scalar_type() !=
        executorch::aten::ScalarType::BFloat16) {
      ET_LOG(
          Error,
          "vision_encoder must return BFloat16 (got dtype=%d)",
          (int)image_embeds_tensor.scalar_type());
      return 1;
    }

    // Pull pooler_mask to host (1 byte / bool element) and count valid rows.
    pooler_mask_host.assign(static_cast<size_t>(num_image_rows), 0);
    if (copy_to_host(
            pooler_mask_tensor,
            pooler_mask_host.data(),
            static_cast<size_t>(num_image_rows) * sizeof(uint8_t)) !=
        Error::Ok) {
      return 1;
    }
    for (int64_t i = 0; i < num_image_rows; ++i) {
      if (pooler_mask_host[i]) {
        ++num_valid_soft_tokens;
      }
    }

    printf(
        "Vision encoder: %" PRId64 " soft tokens (%" PRId64
        " valid), hidden_size=%" PRId64 "\n",
        num_image_rows,
        num_valid_soft_tokens,
        hidden_size);

    // Pull image embeddings (all N rows) to host for splicing later.
    image_host.assign(static_cast<size_t>(num_image_rows * hidden_size), 0);
    if (copy_to_host(
            image_embeds_tensor,
            image_host.data(),
            static_cast<size_t>(num_image_rows * hidden_size) *
                sizeof(uint16_t)) != Error::Ok) {
      return 1;
    }
  }

  // ---- Build the input_ids sequence ----
  if (has_image) {
    // One <image> placeholder per VALID soft token (matches the Python path).
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
    token_vec.insert(token_vec.begin(), static_cast<uint64_t>(FLAGS_bos_id));
    prompt_tokens.assign(token_vec.begin(), token_vec.end());
  }
  num_prompt_tokens = static_cast<int64_t>(prompt_tokens.size());
  printf(
      "Prompt tokens%s: %" PRId64 "\n",
      has_image ? " (image+text)" : "",
      num_prompt_tokens);
  stats.num_prompt_tokens = num_prompt_tokens;

  // ---- embed_text(tokens) -> embeds [1,T,5376] bf16 ----
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

  // ---- Splice (host) ----
  int64_t total_elems = num_prompt_tokens * hidden_size;
  inputs_embeds_host_storage.assign(static_cast<size_t>(total_elems), 0);
  if (copy_to_host(
          text_embeds_tensor,
          inputs_embeds_host_storage.data(),
          total_elems * sizeof(uint16_t)) != Error::Ok) {
    return 1;
  }

  if (has_image) {
    // For every <image> placeholder, advance to the next VALID image row
    // (skipping rows whose pooler_mask is false) and overwrite the text row.
    // This mirrors inference.py and keeps padded/invalid soft tokens out.
    int64_t image_idx = 0;
    int64_t spliced = 0;
    for (int64_t i = 0; i < num_prompt_tokens; ++i) {
      if (prompt_tokens[i] != g4::kImageTokenId) {
        continue;
      }
      while (image_idx < num_image_rows && !pooler_mask_host[image_idx]) {
        ++image_idx;
      }
      if (image_idx >= num_image_rows) {
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

  // ---- Chunked prefill on the embeds tensor ----
  while (prefill_pos < num_prompt_tokens) {
    int64_t chunk_len =
        std::min(num_prompt_tokens - prefill_pos, max_prefill_chunk);

    std::vector<int64_t> pos_data(chunk_len);
    for (int64_t i = 0; i < chunk_len; ++i) {
      pos_data[i] = prefill_pos + i;
    }
    auto pos_tensor = from_blob(
        pos_data.data(), {S(chunk_len)}, executorch::aten::ScalarType::Long);

    // Single text rows use the faster token `decode` graph; image rows must go
    // through `prefill` on the spliced embeds.
    const bool use_decode_fast_path = chunk_len == 1 &&
        !(has_image && prompt_tokens[prefill_pos] == g4::kImageTokenId);
    if (use_decode_fast_path) {
      int64_t tok_val = prompt_tokens[prefill_pos];
      auto chunk_tokens =
          from_blob(&tok_val, {1, 1}, executorch::aten::ScalarType::Long);

      auto decode_result = module->execute(
          "decode",
          {EValue(chunk_tokens), EValue(pos_tensor), EValue(temp_tensor)});
      if (decode_result.error() != Error::Ok) {
        ET_LOG(
            Error, "decode (chunk_len=1) failed at pos %" PRId64, prefill_pos);
        return 1;
      }
      cur_token = read_token(decode_result.get()[0].toTensor());
    } else {
      uint16_t* chunk_embeds_ptr =
          inputs_embeds_host_storage.data() + prefill_pos * hidden_size;
      auto chunk_embeds_tensor = from_blob(
          chunk_embeds_ptr,
          {1, S(chunk_len), S(hidden_size)},
          executorch::aten::ScalarType::BFloat16);

      auto prefill_result = module->execute(
          "prefill",
          {EValue(chunk_embeds_tensor),
           EValue(pos_tensor),
           EValue(temp_tensor)});
      if (prefill_result.error() != Error::Ok) {
        ET_LOG(Error, "prefill failed at pos %" PRId64, prefill_pos);
        return 1;
      }
      cur_token = read_token(prefill_result.get()[0].toTensor());
    }
    prefill_pos += chunk_len;
  }

  stats.prompt_eval_end_ms = llm::time_in_ms();
  stats.first_token_ms = stats.prompt_eval_end_ms;
  cudaDeviceSynchronize();

  {
    auto first_str = tokenizer->decode(prompt_tokens.back(), cur_token);
    if (first_str.ok()) {
      printf("%s", first_str->c_str());
      fflush(stdout);
    }
  }

  // ---- Decode loop (token `decode`, on-device sampling) ----
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

    auto result = module->execute(
        "decode",
        {EValue(decode_tokens), EValue(decode_pos), EValue(temp_tensor)});
    if (result.error() != Error::Ok) {
      ET_LOG(Error, "Decode step %d failed", step);
      return 1;
    }

    prev_token = cur_token;
    cur_token = read_token(result.get()[0].toTensor());
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

  cudaMemGetInfo(&gpu_free_bytes, &gpu_total_bytes);
  stats.gpu_free_after_generate_bytes = gpu_free_bytes;
  stats.gpu_peak_usage_mb =
      (stats.gpu_total_bytes - gpu_free_bytes) / 1024.0 / 1024.0;

  llm::print_report(stats);
  return 0;

#else // EXECUTORCH_BUILD_CUDA
  // =====================================================================
  // MLX backend — text-only (single token-input `forward`, host sampling).
  // The image path is CUDA-only this branch.
  // =====================================================================
  if (!FLAGS_image_path.empty()) {
    ET_LOG(
        Error,
        "--image_path is only supported on the CUDA build in this branch; "
        "MLX vision is added in the g4-vision-mlx branch.");
    return 1;
  }
  if (FLAGS_cuda_graph) {
    ET_LOG(Info, "--cuda_graph ignored on non-CUDA build");
  }

  printf("Loading methods...\n");
  if (module->load_method("forward") != Error::Ok) {
    ET_LOG(Error, "Failed to load forward method");
    return 1;
  }

  stats.model_load_end_ms = llm::time_in_ms();

  // Encode prompt + BOS.
  auto encode_result = tokenizer->encode(prompt_text);
  if (!encode_result.ok()) {
    ET_LOG(Error, "Failed to encode prompt");
    return 1;
  }
  auto prompt_tokens = std::move(*encode_result);
  prompt_tokens.insert(
      prompt_tokens.begin(), static_cast<uint64_t>(FLAGS_bos_id));
  int64_t num_prompt_tokens = static_cast<int64_t>(prompt_tokens.size());
  printf("Prompt tokens: %" PRId64 "\n", num_prompt_tokens);
  stats.num_prompt_tokens = num_prompt_tokens;

  stats.inference_start_ms = llm::time_in_ms();

  // ---- Prefill (chunked token `forward`, host sampling) ----
  uint64_t cur_token = 0;
  int64_t prefill_pos = 0;
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

    auto result =
        module->execute("forward", {EValue(tokens_tensor), EValue(pos_tensor)});
    if (result.error() != Error::Ok) {
      ET_LOG(Error, "forward failed at pos %" PRId64, prefill_pos);
      return 1;
    }
    cur_token = static_cast<uint64_t>(
        llm::logits_to_token(result.get()[0].toTensor(), temp_val));
    prefill_pos += chunk_len;
  }

  stats.prompt_eval_end_ms = llm::time_in_ms();
  stats.first_token_ms = stats.prompt_eval_end_ms;

  {
    auto first_str = tokenizer->decode(prompt_tokens.back(), cur_token);
    if (first_str.ok()) {
      printf("%s", first_str->c_str());
      fflush(stdout);
    }
  }

  // ---- Decode loop (token `forward`, host sampling) ----
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

    auto result =
        module->execute("forward", {EValue(decode_tokens), EValue(decode_pos)});
    if (result.error() != Error::Ok) {
      ET_LOG(Error, "Decode step %d failed", step);
      return 1;
    }

    prev_token = cur_token;
    cur_token = static_cast<uint64_t>(
        llm::logits_to_token(result.get()[0].toTensor(), temp_val));
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

  llm::print_report(stats);
  return 0;
#endif // EXECUTORCH_BUILD_CUDA
}
