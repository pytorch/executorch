/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Gemma4 runner — supports text-only and multimodal (image + audio).
//
// TEXT-ONLY (gemma4.pte with forward method):
//   gemma4_runner --model_path gemma4.pte --prompt "..."
//
// MULTIMODAL (gemma4_multimodal.pte with 5 methods):
//   gemma4_runner --model_path gemma4_multimodal.pte \
//                 --image_path photo.jpg --prompt "Describe this."
//   gemma4_runner --model_path gemma4_multimodal.pte \
//                 --audio_path clip.wav --prompt "Transcribe the audio."
//
// The multimodal path bypasses MultimodalRunner and orchestrates the
// methods directly (vision_encoder takes two tensors; audio_preprocessor
// converts PCM to mel features before audio_encoder).

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <numeric>
#include <vector>

#include <gflags/gflags.h>

#include "gemma4_image_utils.h"

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/log.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

// Simple bilinear resize (avoids stb_image_resize.h dependency).
static void gemma4_bilinear_resize(
    const unsigned char* src, int sw, int sh, int sc,
    unsigned char* dst, int dw, int dh) {
  for (int y = 0; y < dh; ++y) {
    float fy = (y + 0.5f) * sh / dh - 0.5f;
    int y0 = (int)fy; int y1 = y0 + 1;
    float wy = fy - y0;
    if (y0 < 0) y0 = 0; if (y1 >= sh) y1 = sh - 1;
    for (int x = 0; x < dw; ++x) {
      float fx = (x + 0.5f) * sw / dw - 0.5f;
      int x0 = (int)fx; int x1 = x0 + 1;
      float wx = fx - x0;
      if (x0 < 0) x0 = 0; if (x1 >= sw) x1 = sw - 1;
      for (int c = 0; c < sc; ++c) {
        float v = (1-wx)*(1-wy)*src[(y0*sw+x0)*sc+c]
                + (  wx)*(1-wy)*src[(y0*sw+x1)*sc+c]
                + (1-wx)*(  wy)*src[(y1*sw+x0)*sc+c]
                + (  wx)*(  wy)*src[(y1*sw+x1)*sc+c];
        dst[(y*dw+x)*sc+c] = (unsigned char)(v + 0.5f);
      }
    }
  }
}

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

DEFINE_string(model_path, "gemma4.pte", "Model .pte (text-only or multimodal).");
DEFINE_string(data_path, "", "CUDA delegate data file (.ptd).");
DEFINE_string(tokenizer_path, "tokenizer.json", "Tokenizer file.");
DEFINE_string(prompt, "Hello!", "Text prompt.");
DEFINE_string(image_path, "", "Image file for vision+text generation.");
DEFINE_string(audio_path, "", "WAV file for audio+text generation.");
DEFINE_double(temperature, 0.0, "Sampling temperature (0 = greedy).");
DEFINE_int32(cpu_threads, -1, "CPU threads (-1 = auto).");
DEFINE_int32(seq_len, 512, "Max new tokens to generate.");
// Image is resized to kImageW × kImageH = 960 × 672 to produce exactly kMaxPatches=2520
// patches in a 60×42 grid, yielding 280 visual soft tokens after 3×3 spatial pooling.
DEFINE_bool(warmup, false, "Run warmup before generation.");
DEFINE_bool(raw_prompt, false,
    "Pass --prompt verbatim. Use with render_chat.py for system prompts.");

using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::runtime::EValue;
using executorch::aten::ScalarType;

// Helper: run a method with an initializer-list of EValue inputs.
static inline auto run(
    Module& m,
    const char* method,
    std::initializer_list<EValue> inputs) {
  return m.execute(method, std::vector<EValue>(inputs));
}

// ---------------------------------------------------------------------------
// WAV PCM loading (16-bit signed little-endian → float32 [-1, 1])
// ---------------------------------------------------------------------------
static std::vector<float> load_wav_pcm(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { ET_LOG(Error, "Cannot open: %s", path.c_str()); return {}; }

  // Parse RIFF header
  char riff[4]; f.read(riff, 4);
  if (std::strncmp(riff, "RIFF", 4) != 0) {
    ET_LOG(Error, "Not a RIFF file: %s", path.c_str()); return {};
  }
  uint32_t chunk_size; f.read(reinterpret_cast<char*>(&chunk_size), 4);
  char wave[4]; f.read(wave, 4);

  // Find fmt + data chunks
  uint16_t audio_format = 0, n_channels = 0, bits_per_sample = 0;
  uint32_t sample_rate = 0, data_size = 0;
  bool found_data = false;
  while (!found_data && f) {
    char id[4]; f.read(id, 4);
    uint32_t sz; f.read(reinterpret_cast<char*>(&sz), 4);
    if (!f) break;
    if (std::strncmp(id, "fmt ", 4) == 0) {
      f.read(reinterpret_cast<char*>(&audio_format), 2);
      f.read(reinterpret_cast<char*>(&n_channels), 2);
      f.read(reinterpret_cast<char*>(&sample_rate), 4);
      uint32_t byte_rate; f.read(reinterpret_cast<char*>(&byte_rate), 4);
      uint16_t block_align; f.read(reinterpret_cast<char*>(&block_align), 2);
      f.read(reinterpret_cast<char*>(&bits_per_sample), 2);
      if (sz > 16) f.seekg(sz - 16, std::ios::cur);
    } else if (std::strncmp(id, "data", 4) == 0) {
      data_size = sz;
      found_data = true;
    } else {
      f.seekg(sz, std::ios::cur);
    }
  }

  if (!found_data || bits_per_sample != 16) {
    ET_LOG(Error, "WAV must be 16-bit PCM: %s", path.c_str()); return {};
  }

  size_t n_samples = data_size / sizeof(int16_t);
  std::vector<int16_t> raw(n_samples);
  f.read(reinterpret_cast<char*>(raw.data()), data_size);

  // Mono mix-down if stereo; convert to float
  size_t n_out = (n_channels > 1) ? n_samples / n_channels : n_samples;
  std::vector<float> pcm(n_out);
  for (size_t i = 0; i < n_out; ++i) {
    if (n_channels == 2) {
      pcm[i] = (raw[i * 2] + raw[i * 2 + 1]) * 0.5f / 32768.0f;
    } else {
      pcm[i] = raw[i] / 32768.0f;
    }
  }
  ET_LOG(Info, "Loaded WAV %s: %zu samples @ %u Hz (%u ch)",
         path.c_str(), n_out, sample_rate, n_channels);
  return pcm;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int32_t main(int32_t argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

#if defined(ET_USE_THREADPOOL)
  uint32_t n_cores = FLAGS_cpu_threads == -1
      ? ::executorch::extension::cpuinfo::get_num_performant_cores()
      : static_cast<uint32_t>(FLAGS_cpu_threads);
  if (n_cores > 0) {
    ::executorch::extension::threadpool::get_threadpool()
        ->_unsafe_reset_threadpool(n_cores);
  }
#endif

  const std::string model_path = FLAGS_model_path;
  const std::string tokenizer_path = FLAGS_tokenizer_path;
  bool has_image = !FLAGS_image_path.empty();
  bool has_audio = !FLAGS_audio_path.empty();
  bool is_multimodal = has_image || has_audio;
  std::optional<const std::string> data_path =
      FLAGS_data_path.empty() ? std::nullopt
                              : std::optional<const std::string>(FLAGS_data_path);

  auto tokenizer = ::executorch::extension::llm::load_tokenizer(tokenizer_path.c_str());
  if (!tokenizer) {
    ET_LOG(Error, "Failed to load tokenizer: %s", tokenizer_path.c_str());
    return 1;
  }

  ::executorch::extension::llm::GenerationConfig config;
  config.max_new_tokens = FLAGS_seq_len;
  config.temperature = static_cast<float>(FLAGS_temperature);

  // -----------------------------------------------------------------------
  // TEXT-ONLY PATH — uses TextLLMRunner (gemma4.pte, `forward` method)
  // -----------------------------------------------------------------------
  if (!is_multimodal) {
    fprintf(stderr, "Creating text runner...\n");
    auto runner = ::executorch::extension::llm::create_text_llm_runner(
        model_path, std::move(tokenizer), data_path);
    if (!runner) { ET_LOG(Error, "Failed to create text runner"); return 1; }
    fprintf(stderr, "Loading model...\n");
    if (runner->load() != ::executorch::runtime::Error::Ok) {
      ET_LOG(Error, "Failed to load model"); return 1;
    }
    fprintf(stderr, "Model loaded.\n");
    std::string prompt = FLAGS_raw_prompt
        ? FLAGS_prompt
        : "<|turn>user\n" + FLAGS_prompt + "<turn|>\n<|turn>model\n";
    if (runner->generate(prompt, config) != ::executorch::runtime::Error::Ok) {
      ET_LOG(Error, "Generation failed"); return 1;
    }
    printf("\n");
    return 0;
  }

  // -----------------------------------------------------------------------
  // MULTIMODAL PATH — directly orchestrates Module methods
  // (gemma4_multimodal.pte: vision_encoder, audio_preprocessor,
  //  audio_encoder, token_embedding, text_decoder)
  // -----------------------------------------------------------------------
  fprintf(stderr, "Loading multimodal model %s...\n", model_path.c_str());
  auto module = std::make_unique<Module>(
      model_path, Module::LoadMode::MmapUseMlockIgnoreErrors);
  if (module->load() != ::executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Failed to load model"); return 1;
  }
  fprintf(stderr, "Model loaded.\n");

  // ---- 1. Encode modality into soft tokens ----
  std::vector<float> soft_tokens_data;
  int64_t n_soft_tokens = 0;
  int64_t soft_hidden = 1536;

  if (has_image) {
    // Load + resize image
    int w, h, c;
    unsigned char* img = stbi_load(FLAGS_image_path.c_str(), &w, &h, &c, 3);
    if (!img) {
      ET_LOG(Error, "Failed to load image: %s", FLAGS_image_path.c_str()); return 1;
    }
    c = 3; // forced RGB
    // Resize to exactly kImageW × kImageH = 960 × 672 for a 60×42 patch grid.
    const int iw = gemma4::kImageW, ih = gemma4::kImageH;
    std::vector<uint8_t> resized(iw * ih * 3);
    gemma4_bilinear_resize(img, w, h, 3, resized.data(), iw, ih);
    stbi_image_free(img);

    // HWC → CHW, rescale to [0,1] (do_rescale=True, factor=1/255; do_normalize=False)
    std::vector<float> chw(3 * ih * iw);
    for (int hh = 0; hh < ih; ++hh)
      for (int ww = 0; ww < iw; ++ww)
        for (int cc = 0; cc < 3; ++cc)
          chw[cc * ih * iw + hh * iw + ww] = resized[hh * iw * 3 + ww * 3 + cc] / 255.0f;

    // Patchify
    std::vector<float> pv;
    std::vector<int64_t> pp;
    gemma4::patchify(chw.data(), 3, ih, iw, pv, pp);

    // vision_encoder(pixel_values, pixel_position_ids)
    auto pv_t = from_blob(pv.data(),
                          {1, gemma4::kMaxPatches, gemma4::kPatchDim},
                          ScalarType::Float);
    auto pp_t = from_blob(pp.data(),
                          {1, gemma4::kMaxPatches, 2},
                          ScalarType::Long);
    auto vis_res = run(*module, "vision_encoder", {*pv_t, *pp_t});
    if (!vis_res.ok()) { ET_LOG(Error, "vision_encoder failed"); return 1; }
    auto vis_out = vis_res.get()[0].toTensor();
    // vis_out: (N_soft, 1536) — copy to soft_tokens_data
    n_soft_tokens = vis_out.numel() / soft_hidden;
    soft_tokens_data.assign(
        vis_out.data_ptr<float>(),
        vis_out.data_ptr<float>() + vis_out.numel());
    ET_LOG(Info, "Vision encoded: %lld soft tokens", (long long)n_soft_tokens);

  } else if (has_audio) {
    // Load WAV PCM
    auto pcm = load_wav_pcm(FLAGS_audio_path);
    if (pcm.empty()) { ET_LOG(Error, "Failed to load audio"); return 1; }

    // audio_preprocessor: (1, N_samples) → (1, T, 128)
    auto wav_t = from_blob(
        pcm.data(), {1, static_cast<int>(pcm.size())}, ScalarType::Float);
    auto pre_res = run(*module, "audio_preprocessor", {*wav_t});
    if (!pre_res.ok()) { ET_LOG(Error, "audio_preprocessor failed"); return 1; }
    auto mel = pre_res.get()[0].toTensor();
    ET_LOG(Info, "Mel features: (%lld, %lld, %lld)",
           (long long)mel.size(0), (long long)mel.size(1), (long long)mel.size(2));

    // audio_encoder: (1, T, 128) → (1, T', 1536)
    // The encoder is exported with a fixed T=kAudioFrames (200). Truncate if longer,
    // zero-pad if shorter, so shape always matches the exported model.
    constexpr int64_t kAudioFrames = 200;
    std::vector<float> mel_fixed(kAudioFrames * 128, 0.0f);
    int64_t src_frames = std::min(mel.size(1), kAudioFrames);
    std::memcpy(mel_fixed.data(), mel.data_ptr<float>(), src_frames * 128 * sizeof(float));
    auto mel_t = from_blob(mel_fixed.data(), {1, kAudioFrames, 128}, ScalarType::Float);
    auto aud_res = run(*module, "audio_encoder", {*mel_t});
    if (!aud_res.ok()) { ET_LOG(Error, "audio_encoder failed"); return 1; }
    auto aud_out = aud_res.get()[0].toTensor();
    // aud_out: (1, T', 1536) → flatten to (T', 1536)
    n_soft_tokens = aud_out.size(1);
    soft_tokens_data.assign(
        aud_out.data_ptr<float>(),
        aud_out.data_ptr<float>() + aud_out.numel());
    ET_LOG(Info, "Audio encoded: %lld soft tokens", (long long)n_soft_tokens);
  }

  // ---- 2. Build full prompt and tokenize ----
  std::string prefix_text, suffix_text;
  if (!FLAGS_raw_prompt) {
    if (has_image)
      prefix_text = "<bos><|turn>user\n<|image>\n";
    else if (has_audio)
      prefix_text = "<bos><|turn>user\n<|audio>\n";
    else
      prefix_text = "<bos><|turn>user\n";
    suffix_text = FLAGS_prompt + "<turn|>\n<|turn>model\n";
  } else {
    prefix_text = FLAGS_prompt;
    suffix_text = "";
  }

  // Tokenize prefix + suffix (without BOS added by tokenizer; we handle it in text)
  auto encode = [&](const std::string& s) -> std::vector<int64_t> {
    if (s.empty()) return {};
    auto ids_res = tokenizer->encode(s, /*bos=*/0, /*eos=*/0);
    if (!ids_res.ok()) return {};
    const auto& ids_u = ids_res.get();
    return std::vector<int64_t>(ids_u.begin(), ids_u.end());
  };
  auto prefix_ids = encode(prefix_text);
  auto suffix_ids = encode(suffix_text);

  // ---- 3. Embed prefix + suffix ----
  auto embed_tokens = [&](const std::vector<int64_t>& ids) -> std::vector<float> {
    if (ids.empty()) return {};
    auto id_t = from_blob(
        const_cast<int64_t*>(ids.data()),
        {1, static_cast<int64_t>(ids.size())},
        ScalarType::Long);
    auto emb_res = run(*module, "token_embedding", {*id_t});
    if (!emb_res.ok()) return {};
    auto t = emb_res.get()[0].toTensor();
    return std::vector<float>(t.data_ptr<float>(), t.data_ptr<float>() + t.numel());
  };

  auto prefix_emb = embed_tokens(prefix_ids);
  auto suffix_emb = embed_tokens(suffix_ids);

  // Apply embedding scale (sqrt(hidden_size) ≈ 39.19) to text token embeddings.
  // HF Gemma4TextScaledWordEmbedding scales inside embed_tokens(); our exported
  // token_embedding wraps the underlying nn.Embedding directly. The scale is
  // baked into token_embedding in the export script (TokenEmbeddingExport), but
  // older ptes may lack it — applying here is correct for all pte versions.
  // Vision/audio soft tokens from embed_vision/embed_audio must NOT be scaled.
  const float emb_scale = std::sqrt(static_cast<float>(soft_hidden));
  for (float& v : prefix_emb) v *= emb_scale;
  for (float& v : suffix_emb) v *= emb_scale;

  // ---- 4. Build combined embeddings and PLI token IDs ----
  int64_t prefix_len = static_cast<int64_t>(prefix_ids.size());
  int64_t suffix_len = static_cast<int64_t>(suffix_ids.size());
  int64_t total_len = prefix_len + n_soft_tokens + suffix_len;

  // PLI (Per-Layer Input) token IDs: real IDs for text, modality placeholder for soft tokens.
  // <|image>=255999, <|audio>=256000 — same IDs HF Gemma4 uses in unified input_ids.
  // Used in v2 text_decoder (pte includes pli_token_ids as 3rd input).
  const int64_t modality_pli_id = has_image ? 255999LL : 256000LL;
  std::vector<int64_t> pli_ids;
  pli_ids.insert(pli_ids.end(), prefix_ids.begin(), prefix_ids.end());
  for (int64_t k = 0; k < n_soft_tokens; ++k) pli_ids.push_back(modality_pli_id);
  pli_ids.insert(pli_ids.end(), suffix_ids.begin(), suffix_ids.end());

  // Detect if this pte is a v2 export (text_decoder takes 3 inputs including pli_token_ids).
  // v1 pte: text_decoder(embeds[1,1,1536], pos[1]) — PLI=0 inside model
  // v2 pte: text_decoder(embeds[1,1,1536], pos[1], pli_token_ids[1,1]) — PLI from token IDs
  // Heuristic: check if the first decoded method call with 3 inputs succeeds.
  // We'll attempt a dry-run call and fall back to 2-input if it fails.
  bool has_pli = false;  // filled in below during first decode step

  std::vector<float> combined(total_len * soft_hidden);
  {
    float* dst = combined.data();
    if (!prefix_emb.empty())
      std::memcpy(dst, prefix_emb.data(), prefix_emb.size() * sizeof(float));
    dst += prefix_len * soft_hidden;
    if (!soft_tokens_data.empty())
      std::memcpy(dst, soft_tokens_data.data(), soft_tokens_data.size() * sizeof(float));
    dst += n_soft_tokens * soft_hidden;
    if (!suffix_emb.empty())
      std::memcpy(dst, suffix_emb.data(), suffix_emb.size() * sizeof(float));
  }

  // ---- 5. Prefill: feed one token at a time into the KV-cache text_decoder ----
  // v2 pte: 3-input decoder (embeds, pos, pli_token_ids)
  // v1 pte: 2-input decoder (embeds, pos) — PLI is zero inside model
  std::vector<float> logits_data;
  int64_t vocab_size = 0;
  for (int64_t i = 0; i < total_len; ++i) {
    auto emb_slice = from_blob(
        combined.data() + i * soft_hidden, {1, 1, (int)soft_hidden}, ScalarType::Float);
    int64_t pos_val = i;
    auto pos1 = from_blob(&pos_val, {1}, ScalarType::Long);

    auto call_decoder = [&]() {
      if (has_pli || i == 0) {
        // Try 3-input call (v2 pte with PLI token IDs)
        auto pli3 = run(*module, "text_decoder", {*emb_slice, *pos1,
            *from_blob(&pli_ids[i], {1, 1}, ScalarType::Long)});
        if (pli3.ok()) { has_pli = true; return pli3; }
        has_pli = false;  // fall back to v1 on first failure
      }
      return run(*module, "text_decoder", {*emb_slice, *pos1});
    };
    auto dec_res = call_decoder();
    if (!dec_res.ok()) { ET_LOG(Error, "text_decoder prefill failed at pos %lld", (long long)i); return 1; }
    if (i == total_len - 1) {
      auto logits_t = dec_res.get()[0].toTensor();
      vocab_size = logits_t.numel();
      logits_data.assign(logits_t.data_ptr<float>(), logits_t.data_ptr<float>() + vocab_size);
    }
  }
  if (vocab_size == 0) { ET_LOG(Error, "Prefill produced no logits"); return 1; }
  ET_LOG(Info, "Prefill done: PLI %s", has_pli ? "enabled (v2)" : "disabled (v1)");

  // ---- 6. Sample first token ----
  auto argmax = [](const float* data, int64_t n) -> int64_t {
    int64_t best = 0;
    for (int64_t i = 1; i < n; ++i)
      if (data[i] > data[best]) best = i;
    return best;
  };

  // EOS token IDs (from model metadata or defaults)
  const std::vector<int64_t> eos_ids = {1, 106, 50};
  auto is_eos = [&](int64_t tok) {
    for (auto e : eos_ids) if (tok == e) return true;
    return false;
  };

  int64_t cur_pos = total_len;
  int64_t next_token = argmax(logits_data.data(), vocab_size);
  // prev_token: last token of the input context (for BPE piece boundary detection).
  int64_t prev_token = suffix_ids.empty()
      ? (prefix_ids.empty() ? 0 : prefix_ids.back())
      : suffix_ids.back();

  // ---- 7. Decode loop ----
  int generated = 0;
  while (!is_eos(next_token) && generated < FLAGS_seq_len) {
    // Decode token string and print
    auto piece = tokenizer->decode(
        static_cast<uint64_t>(prev_token),
        static_cast<uint64_t>(next_token));
    if (piece.ok()) {
      printf("%s", piece.get().c_str());
      fflush(stdout);
    }
    prev_token = next_token;

    // Embed and decode next position — apply embedding scale
    int64_t tok_arr[1] = {next_token};
    auto tok_t = from_blob(tok_arr, {1, 1}, ScalarType::Long);
    auto emb_res = run(*module, "token_embedding", {*tok_t});
    if (!emb_res.ok()) break;
    auto tok_emb_t = emb_res.get()[0].toTensor();  // (1, 1, 1536)
    std::vector<float> tok_emb_data(
        tok_emb_t.data_ptr<float>(),
        tok_emb_t.data_ptr<float>() + tok_emb_t.numel());
    for (float& v : tok_emb_data) v *= emb_scale;
    auto tok_emb = from_blob(tok_emb_data.data(), {1, 1, (int)soft_hidden}, ScalarType::Float);

    int64_t pos_arr[1] = {cur_pos};
    auto pos1_t = from_blob(pos_arr, {1}, ScalarType::Long);
    auto call_decoder_decode = [&]() {
      if (has_pli)
        return run(*module, "text_decoder", {*tok_emb, *pos1_t,
            *from_blob(&next_token, {1, 1}, ScalarType::Long)});
      return run(*module, "text_decoder", {*tok_emb, *pos1_t});
    };
    auto dec2 = call_decoder_decode();
    if (!dec2.ok()) break;
    auto logits2 = dec2.get()[0].toTensor();
    next_token = argmax(logits2.data_ptr<float>(), vocab_size);
    ++cur_pos;
    ++generated;
  }
  printf("\n");
  return 0;
}
