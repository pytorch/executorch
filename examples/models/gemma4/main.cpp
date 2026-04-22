/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Gemma4 unified runner — single .pte serves text-only, image+text, and
// audio+text via ExecuTorch's standard MultimodalRunner.
//
// Examples:
//   gemma4_runner --model_path gemma4_multimodal.pte --prompt "..."
//   gemma4_runner --model_path gemma4_multimodal.pte --image_path photo.jpg --prompt "..."
//   gemma4_runner --model_path gemma4_multimodal.pte --audio_path clip.wav --prompt "..."

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

#include <gflags/gflags.h>

#include <executorch/extension/llm/runner/audio.h>
#include <executorch/extension/llm/runner/image.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/multimodal_decoder_runner.h>
#include <executorch/extension/llm/runner/multimodal_input.h>
#include <executorch/extension/llm/runner/multimodal_prefiller.h>
#include <executorch/extension/llm/runner/multimodal_runner.h>
#include <executorch/extension/llm/runner/text_token_generator.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/log.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

// ---------------------------------------------------------------------------
// Gemma4 PLI-aware decoder runner (Approach C)
//
// Overrides step() to pass pli_token_ids as the 3rd input to text_decoder
// when the pte supports it (3-input signature). PLI = pli_proj(h) + pli_emb(id)
// where id is the current decode token — this is what HF Gemma4 does internally.
// Falls back to standard 2-input call for older ptes transparently.
// ---------------------------------------------------------------------------
class Gemma4DecoderRunner
    : public ::executorch::extension::llm::MultimodalDecoderRunner {
 public:
  explicit Gemma4DecoderRunner(
      ::executorch::extension::Module* module,
      ::executorch::extension::llm::IOManager* io_manager)
      : MultimodalDecoderRunner(module, io_manager) {
    // PLI detection deferred to first step() call — module must be loaded first.
  }

  ::executorch::runtime::Result<::executorch::aten::Tensor> step(
      ::executorch::extension::TensorPtr& tokens,
      int64_t start_pos) override {
    namespace et = ::executorch;
    namespace ext = ::executorch::extension;
    namespace etllm = ::executorch::extension::llm;

    // Embed the current decode token.
    auto emb_res = module_->execute(etllm::kTokenEmbeddingMethod, tokens);
    if (!emb_res.ok()) return emb_res.error();
    auto emb = (*emb_res)[0];

    // Build start_pos tensor.
    auto pos_t = ext::from_blob(
        &start_pos, {1}, et::aten::ScalarType::Long);

    // Lazy PLI detection — runs after module is loaded (not at construction time).
    if (!pli_detected_) {
      auto meta = module_->method_meta(etllm::kTextModelMethod);
      has_pli_ = meta.ok() && (*meta).num_inputs() >= 3;
      pli_detected_ = true;
      ET_LOG(Info, "Gemma4DecoderRunner: PLI %s",
             has_pli_ ? "enabled (3-input text_decoder)"
                      : "disabled (2-input text_decoder)");
    }

    if (has_pli_) {
      // Pass current decode token as PLI ID — matches HF Gemma4 where each
      // position's PLI embedding is conditioned on that position's token ID.
      // Hold the id in a member so the from_blob tensor's data pointer remains
      // valid through module_->execute (which may queue/defer work).
      pli_id_buf_ = tokens->const_data_ptr<int64_t>()[0];
      auto pli_t = ext::from_blob(
          &pli_id_buf_, {1, 1}, et::aten::ScalarType::Long);
      auto out = module_->execute(
          etllm::kTextModelMethod, {emb, *pos_t, *pli_t});
      if (!out.ok()) return out.error();
      return (*out)[0].toTensor();
    }

    auto out = module_->execute(etllm::kTextModelMethod, {emb, *pos_t});
    if (!out.ok()) return out.error();
    return (*out)[0].toTensor();
  }

 private:
  bool has_pli_ = false;
  bool pli_detected_ = false;
  int64_t pli_id_buf_ = 0;  // backing storage for PLI from_blob in step()
};

// ---------------------------------------------------------------------------
// Gemma4-specific multimodal runner factory
// Identical to create_multimodal_runner but injects Gemma4DecoderRunner.
// ---------------------------------------------------------------------------
static std::unique_ptr<::executorch::extension::llm::MultimodalRunner>
create_gemma4_runner(
    const std::string& model_path,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    std::optional<const std::string> data_path) {
  namespace ext = ::executorch::extension;
  namespace etllm = ::executorch::extension::llm;

  // Use File mode (same as create_multimodal_runner default) so that program
  // metadata is immediately available for get_llm_metadata and method_meta.
  auto module = data_path.has_value()
      ? std::make_unique<ext::Module>(
            model_path, data_path.value(), ext::Module::LoadMode::File)
      : std::make_unique<ext::Module>(model_path, ext::Module::LoadMode::File);

  auto metadata_res = etllm::get_llm_metadata(tokenizer.get(), module.get());
  if (!metadata_res.ok()) {
    ET_LOG(Error, "Failed to get metadata"); return nullptr;
  }
  auto metadata = metadata_res.get();
  auto eos_ids = std::make_unique<std::unordered_set<uint64_t>>(
      etllm::get_eos_ids(tokenizer.get(), module.get()));

  auto io_manager = std::make_unique<etllm::IOManager>(*module);
  auto decoder = std::make_unique<Gemma4DecoderRunner>(
      module.get(), io_manager.get());
  auto prefiller = std::make_unique<etllm::MultimodalPrefiller>(
      module.get(), decoder.get(), tokenizer.get(), io_manager.get());
  auto stats = std::make_unique<etllm::Stats>();
  auto generator = std::make_unique<etllm::TextTokenGenerator>(
      tokenizer.get(), decoder.get(),
      metadata.at(etllm::kUseKVCache), std::move(eos_ids), stats.get());

  return std::make_unique<etllm::MultimodalRunner>(
      std::move(metadata), std::move(tokenizer), std::move(module),
      std::move(decoder), std::move(prefiller),
      std::move(io_manager), std::move(generator), std::move(stats));
}

DEFINE_string(model_path, "gemma4.pte", "Path to .pte (text-only or multimodal).");
DEFINE_string(data_path, "", "Optional CUDA delegate data file (.ptd).");
DEFINE_string(tokenizer_path, "tokenizer.json", "Tokenizer file.");
DEFINE_string(prompt, "Hello!", "Text prompt.");
DEFINE_string(image_path, "", "Image file for vision+text generation.");
DEFINE_string(audio_path, "", "WAV file for audio+text generation.");
DEFINE_double(temperature, 0.0, "Sampling temperature (0 = greedy).");
DEFINE_int32(cpu_threads, -1, "CPU threads (-1 = auto).");
DEFINE_int32(seq_len, 512, "Max new tokens to generate.");
DEFINE_bool(warmup, false, "Run warmup before generation.");

using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::extension::llm::Audio;
using executorch::extension::llm::Image;
using executorch::extension::llm::MultimodalInput;
using executorch::runtime::EValue;
using executorch::aten::ScalarType;

// Gemma4 vision input dims (must match VisionEncoderExport in encoders.py).
constexpr int kImageH = 672;
constexpr int kImageW = 960;
// Gemma4 audio encoder input dims (channels-first mel).
// T_mel must satisfy T = 48*k - 40 (stride-48 conv constraint).
// 1976 = 48*42 - 40 supports ~20s audio at 16kHz/hop=160.
// Must match the --audio-frames value used during export.
constexpr int kMelBins = 128;
constexpr int kMelFrames = 1976;

// ---------------------------------------------------------------------------
// Bilinear resize (avoids stb_image_resize.h dependency)
// ---------------------------------------------------------------------------
static void bilinear_resize_rgb(
    const unsigned char* src, int sw, int sh,
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
      for (int c = 0; c < 3; ++c) {
        float v = (1-wx)*(1-wy)*src[(y0*sw+x0)*3+c]
                + (  wx)*(1-wy)*src[(y0*sw+x1)*3+c]
                + (1-wx)*(  wy)*src[(y1*sw+x0)*3+c]
                + (  wx)*(  wy)*src[(y1*sw+x1)*3+c];
        dst[(y*dw+x)*3+c] = (unsigned char)(v + 0.5f);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Load image → CHW float [0, 1] at the model's expected resolution
// ---------------------------------------------------------------------------
static std::vector<float> load_image_chw(const std::string& path) {
  int w, h, c;
  unsigned char* img = stbi_load(path.c_str(), &w, &h, &c, 3);
  if (!img) { ET_LOG(Error, "Failed to load image: %s", path.c_str()); return {}; }

  std::vector<uint8_t> resized(kImageW * kImageH * 3);
  bilinear_resize_rgb(img, w, h, resized.data(), kImageW, kImageH);
  stbi_image_free(img);

  // HWC uint8 → CHW float [0, 1]. Vision encoder applies 2*(v-0.5) internally.
  std::vector<float> chw(3 * kImageH * kImageW);
  for (int hh = 0; hh < kImageH; ++hh)
    for (int ww = 0; ww < kImageW; ++ww)
      for (int cc = 0; cc < 3; ++cc)
        chw[cc * kImageH * kImageW + hh * kImageW + ww] =
            resized[hh * kImageW * 3 + ww * 3 + cc] / 255.0f;
  return chw;
}

// ---------------------------------------------------------------------------
// WAV PCM loading (16-bit signed little-endian → float32 [-1, 1])
// ---------------------------------------------------------------------------
static std::vector<float> load_wav_pcm(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { ET_LOG(Error, "Cannot open: %s", path.c_str()); return {}; }
  char riff[4]; f.read(riff, 4);
  if (std::strncmp(riff, "RIFF", 4) != 0) {
    ET_LOG(Error, "Not a RIFF file: %s", path.c_str()); return {};
  }
  uint32_t chunk_size; f.read(reinterpret_cast<char*>(&chunk_size), 4);
  char wave[4]; f.read(wave, 4);
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
  size_t n_out = (n_channels > 1) ? n_samples / n_channels : n_samples;
  std::vector<float> pcm(n_out);
  for (size_t i = 0; i < n_out; ++i) {
    pcm[i] = (n_channels == 2)
        ? (raw[i*2] + raw[i*2+1]) * 0.5f / 32768.0f
        : raw[i] / 32768.0f;
  }
  ET_LOG(Info, "Loaded WAV %s: %zu samples @ %u Hz (%u ch)",
         path.c_str(), n_out, sample_rate, n_channels);
  return pcm;
}

// ---------------------------------------------------------------------------
// Compute mel features via the .pte's audio_preprocessor method, then transpose
// and pad/truncate to channels-first (1, 128, 200) for the audio encoder.
// ---------------------------------------------------------------------------
static std::vector<float> compute_mel_chw(
    const std::string& model_path, const std::vector<float>& pcm) {
  // Load just the audio_preprocessor method from a separate Module instance.
  // mmap is shared with the main runner's Module so this is cheap.
  auto prep = std::make_unique<Module>(
      model_path, Module::LoadMode::MmapUseMlockIgnoreErrors);
  if (prep->load_method("audio_preprocessor") != ::executorch::runtime::Error::Ok) {
    ET_LOG(Error, "audio_preprocessor not found in pte");
    return {};
  }
  auto wav_t = from_blob(
      const_cast<float*>(pcm.data()),
      {1, static_cast<int>(pcm.size())}, ScalarType::Float);
  auto res = prep->execute("audio_preprocessor", std::vector<EValue>{*wav_t});
  if (!res.ok()) { ET_LOG(Error, "audio_preprocessor execute failed"); return {}; }
  auto mel = res.get()[0].toTensor();  // (1, T, 128) time-first
  int64_t T = mel.size(1), B = mel.size(2);
  if (B != kMelBins) {
    ET_LOG(Error, "Unexpected mel bin count: got %lld, expected %d", (long long)B, kMelBins);
    return {};
  }
  // Transpose (1, T, 128) → (1, 128, 200), zero-padding/truncating T to kMelFrames.
  std::vector<float> mel_chw(kMelBins * kMelFrames, 0.0f);
  int64_t T_use = std::min<int64_t>(T, kMelFrames);
  const float* src = mel.const_data_ptr<float>();
  for (int b = 0; b < kMelBins; ++b)
    for (int t = 0; t < T_use; ++t)
      mel_chw[b * kMelFrames + t] = src[t * kMelBins + b];
  return mel_chw;
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
  bool has_image = !FLAGS_image_path.empty();
  bool has_audio = !FLAGS_audio_path.empty();
  std::optional<const std::string> data_path =
      FLAGS_data_path.empty() ? std::nullopt
                              : std::optional<const std::string>(FLAGS_data_path);

  auto tokenizer = ::executorch::extension::llm::load_tokenizer(FLAGS_tokenizer_path.c_str());
  if (!tokenizer) {
    ET_LOG(Error, "Failed to load tokenizer: %s", FLAGS_tokenizer_path.c_str());
    return 1;
  }

  // Single code path for all modes — standard MultimodalRunner.
  fprintf(stderr, "Loading model %s...\n", model_path.c_str());
  auto runner = create_gemma4_runner(model_path, std::move(tokenizer), data_path);
  if (!runner) { ET_LOG(Error, "Failed to create multimodal runner"); return 1; }
  if (runner->load() != ::executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Failed to load model"); return 1;
  }
  fprintf(stderr, "Model loaded.\n");

  ::executorch::extension::llm::GenerationConfig config;
  config.max_new_tokens = FLAGS_seq_len;
  config.temperature = static_cast<float>(FLAGS_temperature);
  // The runner's wrapped_callback already prints generated tokens via safe_printf.
  // We don't add our own printing callback to avoid double output. echo=false
  // suppresses re-printing the prompt before generation.
  config.echo = false;

  // Build the input sequence following the Gemma4 chat template:
  //   <bos><|turn>user\n[<|image>\n][<|audio>\n][prompt]<turn|>\n<|turn>model\n
  // We split the template around the modality placeholders so each modality is
  // represented as its own MultimodalInput (image/audio tensor) and the
  // surrounding text remains text inputs. The MultimodalPrefiller iterates the
  // vector and calls the right encoder for each.
  std::vector<MultimodalInput> inputs;

  if (has_image) {
    auto chw = load_image_chw(FLAGS_image_path);
    if (chw.empty()) return 1;
    inputs.emplace_back(std::string("<bos><|turn>user\n<|image>\n"));
    inputs.emplace_back(Image(std::move(chw), kImageW, kImageH, 3));
    inputs.emplace_back(FLAGS_prompt + std::string("<turn|>\n<|turn>model\n"));
  } else if (has_audio) {
    auto pcm = load_wav_pcm(FLAGS_audio_path);
    if (pcm.empty()) return 1;
    auto mel_chw = compute_mel_chw(model_path, pcm);
    if (mel_chw.empty()) return 1;
    inputs.emplace_back(std::string("<bos><|turn>user\n<|audio>\n"));
    inputs.emplace_back(Audio(std::move(mel_chw), 1, kMelBins, kMelFrames));
    inputs.emplace_back(FLAGS_prompt + std::string("<turn|>\n<|turn>model\n"));
  } else {
    // Text-only: a single text input wrapping the chat template.
    inputs.emplace_back(
        std::string("<bos><|turn>user\n") + FLAGS_prompt +
        std::string("<turn|>\n<|turn>model\n"));
  }

  auto err = runner->generate(inputs, config);
  if (err != ::executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Generation failed");
    return 1;
  }
  printf("\n");
  return 0;
}
