/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "runner.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include <executorch/extension/tensor/tensor.h>
#include <nlohmann/json.hpp>

namespace phonon {
namespace {
using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::runtime::Error;
using executorch::runtime::EValue;

// The official runtime's sentence chunk size.
constexpr size_t kMaxChunkTokens = 50;
constexpr int kFormatVersion = 1;

std::vector<EValue> execute(
    Module& model,
    const std::string& method,
    const std::vector<EValue>& inputs = {}) {
  auto result = model.execute(method, inputs);
  if (!result.ok()) {
    throw std::runtime_error(
        method + " failed: " + executorch::runtime::to_string(result.error()));
  }
  return std::move(result.get());
}

const float* float_data(const EValue& value) {
  const auto& tensor = value.toTensor();
  if (tensor.scalar_type() != ScalarType::Float) {
    throw std::runtime_error("Expected a float32 model output");
  }
  return tensor.const_data_ptr<float>();
}

std::vector<float> copy_floats(const EValue& value) {
  const float* data = float_data(value);
  return {data, data + value.toTensor().numel()};
}

// Code points of Python's str.isspace(), so that whitespace handling matches
// str.split() in run_phonon.py.
bool is_space(uint32_t c) {
  return (c >= 0x09 && c <= 0x0d) || (c >= 0x1c && c <= 0x20) || c == 0x85 ||
      c == 0xa0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200a) || c == 0x2028 ||
      c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000;
}

// Decodes the UTF-8 code point at text[i] and advances i; invalid bytes are
// returned as single code units.
uint32_t next_code_point(const std::string& text, size_t& i) {
  const auto byte = [&](size_t k) {
    return static_cast<unsigned char>(text[k]);
  };
  const unsigned char lead = byte(i);
  const int length = lead < 0x80 ? 1
      : (lead >> 5) == 0x6       ? 2
      : (lead >> 4) == 0xe       ? 3
      : (lead >> 3) == 0x1e      ? 4
                                 : 1;
  if (length == 1 || i + length > text.size()) {
    ++i;
    return lead;
  }
  uint32_t c = lead & (0x7f >> length);
  for (int k = 1; k < length; ++k) {
    if ((byte(i + k) & 0xc0) != 0x80) {
      ++i;
      return lead;
    }
    c = (c << 6) | (byte(i + k) & 0x3f);
  }
  i += length;
  return c;
}

std::vector<std::string> split_words(const std::string& text) {
  std::vector<std::string> words;
  std::string word;
  for (size_t i = 0; i < text.size();) {
    const size_t start = i;
    if (is_space(next_code_point(text, i))) {
      if (!word.empty()) {
        words.push_back(std::move(word));
        word.clear();
      }
    } else {
      word.append(text, start, i - start);
    }
  }
  if (!word.empty()) {
    words.push_back(std::move(word));
  }
  return words;
}

std::string join(const std::vector<std::string>& words) {
  std::string out;
  for (const auto& word : words) {
    if (!out.empty()) {
      out += ' ';
    }
    out += word;
  }
  return out;
}

std::string join(const std::string& a, const std::string& b) {
  return a.empty() ? b : b.empty() ? a : a + " " + b;
}

bool is_ascii_alnum(char c) {
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
      (c >= 'A' && c <= 'Z');
}

std::vector<std::string> split_at_sentence_ends(const std::string& text) {
  std::vector<std::string> sentences;
  size_t start = 0;
  for (size_t i = 1; i < text.size(); ++i) {
    if (text[i] == ' ' &&
        (text[i - 1] == '.' || text[i - 1] == '!' || text[i - 1] == '?')) {
      sentences.push_back(text.substr(start, i - start));
      start = i + 1;
    }
  }
  sentences.push_back(text.substr(start));
  return sentences;
}

std::vector<std::string> split_long_sentence(
    const std::string& sentence,
    const std::function<size_t(const std::string&)>& count_tokens,
    size_t max_tokens) {
  std::vector<std::string> pieces;
  std::string piece;
  for (const auto& word : split_words(sentence)) {
    std::string candidate = join(piece, word);
    if (!piece.empty() && count_tokens(candidate) > max_tokens) {
      pieces.push_back(piece);
      candidate = word;
    }
    piece = std::move(candidate);
  }
  pieces.push_back(piece);
  return pieces;
}

int64_t frame_budget(size_t num_tokens, double frame_rate) {
  return static_cast<int64_t>(
      std::ceil((static_cast<double>(num_tokens) / 3.0 + 2.0) * frame_rate));
}

uint16_t read_u16_le(const unsigned char* p) {
  return static_cast<uint16_t>(p[0] | (p[1] << 8));
}

float half_to_float(uint16_t h) {
  const uint32_t sign = (h & 0x8000u) << 16;
  const int exponent = (h >> 10) & 0x1f;
  const uint32_t mantissa = h & 0x3ffu;
  float magnitude;
  if (exponent == 0) {
    magnitude = std::ldexp(static_cast<float>(mantissa), -24);
  } else if (exponent == 31) {
    magnitude = mantissa ? NAN : INFINITY;
  } else {
    magnitude =
        std::ldexp(static_cast<float>(mantissa | 0x400u), exponent - 25);
  }
  uint32_t bits;
  std::memcpy(&bits, &magnitude, 4);
  bits |= sign;
  float out;
  std::memcpy(&out, &bits, 4);
  return out;
}

float bfloat16_to_float(uint16_t b) {
  const uint32_t bits = static_cast<uint32_t>(b) << 16;
  float out;
  std::memcpy(&out, &bits, 4);
  return out;
}
} // namespace

std::pair<std::string, int> prepare_text(const std::string& text) {
  const auto words = split_words(text);
  std::string out = join(words);
  if (out.empty()) {
    return {out, 3};
  }
  const int frames_after_eos = words.size() <= 4 ? 3 : 1;
  if (out[0] >= 'a' && out[0] <= 'z') {
    out[0] = static_cast<char>(out[0] - 'a' + 'A');
  }
  if (is_ascii_alnum(out.back())) {
    out += '.';
  }
  return {out, frames_after_eos};
}

std::vector<std::string> split_sentences(
    const std::string& text,
    const std::function<size_t(const std::string&)>& count_tokens,
    size_t max_tokens) {
  std::vector<std::string> chunks;
  std::string current;
  for (const auto& sentence : split_at_sentence_ends(join(split_words(text)))) {
    std::vector<std::string> pieces{sentence};
    if (count_tokens(sentence) > max_tokens) {
      pieces = split_long_sentence(sentence, count_tokens, max_tokens);
    }
    for (const auto& piece : pieces) {
      std::string candidate = join(current, piece);
      if (!current.empty() && count_tokens(candidate) > max_tokens) {
        chunks.push_back(current);
        candidate = piece;
      }
      current = std::move(candidate);
    }
  }
  if (!current.empty()) {
    chunks.push_back(current);
  }
  return chunks;
}

double NoiseGenerator::uniform() {
  state_ += 0x9e3779b97f4a7c15ull;
  uint64_t z = state_;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  z ^= z >> 31;
  return static_cast<double>(z >> 11) * 0x1.0p-53;
}

void NoiseGenerator::fill(float* out, size_t count, double stddev) {
  constexpr double kTwoPi = 6.283185307179586;
  for (size_t i = 0; i < count; i += 2) {
    const double radius = std::sqrt(-2.0 * std::log(1.0 - uniform()));
    const double angle = kTwoPi * uniform();
    out[i] = static_cast<float>(radius * std::cos(angle) * stddev);
    if (i + 1 < count) {
      out[i + 1] = static_cast<float>(radius * std::sin(angle) * stddev);
    }
  }
}

Runner::Runner(
    const std::string& model_path,
    const std::string& tokenizer_path,
    const std::string& data_path)
    : model_(model_path, data_path, Module::LoadMode::Mmap) {
  if (model_.load() != Error::Ok) {
    throw std::runtime_error("Could not load " + model_path);
  }
  if (tokenizer_.load(tokenizer_path) != tokenizers::Error::Ok) {
    throw std::runtime_error("Could not load tokenizer " + tokenizer_path);
  }
  const auto integer = [this](const char* name) {
    return execute(model_, name).at(0).toInt();
  };
  const auto real = [this](const char* name) {
    return execute(model_, name).at(0).toDouble();
  };
  if (integer("format_version") != kFormatVersion) {
    throw std::runtime_error("Unsupported Phonon program; re-export it");
  }
  sample_rate_ = integer("sample_rate");
  frame_samples_ = integer("frame_samples");
  dim_ = integer("dim");
  latent_dim_ = integer("latent_dim");
  max_context_ = integer("max_context");
  max_prefill_ = integer("max_prefill");
  frame_rate_ = real("frame_rate");
  eos_threshold_ = real("eos_threshold");
  temperature_ = real("temperature");
  bos_embeds_ = copy_floats(execute(model_, "bos_embeds").at(0));
  if (static_cast<int64_t>(bos_embeds_.size()) != dim_) {
    throw std::runtime_error("bos_embeds does not match dim");
  }
  noise_.assign(latent_dim_, 0.0f);
  for (const char* method :
       {"text_embed", "voice_embed", "flow_lm", "mimi_decode"}) {
    if (model_.load_method(method) != Error::Ok) {
      throw std::runtime_error(std::string("Could not load method ") + method);
    }
  }
  auto meta = model_.method_meta("mimi_decode");
  if (!meta.ok()) {
    throw std::runtime_error("Could not read mimi_decode metadata");
  }
  for (size_t i = 1; i < meta->num_inputs(); ++i) {
    auto info = meta->input_tensor_meta(i);
    if (!info.ok() || info->scalar_type() != ScalarType::Float) {
      throw std::runtime_error("mimi_decode state must be float32 tensors");
    }
    const auto sizes = info->sizes();
    mimi_state_sizes_.emplace_back(sizes.begin(), sizes.end());
  }
}

std::vector<uint64_t> Runner::tokenize(const std::string& text) const {
  auto tokens = tokenizer_.encode(text, /*bos=*/0, /*eos=*/0);
  if (!tokens.ok()) {
    throw std::runtime_error("Could not tokenize: " + text);
  }
  return std::move(tokens.get());
}

Voice Runner::load_voice(const std::string& path) const {
  std::ifstream file(path, std::ios::binary);
  uint64_t header_size = 0;
  if (!file.read(reinterpret_cast<char*>(&header_size), 8) ||
      header_size > (1u << 24)) {
    throw std::runtime_error("Not a safetensors file: " + path);
  }
  std::string header(header_size, '\0');
  file.read(header.data(), static_cast<std::streamsize>(header_size));
  const auto json = nlohmann::json::parse(header);
  if (!json.contains("speaker_wavs")) {
    throw std::runtime_error(path + ": expected a tensor named speaker_wavs");
  }
  const auto& entry = json.at("speaker_wavs");
  const auto dtype = entry.at("dtype").get<std::string>();
  const auto shape = entry.at("shape").get<std::vector<int64_t>>();
  const auto offsets = entry.at("data_offsets").get<std::vector<uint64_t>>();
  if (shape.size() != 3 || shape[0] != 1 || shape[1] != dim_ || shape[2] <= 0 ||
      shape[2] > max_prefill_) {
    throw std::runtime_error(
        path + ": speaker_wavs must be [1, " + std::to_string(dim_) +
        ", frames] with at most " + std::to_string(max_prefill_) + " frames");
  }
  const size_t element_size = dtype == "F32" ? 4 : 2;
  if (dtype != "F32" && dtype != "F16" && dtype != "BF16") {
    throw std::runtime_error(path + ": unsupported dtype " + dtype);
  }
  const int64_t frames = shape[2];
  const size_t count = static_cast<size_t>(dim_ * frames);
  if (offsets.size() != 2 || offsets[1] - offsets[0] != count * element_size) {
    throw std::runtime_error(path + ": inconsistent speaker_wavs size");
  }
  std::vector<unsigned char> raw(count * element_size);
  file.seekg(static_cast<std::streamoff>(8 + header_size + offsets[0]));
  if (!file.read(
          reinterpret_cast<char*>(raw.data()),
          static_cast<std::streamsize>(raw.size()))) {
    throw std::runtime_error(path + ": truncated file");
  }
  const auto element = [&](size_t k) -> float {
    const unsigned char* p = raw.data() + k * element_size;
    if (dtype == "F32") {
      float value;
      std::memcpy(&value, p, 4);
      return value;
    }
    return dtype == "F16" ? half_to_float(read_u16_le(p))
                          : bfloat16_to_float(read_u16_le(p));
  };
  Voice voice;
  voice.frames = frames;
  voice.latents.resize(count);
  for (int64_t d = 0; d < dim_; ++d) {
    for (int64_t t = 0; t < frames; ++t) {
      voice.latents[t * dim_ + d] = element(d * frames + t);
    }
  }
  return voice;
}

Runner::FlowOutput Runner::flow(const std::vector<float>& embeds, int64_t pos) {
  return flow(embeds.data(), static_cast<int64_t>(embeds.size()) / dim_, pos);
}

Runner::FlowOutput Runner::flow(const float* embeds, int64_t seq, int64_t pos) {
  int64_t input_pos = pos;
  auto embeds_tensor = from_blob(
      const_cast<float*>(embeds),
      {1, static_cast<SizesType>(seq), static_cast<SizesType>(dim_)});
  auto pos_tensor = from_blob(&input_pos, {1}, ScalarType::Long);
  auto noise_tensor =
      from_blob(noise_.data(), {1, static_cast<SizesType>(latent_dim_)});
  auto out =
      execute(model_, "flow_lm", {embeds_tensor, pos_tensor, noise_tensor});
  return {
      copy_floats(out.at(0)), float_data(out.at(1))[0], copy_floats(out.at(2))};
}

Synthesis Runner::synthesize(
    const std::string& text,
    const Voice& voice,
    double temperature,
    uint64_t seed,
    const std::function<void(const float*, size_t)>& on_audio) {
  if (voice.frames <= 0 || voice.frames > max_prefill_ ||
      static_cast<int64_t>(voice.latents.size()) != voice.frames * dim_) {
    throw std::invalid_argument("Invalid voice prompt");
  }
  NoiseGenerator noise(seed);
  const double stddev = std::sqrt(temperature < 0 ? temperature_ : temperature);

  auto voice_tensor = from_blob(
      const_cast<float*>(voice.latents.data()),
      {1, static_cast<SizesType>(voice.frames), static_cast<SizesType>(dim_)});
  const auto voice_embeds =
      copy_floats(execute(model_, "voice_embed", {voice_tensor}).at(0));
  std::fill(noise_.begin(), noise_.end(), 0.0f);
  flow(voice_embeds, 0);
  const int64_t voice_len = voice.frames;

  std::vector<std::vector<float>> state;
  for (const auto& sizes : mimi_state_sizes_) {
    int64_t numel = 1;
    for (auto s : sizes) {
      numel *= s;
    }
    state.emplace_back(numel, 0.0f);
  }

  Synthesis result;
  const auto count_tokens = [this](const std::string& s) {
    return tokenize(s).size();
  };
  for (const auto& chunk_text : split_sentences(
           prepare_text(text).first, count_tokens, kMaxChunkTokens)) {
    auto [prepared, frames_after_eos] = prepare_text(chunk_text);
    Chunk chunk{prepared, tokenize(prepared), 0};
    const auto num_tokens = static_cast<int64_t>(chunk.tokens.size());
    if (num_tokens == 0 || num_tokens > max_prefill_) {
      throw std::runtime_error(
          "Chunk has an unsupported token count: " + prepared);
    }
    std::vector<int64_t> ids(chunk.tokens.begin(), chunk.tokens.end());
    auto tokens_tensor = from_blob(
        ids.data(), {1, static_cast<SizesType>(num_tokens)}, ScalarType::Long);
    const auto text_embeds =
        copy_floats(execute(model_, "text_embed", {tokens_tensor}).at(0));
    std::fill(noise_.begin(), noise_.end(), 0.0f);
    int64_t pos = voice_len;
    flow(text_embeds, pos);
    pos += num_tokens;

    const int64_t budget = std::min(
        frame_budget(chunk.tokens.size(), frame_rate_), max_context_ - pos);
    std::vector<float> embeds = bos_embeds_;
    int countdown = -1;
    for (int64_t k = 0; k < budget; ++k) {
      noise.fill(noise_.data(), noise_.size(), stddev);
      auto step = flow(embeds, pos);
      embeds = std::move(step.next_embeds);
      ++pos;
      ++chunk.frames;

      // The tensors must outlive the EValues that point to them.
      std::vector<executorch::extension::TensorPtr> tensors;
      tensors.reserve(state.size() + 1);
      tensors.push_back(from_blob(
          step.latent.data(), {1, 1, static_cast<SizesType>(latent_dim_)}));
      for (size_t i = 0; i < state.size(); ++i) {
        tensors.push_back(from_blob(state[i].data(), mimi_state_sizes_[i]));
      }
      auto out = execute(
          model_,
          "mimi_decode",
          std::vector<EValue>(tensors.begin(), tensors.end()));
      const float* pcm = float_data(out.at(0));
      const size_t pcm_size = out.at(0).toTensor().numel();
      result.pcm.insert(result.pcm.end(), pcm, pcm + pcm_size);
      if (on_audio) {
        on_audio(pcm, pcm_size);
      }
      for (size_t i = 0; i < state.size(); ++i) {
        std::memcpy(
            state[i].data(), float_data(out.at(i + 1)), state[i].size() * 4);
      }

      if (step.eos_logit > eos_threshold_ && countdown < 0) {
        countdown = frames_after_eos;
      }
      if (countdown >= 0) {
        if (countdown == 0) {
          break;
        }
        --countdown;
      }
    }
    result.frames += chunk.frames;
    result.chunks.push_back(std::move(chunk));
  }
  return result;
}

} // namespace phonon
