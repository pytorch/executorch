/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "voxtral_tts_runner.h"
#include "wav_writer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/log.h>
#include <nlohmann/json.hpp>

namespace voxtral_tts {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::extension::Module;
using ::executorch::extension::TensorPtr;
using ::executorch::extension::from_blob;
using ::executorch::runtime::EValue;
using ::executorch::runtime::Error;

namespace {

using json = nlohmann::json;

int64_t read_metadata_int(Module& m, const char* name, int64_t fallback) {
  std::vector<EValue> empty;
  auto result = m.execute(name, empty);
  if (result.ok() && !result.get().empty()) {
    return result.get()[0].toInt();
  }
  return fallback;
}

bool has_method(Module& m, const char* name) {
  auto methods = m.method_names();
  if (!methods.ok()) {
    return false;
  }
  return methods.get().count(name) > 0;
}

json topk_logits(const float* logits, int64_t vocab_size, int k = 5) {
  std::vector<int64_t> indices(vocab_size);
  std::iota(indices.begin(), indices.end(), 0);
  auto cmp = [&](int64_t lhs, int64_t rhs) {
    return logits[lhs] > logits[rhs];
  };
  const int64_t topk = std::min<int64_t>(k, vocab_size);
  std::partial_sort(indices.begin(), indices.begin() + topk, indices.end(), cmp);

  json result = json::array();
  for (int64_t i = 0; i < topk; ++i) {
    result.push_back({
        {"id", indices[i]},
        {"logit", logits[indices[i]]},
    });
  }
  return result;
}

json waveform_stats(const std::vector<float>& samples) {
  json result = {
      {"num_samples", samples.size()},
      {"min", 0.0f},
      {"max", 0.0f},
      {"mean_abs", 0.0f},
      {"peak_abs", 0.0f},
  };
  if (samples.empty()) {
    return result;
  }

  float min_val = std::numeric_limits<float>::infinity();
  float max_val = -std::numeric_limits<float>::infinity();
  double sum_abs = 0.0;
  float peak_abs = 0.0f;
  for (float sample : samples) {
    min_val = std::min(min_val, sample);
    max_val = std::max(max_val, sample);
    sum_abs += std::abs(sample);
    peak_abs = std::max(peak_abs, std::abs(sample));
  }
  result["min"] = min_val;
  result["max"] = max_val;
  result["mean_abs"] = static_cast<float>(sum_abs / samples.size());
  result["peak_abs"] = peak_abs;
  return result;
}

void write_trace_json(const std::string& path, const json& trace) {
  std::ofstream file(path);
  ET_CHECK_MSG(file.is_open(), "Failed to open trace output: %s", path.c_str());
  file << trace.dump(2) << std::endl;
}

uint16_t read_u16(const unsigned char* ptr) {
  uint16_t value = 0;
  std::memcpy(&value, ptr, sizeof(value));
  return value;
}

uint32_t read_u32(const unsigned char* ptr) {
  uint32_t value = 0;
  std::memcpy(&value, ptr, sizeof(value));
  return value;
}

bool find_zip_entry(
    const std::vector<unsigned char>& file_data,
    const std::string& target_name,
    const unsigned char*& out_data,
    size_t& out_size) {
  if (file_data.size() < 22) {
    return false;
  }

  size_t eocd_pos = 0;
  bool found_eocd = false;
  const size_t lower_bound =
      file_data.size() > 65536 ? file_data.size() - 65536 : 0;
  for (size_t pos = file_data.size() - 22; pos > lower_bound; --pos) {
    if (read_u32(file_data.data() + pos) == 0x06054b50) {
      eocd_pos = pos;
      found_eocd = true;
      break;
    }
  }
  if (!found_eocd) {
    return false;
  }

  const uint32_t cd_offset = read_u32(file_data.data() + eocd_pos + 16);
  size_t pos = cd_offset;
  while (pos + 46 < file_data.size()) {
    if (read_u32(file_data.data() + pos) != 0x02014b50) {
      break;
    }

    const uint16_t compression = read_u16(file_data.data() + pos + 10);
    const uint32_t comp_size = read_u32(file_data.data() + pos + 20);
    const uint32_t uncomp_size = read_u32(file_data.data() + pos + 24);
    const uint16_t fname_len = read_u16(file_data.data() + pos + 28);
    const uint16_t extra_len = read_u16(file_data.data() + pos + 30);
    const uint16_t comment_len = read_u16(file_data.data() + pos + 32);
    const uint32_t local_offset = read_u32(file_data.data() + pos + 42);

    const char* fname = reinterpret_cast<const char*>(file_data.data() + pos + 46);
    if (std::string(fname, fname_len) == target_name) {
      if (compression != 0) {
        return false;
      }
      const uint16_t local_fname_len = read_u16(file_data.data() + local_offset + 26);
      const uint16_t local_extra_len = read_u16(file_data.data() + local_offset + 28);
      const size_t data_start = local_offset + 30 + local_fname_len + local_extra_len;
      const size_t entry_size = uncomp_size > 0 ? uncomp_size : comp_size;
      if (data_start + entry_size > file_data.size()) {
        return false;
      }
      out_data = file_data.data() + data_start;
      out_size = entry_size;
      return true;
    }

    pos += 46 + fname_len + extra_len + comment_len;
  }
  return false;
}

float bf16_to_float(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

void load_bf16_tensor_data(
    const uint16_t* bf16_data,
    size_t count,
    std::vector<float>& out_data) {
  out_data.resize(count);
  for (size_t i = 0; i < count; ++i) {
    out_data[i] = bf16_to_float(bf16_data[i]);
  }
}

bool load_pt_voice_tensor(
    const std::filesystem::path& path,
    int64_t dim,
    std::vector<float>& out_data,
    int64_t& out_frames) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    return false;
  }
  const auto file_size = static_cast<size_t>(file.tellg());
  file.seekg(0, std::ios::beg);
  std::vector<unsigned char> file_data(file_size);
  file.read(reinterpret_cast<char*>(file_data.data()), file_data.size());

  const char* candidate_paths[] = {"voice_embed/data/0", "archive/data/0", "data/0"};
  const unsigned char* tensor_data = nullptr;
  size_t tensor_size = 0;
  bool found = false;
  for (const char* candidate : candidate_paths) {
    if (find_zip_entry(file_data, candidate, tensor_data, tensor_size)) {
      found = true;
      break;
    }
  }
  if (!found || tensor_size % (static_cast<size_t>(dim) * sizeof(uint16_t)) != 0) {
    return false;
  }

  out_frames = static_cast<int64_t>(tensor_size / (static_cast<size_t>(dim) * sizeof(uint16_t)));
  load_bf16_tensor_data(
      reinterpret_cast<const uint16_t*>(tensor_data),
      static_cast<size_t>(out_frames) * static_cast<size_t>(dim),
      out_data);
  return true;
}

bool load_bin_voice_tensor(
    const std::filesystem::path& path,
    int64_t dim,
    int64_t expected_frames_hint,
    std::vector<float>& out_data,
    int64_t& out_frames) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    return false;
  }
  const auto file_size = static_cast<size_t>(file.tellg());
  file.seekg(0, std::ios::beg);
  std::vector<unsigned char> raw(file_size);
  file.read(reinterpret_cast<char*>(raw.data()), raw.size());

  const size_t bf16_row_bytes = static_cast<size_t>(dim) * sizeof(uint16_t);
  const size_t f32_row_bytes = static_cast<size_t>(dim) * sizeof(float);
  const bool matches_hint_bf16 =
      expected_frames_hint > 0 &&
      file_size == static_cast<size_t>(expected_frames_hint) * bf16_row_bytes;
  const bool matches_hint_f32 =
      expected_frames_hint > 0 &&
      file_size == static_cast<size_t>(expected_frames_hint) * f32_row_bytes;

  if (matches_hint_f32) {
    out_frames = expected_frames_hint;
    out_data.resize(static_cast<size_t>(out_frames) * static_cast<size_t>(dim));
    std::memcpy(out_data.data(), raw.data(), raw.size());
    return true;
  }

  if (matches_hint_bf16 || file_size % bf16_row_bytes == 0) {
    out_frames = static_cast<int64_t>(file_size / bf16_row_bytes);
    load_bf16_tensor_data(
        reinterpret_cast<const uint16_t*>(raw.data()),
        static_cast<size_t>(out_frames) * static_cast<size_t>(dim),
        out_data);
    return true;
  }

  if (file_size % f32_row_bytes == 0) {
    out_frames = static_cast<int64_t>(file_size / f32_row_bytes);
    out_data.resize(static_cast<size_t>(out_frames) * static_cast<size_t>(dim));
    std::memcpy(out_data.data(), raw.data(), raw.size());
    return true;
  }
  return false;
}

} // namespace

VoxtralTTSRunner::VoxtralTTSRunner(
    const std::string& model_path,
    const std::string& codec_path,
    const std::string& tokenizer_path)
    : rng_(42),
      asset_root_dir_(std::filesystem::path(tokenizer_path).parent_path()),
      model_path_(model_path) {
  model_ = std::make_unique<Module>(model_path, Module::LoadMode::Mmap);
  ET_CHECK_MSG(model_->load() == Error::Ok, "Failed to load model.");

  codec_ = std::make_unique<Module>(codec_path, Module::LoadMode::Mmap);
  ET_CHECK_MSG(codec_->load() == Error::Ok, "Failed to load codec decoder.");

  tokenizer_ = ::executorch::extension::llm::load_tokenizer(tokenizer_path);
  ET_CHECK_MSG(tokenizer_ != nullptr, "Failed to load tokenizer.");

  load_metadata();
  warmup();
}

void VoxtralTTSRunner::set_trace_output_path(
    const std::string& trace_output_path) {
  trace_output_path_ = trace_output_path;
}

void VoxtralTTSRunner::set_seed(uint32_t seed) {
  seed_ = seed;
  rng_.seed(seed_);
}

void VoxtralTTSRunner::reload_stateful_model() {
  model_ = std::make_unique<Module>(model_path_, Module::LoadMode::Mmap);
  ET_CHECK_MSG(model_->load() == Error::Ok, "Failed to reload model.");
  load_metadata();
}

void VoxtralTTSRunner::load_metadata() {
  sample_rate_ = read_metadata_int(*model_, "sample_rate", 24000);
  n_decoding_steps_ = read_metadata_int(*model_, "n_decoding_steps", 7);
  int64_t alpha_x100 = read_metadata_int(*model_, "cfg_alpha_x100", 120);
  cfg_alpha_ = static_cast<float>(alpha_x100) / 100.0f;
  n_acoustic_codebook_ = read_metadata_int(*model_, "n_acoustic_codebook", 36);
  acoustic_levels_ = read_metadata_int(*model_, "acoustic_levels", 21);
  n_special_tokens_ = read_metadata_int(*model_, "n_special_tokens", 2);
  vocab_size_ = read_metadata_int(*model_, "vocab_size", 131072);
  max_seq_len_ = read_metadata_int(*model_, "max_seq_len", 4096);
  dim_ = read_metadata_int(*model_, "dim", 3072);
  downsample_factor_ = read_metadata_int(*model_, "downsample_factor", 1920);
  n_codebooks_ = read_metadata_int(*model_, "n_codebooks", 37);
  end_audio_code_ = read_metadata_int(*model_, "end_audio_code", 1);
  empty_audio_code_ = read_metadata_int(*model_, "empty_audio_code", 0);
  audio_token_id_ = read_metadata_int(*model_, "audio_token_id", 24);
  begin_audio_token_id_ =
      read_metadata_int(*model_, "begin_audio_token_id", 25);
  text_to_audio_token_id_ =
      read_metadata_int(*model_, "text_to_audio_token_id", 36);
  repeat_audio_text_token_id_ =
      read_metadata_int(*model_, "repeat_audio_text_token_id", 35);
  voice_embed_len_ = read_metadata_int(*model_, "voice_embed_len", 147);

  is_streaming_ = read_metadata_int(*model_, "streaming", 0) != 0;
  streaming_chunk_frames_ =
      read_metadata_int(*model_, "streaming_chunk_frames", 25);
  streaming_initial_chunk_ =
      read_metadata_int(*model_, "streaming_initial_chunk", 5);
  streaming_left_context_ =
      read_metadata_int(*model_, "streaming_left_context", 25);

  max_codec_frames_ = read_metadata_int(*codec_, "max_codec_frames", 256);
  codec_supports_exact_frames_ = has_method(*codec_, "codec_supports_exact_frames")
      ? (read_metadata_int(*codec_, "codec_supports_exact_frames", 0) != 0)
      : false;

  std::cout << "Model config: dim=" << dim_ << " voice_embed_len="
            << voice_embed_len_ << " audio_tok=" << audio_token_id_
            << " begin_audio=" << begin_audio_token_id_
            << " max_seq=" << max_seq_len_ << " codec_frames="
            << max_codec_frames_ << std::endl;
}

std::filesystem::path VoxtralTTSRunner::resolve_voice_path(
    const std::string& voice_path) const {
  const std::string requested = voice_path.empty() ? "neutral_female" : voice_path;
  std::filesystem::path candidate(requested);
  if (std::filesystem::exists(candidate)) {
    return candidate;
  }

  const auto voice_dir = asset_root_dir_ / "voice_embedding";
  if (candidate.has_extension()) {
    auto local_candidate = voice_dir / candidate.filename();
    if (std::filesystem::exists(local_candidate)) {
      return local_candidate;
    }
    return candidate;
  }

  for (const char* ext : {".pt", ".bin"}) {
    auto local_candidate = voice_dir / (requested + ext);
    if (std::filesystem::exists(local_candidate)) {
      return local_candidate;
    }
  }

  return voice_dir / (requested + ".pt");
}

void VoxtralTTSRunner::load_voice_embedding(const std::string& voice_path) {
  voice_embed_data_.clear();
  runtime_voice_embed_len_ = 0;

  const auto resolved_path = resolve_voice_path(voice_path);
  if (!std::filesystem::exists(resolved_path)) {
    if (voice_path.empty()) {
      std::cout << "No default voice embedding found at " << resolved_path
                << ", continuing without voice conditioning." << std::endl;
      return;
    }
    ET_CHECK_MSG(false, "Failed to open voice embedding: %s",
                 resolved_path.string().c_str());
  }

  bool ok = false;
  if (resolved_path.extension() == ".pt") {
    ok = load_pt_voice_tensor(
        resolved_path, dim_, voice_embed_data_, runtime_voice_embed_len_);
  } else {
    int64_t expected_frames_hint = voice_embed_len_;
    auto pt_peer = resolved_path;
    pt_peer.replace_extension(".pt");
    if (std::filesystem::exists(pt_peer)) {
      std::vector<float> peer_voice_data;
      int64_t peer_frames = 0;
      if (load_pt_voice_tensor(pt_peer, dim_, peer_voice_data, peer_frames)) {
        expected_frames_hint = peer_frames;
      }
    }
    ok = load_bin_voice_tensor(
        resolved_path,
        dim_,
        expected_frames_hint,
        voice_embed_data_,
        runtime_voice_embed_len_);
  }
  ET_CHECK_MSG(
      ok,
      "Failed to load voice embedding from %s",
      resolved_path.string().c_str());

  std::cout << "Loaded voice embedding: " << runtime_voice_embed_len_ << " x "
            << dim_ << " from " << resolved_path << std::endl;
}

int64_t VoxtralTTSRunner::sample_semantic_code(
    const float* logits,
    int64_t vocab_size,
    float temperature) {
  if (temperature <= 0.0f) {
    int64_t best = 0;
    float best_val = logits[0];
    for (int64_t i = 1; i < vocab_size; ++i) {
      if (logits[i] > best_val) {
        best_val = logits[i];
        best = i;
      }
    }
    return best;
  }
  float max_val = *std::max_element(logits, logits + vocab_size);
  std::vector<float> probs(vocab_size);
  float sum = 0;
  for (int64_t i = 0; i < vocab_size; ++i) {
    probs[i] = std::exp((logits[i] - max_val) / temperature);
    sum += probs[i];
  }
  for (auto& p : probs) p /= sum;

  std::discrete_distribution<int64_t> dist(probs.begin(), probs.end());
  return dist(rng_);
}

void VoxtralTTSRunner::warmup() {
  std::cout << "Warming up..." << std::endl;
  int dim = static_cast<int>(dim_);
  int n_aco = static_cast<int>(n_acoustic_codebook_);
  int n_cb = static_cast<int>(n_codebooks_);
  int mcf = static_cast<int>(max_codec_frames_);

  int64_t tok_data = 1;
  auto tok_t = from_blob(&tok_data, {1, 1}, ScalarType::Long);
  auto token_embed_result =
      model_->execute("token_embedding", std::vector<EValue>{*tok_t});
  ET_CHECK_MSG(token_embed_result.ok(), "token_embedding warmup failed");

  std::vector<int64_t> audio_code_data(n_cb, 0);
  auto audio_codes_t =
      from_blob(audio_code_data.data(), {1, n_cb, 1}, ScalarType::Long);
  auto audio_embed_result =
      model_->execute("audio_token_embedding", std::vector<EValue>{*audio_codes_t});
  ET_CHECK_MSG(audio_embed_result.ok(), "audio_token_embedding warmup failed");

  std::vector<float> embed_data(dim, 0.0f);
  // Avoid warming the stateful decoder because the Module API does not expose
  // a cache reset; a dummy prefill would pollute the first real synthesis.
  auto hid_t = from_blob(embed_data.data(), {1, dim}, ScalarType::Float);
  auto semantic_result =
      model_->execute("semantic_head", std::vector<EValue>{*hid_t});
  ET_CHECK_MSG(semantic_result.ok(), "semantic_head warmup failed");

  std::vector<float> xt_data(n_aco, 0.0f);
  auto xt_t = from_blob(xt_data.data(), {1, n_aco}, ScalarType::Float);
  int64_t tidx_data = 0;
  auto ti_t = from_blob(&tidx_data, {1}, ScalarType::Long);
  auto hv_t = from_blob(embed_data.data(), {1, dim}, ScalarType::Float);
  auto velocity_result = model_->execute(
      "predict_velocity", std::vector<EValue>{*xt_t, *ti_t, *hv_t});
  ET_CHECK_MSG(velocity_result.ok(), "predict_velocity warmup failed");

  std::vector<int64_t> code_data(n_cb * mcf, 0);
  auto codes_t = from_blob(code_data.data(), {1, n_cb, mcf}, ScalarType::Long);
  auto codec_result = codec_->execute("forward", std::vector<EValue>{*codes_t});
  ET_CHECK_MSG(codec_result.ok(), "codec warmup failed");

  std::cout << "Warmup complete." << std::endl;
}

std::vector<int64_t> VoxtralTTSRunner::tokenize(const std::string& text) {
  auto encoded = tokenizer_->encode(text, /*bos=*/0, /*eos=*/0);
  ET_CHECK_MSG(encoded.ok(), "Tokenizer encode failed");
  std::vector<int64_t> result;
  result.reserve(encoded.get().size());
  for (auto id : encoded.get()) {
    result.push_back(static_cast<int64_t>(id));
  }
  return result;
}

void VoxtralTTSRunner::build_prompt(
    const std::string& text,
    std::vector<int64_t>& token_ids,
    int& voice_start,
    int& voice_len) {
  // Match mistral_common encode_speech_request():
  // [BOS] [BEGIN_AUDIO] [AUDIO]*N [TEXT_TO_AUDIO] {text_tokens}
  // [AUDIO_TO_TEXT] [BEGIN_AUDIO]
  auto text_tokens = tokenize(text);

  token_ids.clear();
  token_ids.push_back(1); // BOS
  token_ids.push_back(begin_audio_token_id_); // [BEGIN_AUDIO]

  voice_start = static_cast<int>(token_ids.size());
  voice_len = static_cast<int>(runtime_voice_embed_len_);
  for (int i = 0; i < voice_len; ++i) {
    token_ids.push_back(audio_token_id_); // [AUDIO] placeholder
  }

  token_ids.push_back(text_to_audio_token_id_);
  for (auto t : text_tokens) {
    token_ids.push_back(t);
  }
  token_ids.push_back(repeat_audio_text_token_id_); // [REPEAT_AUDIO_TEXT]
  token_ids.push_back(begin_audio_token_id_); // [BEGIN_AUDIO]

  std::cout << "Prompt: " << token_ids.size() << " tokens (voice_start="
            << voice_start << " voice_len=" << voice_len << " text_tokens="
            << text_tokens.size() << ")" << std::endl;
}

void VoxtralTTSRunner::synthesize_offline(
    const std::string& text,
    const std::string& voice_path,
    const std::string& output_path,
    float temperature,
    int max_new_tokens) {
  auto start = std::chrono::high_resolution_clock::now();
  int dim = static_cast<int>(dim_);
  int n_aco = static_cast<int>(n_acoustic_codebook_);
  int n_cb = static_cast<int>(n_codebooks_);
  const bool capture_trace = !trace_output_path_.empty();
  json trace;

  reload_stateful_model();
  rng_.seed(seed_);
  dim = static_cast<int>(dim_);
  n_aco = static_cast<int>(n_acoustic_codebook_);
  n_cb = static_cast<int>(n_codebooks_);

  load_voice_embedding(voice_path);
  const auto resolved_voice_path = resolve_voice_path(voice_path);

  std::vector<int64_t> token_ids;
  int voice_start, voice_len;
  build_prompt(text, token_ids, voice_start, voice_len);
  int prompt_len = static_cast<int>(token_ids.size());
  if (capture_trace) {
    trace = {
        {"mode", "runner_exported"},
        {"text", text},
        {"voice_path", resolved_voice_path.string()},
        {"seed", seed_},
        {"prompt_token_ids", token_ids},
        {"voice_start", voice_start},
        {"voice_len", voice_len},
        {"seed_step_applied", false},
        {"frames", json::array()},
    };
  }

  // Embed all tokens
  auto tok_t = from_blob(token_ids.data(), {1, prompt_len}, ScalarType::Long);
  auto embed_result =
      model_->execute("token_embedding", std::vector<EValue>{*tok_t});
  ET_CHECK_MSG(embed_result.ok(), "token_embedding failed");
  auto embeds = embed_result.get()[0].toTensor();
  float* embed_ptr = embeds.mutable_data_ptr<float>();

  // Splice voice embedding into [AUDIO] positions
  if (!voice_embed_data_.empty()) {
    for (int i = 0; i < voice_len; ++i) {
      int pos = voice_start + i;
      std::memcpy(
          embed_ptr + pos * dim,
          voice_embed_data_.data() + i * dim,
          dim * sizeof(float));
    }
    std::cout << "Voice embedding spliced at positions " << voice_start
              << ".." << (voice_start + voice_len - 1) << std::endl;
  }

  // Prefill decoder with combined embeddings
  std::vector<int64_t> pos_vec(prompt_len);
  std::iota(pos_vec.begin(), pos_vec.end(), 0);
  auto pos_t = from_blob(pos_vec.data(), {prompt_len}, ScalarType::Long);

  auto emb_t = from_blob(embed_ptr, {1, prompt_len, dim}, ScalarType::Float);
  auto dec_result =
      model_->execute("text_decoder", std::vector<EValue>{*emb_t, *pos_t});
  ET_CHECK_MSG(dec_result.ok(), "text_decoder prefill failed");

  auto hidden_out = dec_result.get()[0].toTensor();
  std::vector<float> hidden_state(dim);
  std::memcpy(
      hidden_state.data(),
      hidden_out.mutable_data_ptr<float>() + (prompt_len - 1) * dim,
      static_cast<size_t>(dim) * sizeof(float));

  std::vector<float> prefill_hidden(hidden_state);

  std::vector<int64_t> seed_token{audio_token_id_};
  auto seed_tok_t = from_blob(seed_token.data(), {1, 1}, ScalarType::Long);
  auto seed_embed_result =
      model_->execute("token_embedding", std::vector<EValue>{*seed_tok_t});
  ET_CHECK_MSG(seed_embed_result.ok(), "token_embedding seed step failed");
  auto seed_embed = seed_embed_result.get()[0].toTensor();

  int64_t seed_pos_val = prompt_len;
  auto seed_pos_t = from_blob(&seed_pos_val, {1}, ScalarType::Long);
  auto seed_emb_t = from_blob(
      seed_embed.mutable_data_ptr<float>(), {1, 1, dim}, ScalarType::Float);
  auto seed_decode_result =
      model_->execute("text_decoder", std::vector<EValue>{*seed_emb_t, *seed_pos_t});
  ET_CHECK_MSG(seed_decode_result.ok(), "text_decoder seed step failed");
  std::memcpy(
      hidden_state.data(),
      seed_decode_result.get()[0].toTensor().mutable_data_ptr<float>(),
      static_cast<size_t>(dim) * sizeof(float));
  if (capture_trace) {
    trace["prefill_hidden"] = prefill_hidden;
    trace["frame0_hidden"] = hidden_state;
    trace["seed_hidden"] = hidden_state;
    trace["seed_position"] = prompt_len;
    trace["frame0_position"] = prompt_len;
    trace["seed_step_applied"] = true;
  }

  // Autoregressive decode
  std::vector<std::vector<int64_t>> frame_codes;
  int64_t cur_pos = prompt_len + 1;
  std::normal_distribution<float> normal_dist(0.0f, 1.0f);

  std::vector<float> timesteps(n_decoding_steps_ + 1);
  for (int i = 0; i <= n_decoding_steps_; ++i) {
    timesteps[i] =
        static_cast<float>(i) / static_cast<float>(n_decoding_steps_);
  }

  for (int frame = 0; frame < max_new_tokens && cur_pos < max_seq_len_;
       ++frame) {
    auto h_t = from_blob(hidden_state.data(), {1, dim}, ScalarType::Float);
    auto sem_r =
        model_->execute("semantic_head", std::vector<EValue>{*h_t});
    ET_CHECK_MSG(sem_r.ok(), "semantic_head failed");

    auto sem_t = sem_r.get()[0].toTensor();
    int64_t sem_vocab = sem_t.numel();
    json semantic_topk = json::array();
    if (capture_trace && frame < 3) {
      semantic_topk = topk_logits(sem_t.data_ptr<float>(), sem_vocab);
    }
    int64_t semantic_code = sample_semantic_code(
        sem_t.data_ptr<float>(), sem_vocab, temperature);

    if (semantic_code == end_audio_code_) {
      if (capture_trace && frame < 3) {
        trace["frames"].push_back({
            {"frame", frame},
            {"hidden_norm_before_frame",
             std::sqrt(std::inner_product(
                 hidden_state.begin(),
                 hidden_state.end(),
                 hidden_state.begin(),
                 0.0f))},
            {"semantic_code", semantic_code},
            {"semantic_topk", semantic_topk},
            {"full_codes", json::array()},
            {"end_audio", true},
        });
      }
      if (capture_trace) {
        trace["end_audio_at_frame"] = frame;
      }
      std::cout << "END_AUDIO at frame " << frame << std::endl;
      break;
    }

    // Flow matching ODE (7 steps with CFG)
    std::vector<float> x(n_aco);
    for (auto& v : x) {
      v = normal_dist(rng_);
    }
    std::vector<float> zeros(dim, 0.0f);

    for (int step = 0; step < n_decoding_steps_; ++step) {
      float dt = timesteps[step + 1] - timesteps[step];
      int64_t tidx_val = step;

      auto xt1 = from_blob(x.data(), {1, n_aco}, ScalarType::Float);
      auto ti1 = from_blob(&tidx_val, {1}, ScalarType::Long);
      auto hc = from_blob(hidden_state.data(), {1, dim}, ScalarType::Float);
      auto vc = model_->execute(
          "predict_velocity", std::vector<EValue>{*xt1, *ti1, *hc});
      ET_CHECK_MSG(vc.ok(), "predict_velocity (cond) failed");
      std::vector<float> v_cond(n_aco);
      std::memcpy(
          v_cond.data(),
          vc.get()[0].toTensor().mutable_data_ptr<float>(),
          static_cast<size_t>(n_aco) * sizeof(float));

      auto xt2 = from_blob(x.data(), {1, n_aco}, ScalarType::Float);
      auto ti2 = from_blob(&tidx_val, {1}, ScalarType::Long);
      auto hu = from_blob(zeros.data(), {1, dim}, ScalarType::Float);
      auto vu = model_->execute(
          "predict_velocity", std::vector<EValue>{*xt2, *ti2, *hu});
      ET_CHECK_MSG(vu.ok(), "predict_velocity (uncond) failed");
      float* v_uncond = vu.get()[0].toTensor().mutable_data_ptr<float>();

      for (int j = 0; j < n_aco; ++j) {
        float v =
            cfg_alpha_ * v_cond[j] + (1.0f - cfg_alpha_) * v_uncond[j];
        x[j] += v * dt;
      }
    }

    // Quantize acoustic codes
    std::vector<int64_t> codes(n_codebooks_);
    codes[0] = semantic_code;
    float x_min = std::numeric_limits<float>::infinity();
    float x_max = -std::numeric_limits<float>::infinity();
    for (int j = 0; j < n_aco; ++j) {
      float clamped = std::clamp(x[j], -1.0f, 1.0f);
      x_min = std::min(x_min, clamped);
      x_max = std::max(x_max, clamped);
      float scaled = ((clamped + 1.0f) / 2.0f) *
                     static_cast<float>(acoustic_levels_ - 1);
      codes[j + 1] =
          static_cast<int64_t>(std::round(scaled)) + n_special_tokens_;
    }
    frame_codes.push_back(codes);
    if (capture_trace && frame == 0) {
      trace["frame0_full_codes"] = codes;
    }
    if (capture_trace && frame < 3) {
      trace["frames"].push_back({
          {"frame", frame},
          {"hidden_norm_before_frame",
           std::sqrt(std::inner_product(
               hidden_state.begin(),
               hidden_state.end(),
               hidden_state.begin(),
               0.0f))},
          {"semantic_code", semantic_code},
          {"semantic_topk", semantic_topk},
          {"full_codes", codes},
          {"x_min", x_min},
          {"x_max", x_max},
      });
    }

    // Feed the generated multi-codebook frame back through the learned
    // audio-token embedding path instead of the generic [AUDIO] placeholder.
    auto next_codes =
        from_blob(codes.data(), {1, n_cb, 1}, ScalarType::Long);
    auto ne =
        model_->execute("audio_token_embedding", std::vector<EValue>{*next_codes});
    ET_CHECK_MSG(ne.ok(), "audio_token_embedding (next) failed");
    auto next_embeds = ne.get()[0].toTensor();
    if (capture_trace && frame == 0) {
      std::vector<float> first_audio_embed(dim);
      std::memcpy(
          first_audio_embed.data(),
          next_embeds.mutable_data_ptr<float>(),
          static_cast<size_t>(dim) * sizeof(float));
      trace["frame0_audio_embed"] = first_audio_embed;
    }

    int64_t next_pos_val = cur_pos;
    auto np = from_blob(&next_pos_val, {1}, ScalarType::Long);
    auto next_emb = from_blob(
        next_embeds.mutable_data_ptr<float>(), {1, 1, dim},
        ScalarType::Float);
    auto nd =
        model_->execute("text_decoder", std::vector<EValue>{*next_emb, *np});
    ET_CHECK_MSG(nd.ok(), "text_decoder (next) failed");
    std::memcpy(
        hidden_state.data(),
        nd.get()[0].toTensor().mutable_data_ptr<float>(),
        static_cast<size_t>(dim) * sizeof(float));
    if (capture_trace && frame == 0) {
      trace["frame1_position"] = cur_pos;
      trace["frame1_hidden"] = hidden_state;
    }
    cur_pos++;

    if ((frame + 1) % 25 == 0) {
      float audio_sec = static_cast<float>((frame + 1) * downsample_factor_) /
                        static_cast<float>(sample_rate_);
      std::cout << "  Frame " << (frame + 1) << " (" << audio_sec
                << "s audio)" << std::endl;
    }
  }

  auto gen_end = std::chrono::high_resolution_clock::now();

  if (frame_codes.empty()) {
    if (capture_trace) {
      trace["generated_frames"] = 0;
      trace["waveform"] = waveform_stats({});
      write_trace_json(trace_output_path_, trace);
      std::cout << "Wrote trace JSON: " << trace_output_path_ << std::endl;
    }
    std::cerr << "No audio frames generated." << std::endl;
    return;
  }

  int64_t total_frames = static_cast<int64_t>(frame_codes.size());
  float audio_duration = static_cast<float>(total_frames * downsample_factor_) /
                         static_cast<float>(sample_rate_);
  auto gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    gen_end - start)
                    .count();

  std::cout << "Generated " << total_frames << " frames (" << audio_duration
            << "s audio) in " << gen_ms << "ms" << std::endl;
  std::cout << "RTF: "
            << (static_cast<float>(gen_ms) / 1000.0f) / audio_duration
            << std::endl;

  std::vector<float> decoded_samples;
  decode_codes_to_wav(
      frame_codes,
      output_path,
      capture_trace ? &decoded_samples : nullptr);
  if (capture_trace) {
    trace["generated_frames"] = total_frames;
    trace["waveform"] = waveform_stats(decoded_samples);
    write_trace_json(trace_output_path_, trace);
    std::cout << "Wrote trace JSON: " << trace_output_path_ << std::endl;
  }

  auto total_end = std::chrono::high_resolution_clock::now();
  auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      total_end - start)
                      .count();
  std::cout << "Total time: " << total_ms << "ms" << std::endl;
}

void VoxtralTTSRunner::decode_codes_to_wav(
    const std::vector<std::vector<int64_t>>& frame_codes,
    const std::string& output_path,
    std::vector<float>* out_samples) {
  int64_t n_frames = static_cast<int64_t>(frame_codes.size());

  std::vector<float> all_samples;
  for (int64_t s = 0; s < n_frames; s += max_codec_frames_) {
    int64_t e = std::min(s + max_codec_frames_, n_frames);
    std::vector<float> chunk_samples;
    decode_code_window(frame_codes, s, e, chunk_samples);
    all_samples.insert(
        all_samples.end(), chunk_samples.begin(), chunk_samples.end());
  }

  WavWriter wav(output_path, static_cast<int>(sample_rate_));
  if (!wav.IsOpen()) {
    std::cerr << "Failed to open output: " << output_path << std::endl;
    return;
  }
  wav.Write(all_samples.data(), all_samples.size());
  wav.Close();
  if (out_samples != nullptr) {
    *out_samples = all_samples;
  }
  std::cout << "Wrote " << all_samples.size() << " samples to " << output_path
            << std::endl;
}

void VoxtralTTSRunner::decode_code_window(
    const std::vector<std::vector<int64_t>>& frame_codes,
    int64_t start_frame,
    int64_t end_frame,
    std::vector<float>& out_samples) {
  int64_t window_frames = end_frame - start_frame;
  ET_CHECK_MSG(window_frames > 0, "codec decode window must be non-empty");
  int n_cb = static_cast<int>(n_codebooks_);
  int mcf = static_cast<int>(max_codec_frames_);
  ET_CHECK_MSG(
      window_frames <= max_codec_frames_,
      "codec decode window exceeds exported maximum");

  auto build_code_tensor = [&](int64_t target_frames) {
    std::vector<int64_t> code_data(
        static_cast<size_t>(n_cb) * static_cast<size_t>(target_frames), 0);
    for (int64_t f = 0; f < window_frames; ++f) {
      for (int64_t c = 0; c < n_codebooks_; ++c) {
        code_data[c * target_frames + f] = frame_codes[start_frame + f][c];
      }
    }
    return code_data;
  };

  auto copy_waveform = [&](const auto& exec_result) {
    auto waveform = exec_result.get()[0].toTensor();
    float* wav_ptr = waveform.template mutable_data_ptr<float>();
    int64_t valid_samples = window_frames * downsample_factor_;
    int64_t total_samples = waveform.numel();
    valid_samples = std::min(valid_samples, total_samples);
    out_samples.assign(wav_ptr, wav_ptr + valid_samples);
  };

  const bool try_exact =
      codec_supports_exact_frames_ || window_frames == max_codec_frames_;
  if (try_exact) {
    auto code_data = build_code_tensor(window_frames);
    auto codes_t =
        from_blob(
            code_data.data(),
            {1, n_cb, static_cast<int>(window_frames)},
            ScalarType::Long);
    auto exact_result =
        codec_->execute("forward", std::vector<EValue>{*codes_t});
    if (exact_result.ok()) {
      copy_waveform(exact_result);
      return;
    }
  }

  auto padded_code_data = build_code_tensor(mcf);
  auto padded_codes_t =
      from_blob(padded_code_data.data(), {1, n_cb, mcf}, ScalarType::Long);
  auto padded_result =
      codec_->execute("forward", std::vector<EValue>{*padded_codes_t});
  ET_CHECK_MSG(padded_result.ok(), "codec decode failed");
  copy_waveform(padded_result);
}

void VoxtralTTSRunner::synthesize_streaming(
    const std::string& text,
    const std::string& voice_path,
    const std::string& output_path,
    AudioChunkCallback callback,
    float temperature,
    int max_new_tokens) {
  auto start_time = std::chrono::high_resolution_clock::now();
  int dim = static_cast<int>(dim_);
  int n_aco = static_cast<int>(n_acoustic_codebook_);
  int n_cb = static_cast<int>(n_codebooks_);

  reload_stateful_model();
  rng_.seed(seed_);
  dim = static_cast<int>(dim_);
  n_aco = static_cast<int>(n_acoustic_codebook_);
  n_cb = static_cast<int>(n_codebooks_);

  load_voice_embedding(voice_path);

  std::vector<int64_t> token_ids;
  int voice_start, voice_len;
  build_prompt(text, token_ids, voice_start, voice_len);
  int prompt_len = static_cast<int>(token_ids.size());

  // Embed + splice voice
  auto tok_t = from_blob(token_ids.data(), {1, prompt_len}, ScalarType::Long);
  auto embed_result =
      model_->execute("token_embedding", std::vector<EValue>{*tok_t});
  ET_CHECK_MSG(embed_result.ok(), "token_embedding failed");
  auto embeds = embed_result.get()[0].toTensor();
  float* embed_ptr = embeds.mutable_data_ptr<float>();

  if (!voice_embed_data_.empty()) {
    for (int i = 0; i < voice_len; ++i) {
      std::memcpy(
          embed_ptr + (voice_start + i) * dim,
          voice_embed_data_.data() + i * dim,
          dim * sizeof(float));
    }
  }

  // Prefill
  std::vector<int64_t> pos_vec(prompt_len);
  std::iota(pos_vec.begin(), pos_vec.end(), 0);
  auto pos_t = from_blob(pos_vec.data(), {prompt_len}, ScalarType::Long);
  auto emb_t = from_blob(embed_ptr, {1, prompt_len, dim}, ScalarType::Float);
  auto dec_result =
      model_->execute("text_decoder", std::vector<EValue>{*emb_t, *pos_t});
  ET_CHECK_MSG(dec_result.ok(), "text_decoder prefill failed");

  auto hidden_out = dec_result.get()[0].toTensor();
  std::vector<float> hidden_state(dim);
  std::memcpy(
      hidden_state.data(),
      hidden_out.mutable_data_ptr<float>() + (prompt_len - 1) * dim,
      static_cast<size_t>(dim) * sizeof(float));

  std::vector<int64_t> seed_token{audio_token_id_};
  auto seed_tok_t = from_blob(seed_token.data(), {1, 1}, ScalarType::Long);
  auto seed_embed_result =
      model_->execute("token_embedding", std::vector<EValue>{*seed_tok_t});
  ET_CHECK_MSG(seed_embed_result.ok(), "token_embedding seed step failed");
  auto seed_embed = seed_embed_result.get()[0].toTensor();

  int64_t seed_pos_val = prompt_len;
  auto seed_pos_t = from_blob(&seed_pos_val, {1}, ScalarType::Long);
  auto seed_emb_t = from_blob(
      seed_embed.mutable_data_ptr<float>(), {1, 1, dim}, ScalarType::Float);
  auto seed_decode_result =
      model_->execute("text_decoder", std::vector<EValue>{*seed_emb_t, *seed_pos_t});
  ET_CHECK_MSG(seed_decode_result.ok(), "text_decoder seed step failed");
  std::memcpy(
      hidden_state.data(),
      seed_decode_result.get()[0].toTensor().mutable_data_ptr<float>(),
      static_cast<size_t>(dim) * sizeof(float));

  std::vector<std::vector<int64_t>> frame_codes;
  int64_t cur_pos = prompt_len + 1;
  int64_t emitted_frames = 0;
  std::normal_distribution<float> normal_dist(0.0f, 1.0f);

  std::vector<float> timesteps(n_decoding_steps_ + 1);
  for (int i = 0; i <= n_decoding_steps_; ++i) {
    timesteps[i] =
        static_cast<float>(i) / static_cast<float>(n_decoding_steps_);
  }

  WavWriter wav(output_path, static_cast<int>(sample_rate_));
  ET_CHECK_MSG(wav.IsOpen(), "Failed to open WAV output");

  auto emit_ready_audio = [&]() {
    int64_t total = static_cast<int64_t>(frame_codes.size());
    int64_t pending = total - emitted_frames;
    int64_t chunk_threshold = (emitted_frames == 0)
                                  ? streaming_initial_chunk_
                                  : streaming_chunk_frames_;
    if (pending < chunk_threshold)
      return;

    int64_t decode_start =
        std::max(int64_t(0), emitted_frames - streaming_left_context_);
    int64_t crop_frames = emitted_frames - decode_start;

    std::vector<float> chunk_samples;
    decode_code_window(frame_codes, decode_start, total, chunk_samples);

    int64_t crop_samples = crop_frames * downsample_factor_;
    if (crop_samples < static_cast<int64_t>(chunk_samples.size())) {
      float* new_start = chunk_samples.data() + crop_samples;
      std::size_t new_count = chunk_samples.size() - crop_samples;
      wav.Write(new_start, new_count);
      if (callback)
        callback(new_start, new_count);
    }
    emitted_frames = total;
  };

  for (int frame = 0; frame < max_new_tokens && cur_pos < max_seq_len_;
       ++frame) {
    auto h_t = from_blob(hidden_state.data(), {1, dim}, ScalarType::Float);
    auto sem_r =
        model_->execute("semantic_head", std::vector<EValue>{*h_t});
    ET_CHECK_MSG(sem_r.ok(), "semantic_head failed");

    auto sem_t = sem_r.get()[0].toTensor();
    int64_t sem_vocab = sem_t.numel();
    int64_t semantic_code = sample_semantic_code(
        sem_t.data_ptr<float>(), sem_vocab, temperature);

    if (semantic_code == end_audio_code_) {
      std::cout << "END_AUDIO at frame " << frame << std::endl;
      break;
    }

    std::vector<float> x(n_aco);
    for (auto& v : x)
      v = normal_dist(rng_);
    std::vector<float> zeros(dim, 0.0f);

    for (int step = 0; step < n_decoding_steps_; ++step) {
      float dt = timesteps[step + 1] - timesteps[step];
      int64_t tidx_val = step;

      auto xt1 = from_blob(x.data(), {1, n_aco}, ScalarType::Float);
      auto ti1 = from_blob(&tidx_val, {1}, ScalarType::Long);
      auto hc = from_blob(hidden_state.data(), {1, dim}, ScalarType::Float);
      auto vc = model_->execute(
          "predict_velocity", std::vector<EValue>{*xt1, *ti1, *hc});
      ET_CHECK_MSG(vc.ok(), "predict_velocity (cond) failed");
      std::vector<float> v_cond(n_aco);
      std::memcpy(
          v_cond.data(),
          vc.get()[0].toTensor().mutable_data_ptr<float>(),
          static_cast<size_t>(n_aco) * sizeof(float));

      auto xt2 = from_blob(x.data(), {1, n_aco}, ScalarType::Float);
      auto ti2 = from_blob(&tidx_val, {1}, ScalarType::Long);
      auto hu = from_blob(zeros.data(), {1, dim}, ScalarType::Float);
      auto vu = model_->execute(
          "predict_velocity", std::vector<EValue>{*xt2, *ti2, *hu});
      ET_CHECK_MSG(vu.ok(), "predict_velocity (uncond) failed");
      float* v_uncond = vu.get()[0].toTensor().mutable_data_ptr<float>();

      for (int j = 0; j < n_aco; ++j) {
        float v =
            cfg_alpha_ * v_cond[j] + (1.0f - cfg_alpha_) * v_uncond[j];
        x[j] += v * dt;
      }
    }

    std::vector<int64_t> codes(n_codebooks_);
    codes[0] = semantic_code;
    for (int j = 0; j < n_aco; ++j) {
      float clamped = std::clamp(x[j], -1.0f, 1.0f);
      float scaled = ((clamped + 1.0f) / 2.0f) *
                     static_cast<float>(acoustic_levels_ - 1);
      codes[j + 1] =
          static_cast<int64_t>(std::round(scaled)) + n_special_tokens_;
    }
    frame_codes.push_back(codes);
    emit_ready_audio();

    auto next_codes =
        from_blob(codes.data(), {1, n_cb, 1}, ScalarType::Long);
    auto ne =
        model_->execute("audio_token_embedding", std::vector<EValue>{*next_codes});
    ET_CHECK_MSG(ne.ok(), "audio_token_embedding (next) failed");
    auto next_embeds = ne.get()[0].toTensor();

    int64_t next_pos_val = cur_pos;
    auto np = from_blob(&next_pos_val, {1}, ScalarType::Long);
    auto next_emb = from_blob(
        next_embeds.mutable_data_ptr<float>(), {1, 1, dim},
        ScalarType::Float);
    auto nd =
        model_->execute("text_decoder", std::vector<EValue>{*next_emb, *np});
    ET_CHECK_MSG(nd.ok(), "text_decoder (next) failed");
    std::memcpy(
        hidden_state.data(),
        nd.get()[0].toTensor().mutable_data_ptr<float>(),
        static_cast<size_t>(dim) * sizeof(float));
    cur_pos++;
  }

  // Flush remaining
  if (emitted_frames < static_cast<int64_t>(frame_codes.size())) {
    int64_t decode_start =
        std::max(int64_t(0), emitted_frames - streaming_left_context_);
    int64_t decode_end = static_cast<int64_t>(frame_codes.size());
    int64_t crop_frames = emitted_frames - decode_start;

    std::vector<float> chunk_samples;
    decode_code_window(frame_codes, decode_start, decode_end, chunk_samples);

    int64_t crop_samples = crop_frames * downsample_factor_;
    if (crop_samples < static_cast<int64_t>(chunk_samples.size())) {
      float* new_start = chunk_samples.data() + crop_samples;
      std::size_t new_count = chunk_samples.size() - crop_samples;
      wav.Write(new_start, new_count);
      if (callback)
        callback(new_start, new_count);
    }
  }

  wav.Close();

  auto end_time = std::chrono::high_resolution_clock::now();
  auto total_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          end_time - start_time)
          .count();
  int64_t total_frames = static_cast<int64_t>(frame_codes.size());
  float audio_duration = static_cast<float>(total_frames * downsample_factor_) /
                         static_cast<float>(sample_rate_);
  std::cout << "Streaming: " << total_frames << " frames (" << audio_duration
            << "s) in " << total_ms << "ms, RTF="
            << (static_cast<float>(total_ms) / 1000.0f) / audio_duration
            << std::endl;
}

} // namespace voxtral_tts
