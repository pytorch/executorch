/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include <gflags/gflags.h>

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/wav_loader.h>
#include <executorch/extension/llm/tokenizers/third-party/llama.cpp-unicode/include/unicode.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/log.h>

DEFINE_string(model_path, "parakeet.pte", "Path to Parakeet model (.pte).");
DEFINE_string(audio_path, "", "Path to input audio file (.wav).");
DEFINE_string(
    tokenizer_path,
    "tokenizer.model",
    "Path to SentencePiece tokenizer model file.");
DEFINE_string(
    data_path,
    "",
    "Path to data file (.ptd) for delegate data (optional, required for CUDA).");

using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;

namespace {
// Matches output type of tokenizers::Tokenizer methods
using TokenId = uint64_t;

// TDT duration values
const std::vector<int> DURATIONS = {0, 1, 2, 3, 4};

struct DecodedToken {
  TokenId token_id;
  int64_t start_offset;
  int64_t duration;
};

struct FrameAlignedToken {
  TokenId token_id;
  // Raw vocabulary piece for the token_id (i.e., "##ing", "▁hello")
  std::string token_piece;
  // Decoded text for the token_id (i.e., "ing", " hello")
  std::string token_text;
  int64_t start_offset;
  int64_t end_offset;
};

struct TextWithOffsets {
  std::string text;
  int64_t start_offset;
  int64_t end_offset;
};

bool is_whitespace_only(const std::string& token) {
  if (token.empty()) {
    return true;
  }

  try {
    const auto codepoints = unicode_cpts_from_utf8(token);
    for (const auto cp : codepoints) {
      if (!unicode_cpt_flags(cp).is_whitespace) {
        return false;
      }
    }
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

bool is_special_token(const std::string& token) {
  if (token.size() >= 2 && token.front() == '[' && token.back() == ']') {
    return true;
  }
  if (token.size() >= 2 && token.front() == '<' && token.back() == '>') {
    return true;
  }
  if (token.rfind("##", 0) == 0) {
    return true;
  }
  if (token.rfind(u8"▁", 0) == 0) {
    return true;
  }
  if (is_whitespace_only(token)) {
    return true;
  }
  return false;
}

// Matches NeMo extract_punctuation_from_vocab method
// https://github.com/NVIDIA-NeMo/NeMo/blob/b90a528/nemo/collections/asr/parts/utils/tokenizer_utils.py#L20
std::unordered_set<std::string> derive_supported_punctuation(
    const tokenizers::Tokenizer& tokenizer) {
  std::unordered_set<std::string> punctuation;

  const int32_t vocab_size = tokenizer.vocab_size();
  for (int32_t id = 0; id < vocab_size; id++) {
    const auto piece_result = tokenizer.id_to_piece(static_cast<TokenId>(id));
    if (!piece_result.ok()) {
      continue;
    }
    const std::string& piece = piece_result.get();
    if (is_special_token(piece)) {
      continue;
    }

    try {
      const auto codepoints = unicode_cpts_from_utf8(piece);
      for (const auto cp : codepoints) {
        if (unicode_cpt_flags(cp).is_punctuation) {
          punctuation.insert(unicode_cpt_to_utf8(cp));
        }
      }
    } catch (const std::exception&) {
      ET_LOG(
          Error,
          "Failed to decode token piece '%s' to codepoints",
          piece.c_str());
    }
  }

  return punctuation;
}

std::string decode_token_sequence(
    const std::vector<TokenId>& tokens,
    const tokenizers::Tokenizer& tokenizer) {
  std::string result;
  TokenId prev_token = tokenizer.bos_tok();
  for (const TokenId token : tokens) {
    auto decode_result = tokenizer.decode(prev_token, token);
    if (decode_result.ok()) {
      result += decode_result.get();
    }
    prev_token = token;
  }
  return result;
}

// convenience overload
std::string decode_token_sequence(
    const std::vector<DecodedToken>& decoded_tokens,
    const tokenizers::Tokenizer& tokenizer) {
  std::vector<TokenId> token_ids;
  token_ids.reserve(decoded_tokens.size());
  for (const auto& tok : decoded_tokens) {
    token_ids.push_back(tok.token_id);
  }
  return decode_token_sequence(token_ids, tokenizer);
}

// ref:
// https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c9/nemo/collections/asr/parts/utils/timestamp_utils.py#L54
// assumes BPE tokenizer type
std::vector<TextWithOffsets> get_words_offsets(
    const std::vector<FrameAlignedToken>& tokens,
    const tokenizers::Tokenizer& tokenizer,
    const std::unordered_set<std::string>& supported_punctuation,
    const std::string& word_delimiter_char = " ") {
  std::vector<TextWithOffsets> word_offsets;
  if (tokens.empty()) {
    return word_offsets;
  }

  size_t previous_token_index = 0;
  std::vector<size_t> build_token_indices;

  auto is_curr_punctuation = [&](const std::string& token_text) {
    return token_text != word_delimiter_char &&
        supported_punctuation.count(token_text) > 0;
  };

  auto is_word_start = [&](const std::string& token_piece,
                           const std::string& token_text,
                           const std::string& next_non_delim_token) {
    const bool next_is_punctuation =
        supported_punctuation.count(next_non_delim_token) > 0;
    return token_piece != token_text ||
        (token_text == word_delimiter_char && !next_is_punctuation);
  };

  for (size_t i = 0; i < tokens.size(); i++) {
    const auto& token = tokens[i];

    const bool curr_punctuation = is_curr_punctuation(token.token_text);

    std::string next_non_delim_token;
    for (size_t j = i + 1; j < tokens.size(); j++) {
      if (tokens[j].token_text != word_delimiter_char) {
        next_non_delim_token = tokens[j].token_text;
        break;
      }
    }

    if (is_word_start(
            token.token_piece, token.token_text, next_non_delim_token) &&
        !curr_punctuation) {
      if (!build_token_indices.empty()) {
        std::vector<TokenId> built_ids;
        built_ids.reserve(build_token_indices.size());
        for (size_t idx : build_token_indices) {
          built_ids.push_back(tokens[idx].token_id);
        }
        word_offsets.push_back(
            {decode_token_sequence(built_ids, tokenizer),
             tokens[previous_token_index].start_offset,
             tokens[i - 1].end_offset});
      }

      build_token_indices.clear();

      if (token.token_text != word_delimiter_char) {
        build_token_indices.push_back(i);
        previous_token_index = i;
      }
    } else if (
        curr_punctuation && build_token_indices.empty() &&
        !word_offsets.empty()) {
      auto& last_built_word = word_offsets.back();
      last_built_word.end_offset = token.end_offset;
      if (!last_built_word.text.empty() && last_built_word.text.back() == ' ') {
        last_built_word.text.pop_back();
      }
      last_built_word.text += token.token_text;
    } else if (curr_punctuation && !build_token_indices.empty()) {
      const auto& last = tokens[build_token_indices.back()].token_piece;
      if (last == " " || last == "_" || last == "▁") {
        build_token_indices.pop_back();
      }
      build_token_indices.push_back(i);
    } else {
      if (build_token_indices.empty()) {
        previous_token_index = i;
      }
      build_token_indices.push_back(i);
    }
  }

  // Match NeMo behavior: inject first start_offset and append any remaining
  // built tokens as the final word.
  if (word_offsets.empty()) {
    if (!build_token_indices.empty()) {
      std::vector<TokenId> built_ids;
      built_ids.reserve(build_token_indices.size());
      for (const size_t idx : build_token_indices) {
        built_ids.push_back(tokens[idx].token_id);
      }
      word_offsets.push_back(
          {decode_token_sequence(built_ids, tokenizer),
           tokens[0].start_offset,
           tokens.back().end_offset});
    }
  } else {
    word_offsets[0].start_offset = tokens[0].start_offset;

    if (!build_token_indices.empty()) {
      std::vector<TokenId> built_ids;
      built_ids.reserve(build_token_indices.size());
      for (size_t idx : build_token_indices) {
        built_ids.push_back(tokens[idx].token_id);
      }
      word_offsets.push_back(
          {decode_token_sequence(built_ids, tokenizer),
           tokens[previous_token_index].start_offset,
           tokens.back().end_offset});
    }
  }

  return word_offsets;
}

// ref
// https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c9/nemo/collections/asr/parts/utils/timestamp_utils.py#L227
std::vector<TextWithOffsets> get_segment_offsets(
    const std::vector<TextWithOffsets>& word_offsets,
    const std::vector<std::string>& segment_delimiters = {".", "?", "!"},
    const std::optional<int64_t>& segment_gap_threshold = std::nullopt) {
  std::vector<TextWithOffsets> segment_offsets;
  if (word_offsets.empty()) {
    return segment_offsets;
  }

  std::vector<std::string> segment_words;
  size_t previous_word_index = 0;

  for (size_t i = 0; i < word_offsets.size(); i++) {
    const auto& offset = word_offsets[i];
    const auto& word = offset.text;

    if (segment_gap_threshold.has_value() && !segment_words.empty()) {
      const int64_t gap_between_words =
          offset.start_offset - word_offsets[i - 1].end_offset;
      if (gap_between_words >= segment_gap_threshold.value()) {
        std::string segment;
        for (size_t j = 0; j < segment_words.size(); j++) {
          if (j > 0) {
            segment += " ";
          }
          segment += segment_words[j];
        }
        segment_offsets.push_back(
            {segment,
             word_offsets[previous_word_index].start_offset,
             word_offsets[i - 1].end_offset});
        segment_words = {word};
        previous_word_index = i;
        continue;
      }
    }

    const bool is_delimiter_word =
        std::find(segment_delimiters.begin(), segment_delimiters.end(), word) !=
        segment_delimiters.end();

    const bool ends_with_delimiter = !word.empty() &&
        std::find(
            segment_delimiters.begin(),
            segment_delimiters.end(),
            std::string(1, word.back())) != segment_delimiters.end();

    if (!word.empty() && (ends_with_delimiter || is_delimiter_word)) {
      segment_words.push_back(word);
      if (!segment_words.empty()) {
        std::string segment;
        for (size_t j = 0; j < segment_words.size(); j++) {
          if (j > 0) {
            segment += " ";
          }
          segment += segment_words[j];
        }
        segment_offsets.push_back(
            {segment,
             word_offsets[previous_word_index].start_offset,
             offset.end_offset});
      }
      segment_words.clear();
      previous_word_index = i + 1;
      continue;
    }

    segment_words.push_back(word);
  }

  if (!segment_words.empty()) {
    std::string segment;
    for (size_t j = 0; j < segment_words.size(); j++) {
      if (j > 0) {
        segment += " ";
      }
      segment += segment_words[j];
    }
    segment_offsets.push_back(
        {segment,
         word_offsets[previous_word_index].start_offset,
         word_offsets.back().end_offset});
  }

  return segment_offsets;
}

std::vector<DecodedToken> greedy_decode_executorch(
    Module& model,
    const ::executorch::aten::Tensor& encoder_output,
    int64_t encoder_len,
    int64_t blank_id,
    int64_t vocab_size,
    int64_t num_rnn_layers = 2,
    int64_t pred_hidden = 640,
    int64_t max_symbols_per_step = 10) {
  std::vector<DecodedToken> hypothesis;
  int64_t num_token_classes = vocab_size + 1;

  // Transpose encoder output from [1, enc_dim, time] to [1, time, enc_dim]
  auto enc_sizes = encoder_output.sizes();
  int64_t batch = enc_sizes[0];
  int64_t enc_dim = enc_sizes[1];
  int64_t time_steps = enc_sizes[2];

  // Create transposed tensor
  std::vector<float> transposed_data(batch * time_steps * enc_dim);
  const float* src = encoder_output.const_data_ptr<float>();
  for (int64_t t = 0; t < time_steps; t++) {
    for (int64_t d = 0; d < enc_dim; d++) {
      transposed_data[t * enc_dim + d] = src[d * time_steps + t];
    }
  }

  auto transposed_tensor = from_blob(
      transposed_data.data(),
      {static_cast<::executorch::aten::SizesType>(batch),
       static_cast<::executorch::aten::SizesType>(time_steps),
       static_cast<::executorch::aten::SizesType>(enc_dim)},
      ::executorch::aten::ScalarType::Float);

  // Project encoder output
  auto proj_enc_result = model.execute(
      "joint_project_encoder",
      std::vector<::executorch::runtime::EValue>{transposed_tensor});
  if (!proj_enc_result.ok()) {
    ET_LOG(Error, "joint_project_encoder failed");
    return hypothesis;
  }
  auto f_proj = proj_enc_result.get()[0].toTensor();

  // Initialize LSTM state
  std::vector<float> h_data(num_rnn_layers * 1 * pred_hidden, 0.0f);
  std::vector<float> c_data(num_rnn_layers * 1 * pred_hidden, 0.0f);

  auto h = from_blob(
      h_data.data(),
      {static_cast<::executorch::aten::SizesType>(num_rnn_layers),
       1,
       static_cast<::executorch::aten::SizesType>(pred_hidden)},
      ::executorch::aten::ScalarType::Float);
  auto c = from_blob(
      c_data.data(),
      {static_cast<::executorch::aten::SizesType>(num_rnn_layers),
       1,
       static_cast<::executorch::aten::SizesType>(pred_hidden)},
      ::executorch::aten::ScalarType::Float);

  // Prime the prediction network state with SOS (= blank_id) to match NeMo TDT
  // greedy label-looping decoding behavior:
  // - SOS is defined as blank:
  // https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c980b70cecc184fa8a083a9c3ddb87f905e/nemo/collections/asr/parts/submodules/transducer_decoding/tdt_label_looping.py#L250
  // - Predictor priming with SOS:
  // https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c980b70cecc184fa8a083a9c3ddb87f905e/nemo/collections/asr/parts/submodules/transducer_decoding/tdt_label_looping.py#L363-L368
  std::vector<int64_t> sos_token_data = {blank_id};
  auto sos_token = from_blob(
      sos_token_data.data(), {1, 1}, ::executorch::aten::ScalarType::Long);
  auto decoder_init_result = model.execute(
      "decoder_predict",
      std::vector<::executorch::runtime::EValue>{sos_token, h, c});
  if (!decoder_init_result.ok()) {
    ET_LOG(Error, "decoder_predict (SOS) failed");
    return hypothesis;
  }
  auto& init_outputs = decoder_init_result.get();
  auto g_init = init_outputs[0].toTensor();
  auto new_h_init = init_outputs[1].toTensor();
  auto new_c_init = init_outputs[2].toTensor();
  std::memcpy(
      h_data.data(),
      new_h_init.const_data_ptr<float>(),
      h_data.size() * sizeof(float));
  std::memcpy(
      c_data.data(),
      new_c_init.const_data_ptr<float>(),
      c_data.size() * sizeof(float));

  auto g_proj_result = model.execute(
      "joint_project_decoder",
      std::vector<::executorch::runtime::EValue>{g_init});
  if (!g_proj_result.ok()) {
    ET_LOG(Error, "joint_project_decoder failed");
    return hypothesis;
  }
  auto g_proj_tensor = g_proj_result.get()[0].toTensor();

  // Copy g_proj data for reuse
  std::vector<float> g_proj_data(
      g_proj_tensor.const_data_ptr<float>(),
      g_proj_tensor.const_data_ptr<float>() + g_proj_tensor.numel());

  int64_t t = 0;
  int64_t symbols_on_frame = 0;

  // Scan over encoder output
  while (t < encoder_len) {
    // Get encoder frame at time t: f_proj[:, t:t+1, :]
    const float* f_proj_data = f_proj.const_data_ptr<float>();
    int64_t proj_dim = f_proj.sizes()[2];

    std::vector<float> f_t_data(1 * 1 * proj_dim);
    for (int64_t d = 0; d < proj_dim; d++) {
      f_t_data[d] = f_proj_data[t * proj_dim + d];
    }
    auto f_t = from_blob(
        f_t_data.data(),
        {1, 1, static_cast<::executorch::aten::SizesType>(proj_dim)},
        ::executorch::aten::ScalarType::Float);

    auto g_proj = from_blob(
        g_proj_data.data(),
        {1, 1, static_cast<::executorch::aten::SizesType>(proj_dim)},
        ::executorch::aten::ScalarType::Float);

    // Joint network
    auto joint_result = model.execute(
        "joint", std::vector<::executorch::runtime::EValue>{f_t, g_proj});
    if (!joint_result.ok()) {
      ET_LOG(Error, "joint failed at t=%lld", static_cast<long long>(t));
      return hypothesis;
    }
    auto full_logits = joint_result.get()[0].toTensor();

    // Split logits into token and duration
    const float* logits_data = full_logits.const_data_ptr<float>();

    // Find argmax for token logits
    int64_t k = 0;
    float max_token_logit = logits_data[0];
    for (int64_t i = 1; i < num_token_classes; i++) {
      if (logits_data[i] > max_token_logit) {
        max_token_logit = logits_data[i];
        k = i;
      }
    }

    // Find argmax for duration logits
    int64_t dur_idx = 0;
    float max_dur_logit = logits_data[num_token_classes];
    for (size_t i = 1; i < DURATIONS.size(); i++) {
      if (logits_data[num_token_classes + i] > max_dur_logit) {
        max_dur_logit = logits_data[num_token_classes + i];
        dur_idx = i;
      }
    }
    int64_t dur = DURATIONS[dur_idx];

    if (k == blank_id) {
      t += std::max(dur, static_cast<int64_t>(1));
      symbols_on_frame = 0;
    } else {
      hypothesis.push_back({static_cast<TokenId>(k), t, dur});

      // Update decoder state
      std::vector<int64_t> token_data = {k};
      auto token = from_blob(
          token_data.data(), {1, 1}, ::executorch::aten::ScalarType::Long);

      auto decoder_result = model.execute(
          "decoder_predict",
          std::vector<::executorch::runtime::EValue>{token, h, c});
      if (!decoder_result.ok()) {
        ET_LOG(Error, "decoder_predict failed");
        return hypothesis;
      }
      auto& outputs = decoder_result.get();
      auto g = outputs[0].toTensor();
      auto new_h = outputs[1].toTensor();
      auto new_c = outputs[2].toTensor();

      // Update h and c
      std::memcpy(
          h_data.data(),
          new_h.const_data_ptr<float>(),
          h_data.size() * sizeof(float));
      std::memcpy(
          c_data.data(),
          new_c.const_data_ptr<float>(),
          c_data.size() * sizeof(float));

      // Project decoder output
      auto proj_dec_result = model.execute(
          "joint_project_decoder",
          std::vector<::executorch::runtime::EValue>{g});
      if (!proj_dec_result.ok()) {
        ET_LOG(Error, "joint_project_decoder failed");
        return hypothesis;
      }
      auto new_g_proj = proj_dec_result.get()[0].toTensor();
      std::memcpy(
          g_proj_data.data(),
          new_g_proj.const_data_ptr<float>(),
          g_proj_data.size() * sizeof(float));

      t += dur;

      if (dur == 0) {
        symbols_on_frame++;
        if (symbols_on_frame >= max_symbols_per_step) {
          t++;
          symbols_on_frame = 0;
        }
      } else {
        symbols_on_frame = 0;
      }
    }
  }

  return hypothesis;
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_audio_path.empty()) {
    ET_LOG(Error, "audio_path flag must be provided.");
    return 1;
  }

  // Load model (which includes the bundled preprocessor)
  ET_LOG(Info, "Loading model from: %s", FLAGS_model_path.c_str());
  std::unique_ptr<Module> model;
  if (!FLAGS_data_path.empty()) {
    ET_LOG(Info, "Loading data from: %s", FLAGS_data_path.c_str());
    model = std::make_unique<Module>(
        FLAGS_model_path, FLAGS_data_path, Module::LoadMode::Mmap);
  } else {
    model = std::make_unique<Module>(FLAGS_model_path, Module::LoadMode::Mmap);
  }
  auto model_load_error = model->load();
  if (model_load_error != Error::Ok) {
    ET_LOG(Error, "Failed to load model.");
    return 1;
  }

  // Load audio
  ET_LOG(Info, "Loading audio from: %s", FLAGS_audio_path.c_str());
  std::vector<float> audio_data =
      ::executorch::extension::llm::load_wav_audio_data(FLAGS_audio_path);
  ET_LOG(Info, "Loaded %zu audio samples", audio_data.size());

  auto audio_tensor = from_blob(
      audio_data.data(),
      {static_cast<::executorch::aten::SizesType>(audio_data.size())},
      ::executorch::aten::ScalarType::Float);
  std::vector<int64_t> audio_len_data = {
      static_cast<int64_t>(audio_data.size())};
  auto audio_len_tensor = from_blob(
      audio_len_data.data(), {1}, ::executorch::aten::ScalarType::Long);

  ET_LOG(Info, "Running preprocessor...");
  auto proc_result = model->execute(
      "preprocessor",
      std::vector<::executorch::runtime::EValue>{
          audio_tensor, audio_len_tensor});
  if (!proc_result.ok()) {
    ET_LOG(Error, "Preprocessor forward failed.");
    return 1;
  }
  auto& proc_outputs = proc_result.get();
  auto mel = proc_outputs[0].toTensor();
  auto mel_len_tensor_out = proc_outputs[1].toTensor();
  int64_t mel_len_value = mel_len_tensor_out.const_data_ptr<int64_t>()[0];

  // Create mel_len tensor for encoder
  std::vector<int64_t> mel_len_data = {mel_len_value};
  auto mel_len =
      from_blob(mel_len_data.data(), {1}, ::executorch::aten::ScalarType::Long);

  ET_LOG(
      Info,
      "Mel spectrogram shape: [%ld, %ld, %ld], mel_len: %lld",
      static_cast<long>(mel.sizes()[0]),
      static_cast<long>(mel.sizes()[1]),
      static_cast<long>(mel.sizes()[2]),
      static_cast<long long>(mel_len_value));

  // Run encoder
  ET_LOG(Info, "Running encoder...");
  auto enc_result = model->execute(
      "encoder", std::vector<::executorch::runtime::EValue>{mel, mel_len});
  if (!enc_result.ok()) {
    ET_LOG(Error, "Encoder forward failed.");
    return 1;
  }
  auto& enc_outputs = enc_result.get();
  auto encoded = enc_outputs[0].toTensor();
  int64_t encoded_len = enc_outputs[1].toTensor().const_data_ptr<int64_t>()[0];

  ET_LOG(
      Info,
      "Encoder output shape: [%ld, %ld, %ld], len=%ld",
      static_cast<long>(encoded.sizes()[0]),
      static_cast<long>(encoded.sizes()[1]),
      static_cast<long>(encoded.sizes()[2]),
      static_cast<long>(encoded_len));

  // Query model metadata from constant_methods
  std::vector<::executorch::runtime::EValue> empty_inputs;
  auto num_rnn_layers_result = model->execute("num_rnn_layers", empty_inputs);
  auto pred_hidden_result = model->execute("pred_hidden", empty_inputs);
  auto vocab_size_result = model->execute("vocab_size", empty_inputs);
  auto blank_id_result = model->execute("blank_id", empty_inputs);
  auto sample_rate_result = model->execute("sample_rate", empty_inputs);
  auto window_stride_result = model->execute("window_stride", empty_inputs);
  auto encoder_subsampling_factor_result =
      model->execute("encoder_subsampling_factor", empty_inputs);

  if (!num_rnn_layers_result.ok() || !pred_hidden_result.ok() ||
      !vocab_size_result.ok() || !blank_id_result.ok() ||
      !sample_rate_result.ok() || !window_stride_result.ok() ||
      !encoder_subsampling_factor_result.ok()) {
    ET_LOG(
        Error,
        "Failed to query model metadata. Make sure the model was exported with constant_methods.");
    return 1;
  }

  int64_t vocab_size = vocab_size_result.get()[0].toInt();
  int64_t blank_id = blank_id_result.get()[0].toInt();
  int64_t num_rnn_layers = num_rnn_layers_result.get()[0].toInt();
  int64_t pred_hidden = pred_hidden_result.get()[0].toInt();
  int64_t sample_rate = sample_rate_result.get()[0].toInt();
  double window_stride = window_stride_result.get()[0].toDouble();
  int64_t encoder_subsampling_factor =
      encoder_subsampling_factor_result.get()[0].toInt();

  ET_LOG(
      Info,
      "Model metadata: vocab_size=%lld, blank_id=%lld, num_rnn_layers=%lld, pred_hidden=%lld, sample_rate=%lld, window_stride=%.6f, encoder_subsampling_factor=%lld",
      static_cast<long long>(vocab_size),
      static_cast<long long>(blank_id),
      static_cast<long long>(num_rnn_layers),
      static_cast<long long>(pred_hidden),
      static_cast<long long>(sample_rate),
      window_stride,
      encoder_subsampling_factor);

  ET_LOG(Info, "Running TDT greedy decode...");
  auto decoded_tokens = greedy_decode_executorch(
      *model,
      encoded,
      encoded_len,
      blank_id,
      vocab_size,
      num_rnn_layers,
      pred_hidden);

  ET_LOG(Info, "Decoded %zu tokens", decoded_tokens.size());

  // Load tokenizer
  ET_LOG(Info, "Loading tokenizer from: %s", FLAGS_tokenizer_path.c_str());
  auto tokenizer =
      ::executorch::extension::llm::load_tokenizer(FLAGS_tokenizer_path);
  if (!tokenizer || !tokenizer->is_loaded()) {
    ET_LOG(
        Error,
        "Failed to load tokenizer from: %s",
        FLAGS_tokenizer_path.c_str());
    return 1;
  }

  // Convert tokens to text
  std::string text = decode_token_sequence(decoded_tokens, *tokenizer);
  std::cout << "Transcription tokens: " << text << std::endl;

  std::unordered_set<std::string> supported_punctuation =
      derive_supported_punctuation(*tokenizer);
  ET_LOG(
      Info,
      "Derived supported_punctuation size=%zu",
      supported_punctuation.size());

  // Compute timestamps matching NeMo's TDT timestamp behavior.
  std::vector<FrameAlignedToken> char_timestamps;
  char_timestamps.reserve(decoded_tokens.size());

  for (const auto& decoded_token : decoded_tokens) {
    const TokenId token_id = decoded_token.token_id;

    auto piece_result = tokenizer->id_to_piece(token_id);
    if (!piece_result.ok()) {
      ET_LOG(Error, "id_to_piece failed for token=%llu", token_id);
      return 1;
    }

    auto text_result = tokenizer->decode(tokenizer->bos_tok(), token_id);
    if (!text_result.ok()) {
      ET_LOG(
          Error,
          "decode failed for token=%llu",
          static_cast<unsigned long long>(token_id));
      return 1;
    }

    const int64_t start_offset = decoded_token.start_offset;
    const int64_t end_offset = start_offset + decoded_token.duration;

    char_timestamps.push_back(
        {token_id,
         piece_result.get(),
         text_result.get(),
         start_offset,
         end_offset});
  }

  // NeMo TDT punctuation refinement: snap punctuation to the end of the
  // previous token.
  for (size_t i = 1; i < char_timestamps.size(); i++) {
    if (supported_punctuation.count(char_timestamps[i].token_text) > 0) {
      char_timestamps[i].start_offset = char_timestamps[i - 1].end_offset;
      char_timestamps[i].end_offset = char_timestamps[i].start_offset;
    }
  }

  auto word_timestamps =
      get_words_offsets(char_timestamps, *tokenizer, supported_punctuation);
  auto segment_timestamps = get_segment_offsets(word_timestamps);

  const double frame_to_seconds =
      window_stride * static_cast<double>(encoder_subsampling_factor);

  std::cout << "\nSegment timestamps:" << std::endl;
  for (const auto& stamp : segment_timestamps) {
    const double start = stamp.start_offset * frame_to_seconds;
    const double end = stamp.end_offset * frame_to_seconds;
    std::cout << start << "s - " << end << "s : " << stamp.text << std::endl;
  }

  std::cout << "\nWord timestamps:" << std::endl;
  for (const auto& stamp : word_timestamps) {
    const double start = stamp.start_offset * frame_to_seconds;
    const double end = stamp.end_offset * frame_to_seconds;
    std::cout << start << "s - " << end << "s : " << stamp.text << std::endl;
  }

  std::cout << "\nChar timestamps:" << std::endl;
  for (const auto& stamp : char_timestamps) {
    const double start = stamp.start_offset * frame_to_seconds;
    const double end = stamp.end_offset * frame_to_seconds;
    std::cout << start << "s - " << end << "s : " << stamp.token_text
              << std::endl;
  }

  ET_LOG(Info, "Done!");
  return 0;
}
