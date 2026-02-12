/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <jni.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/wav_loader.h>
#include <executorch/extension/llm/tokenizers/third-party/llama.cpp-unicode/include/unicode.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/log.h>
#include <pytorch/tokenizers/tokenizer.h>

using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;

// ============================================================================
// Parakeet types (inlined from types.h)
// ============================================================================
namespace parakeet {

using TokenId = uint64_t;

struct Token {
  TokenId id;
  int64_t start_offset;
  int64_t duration;
};

struct TokenWithTextInfo {
  TokenId id;
  std::string raw_piece;
  std::string decoded_text;
  int64_t start_offset;
  int64_t end_offset;
};

struct TextWithOffsets {
  std::string text;
  int64_t start_offset;
  int64_t end_offset;
};

} // namespace parakeet

// ============================================================================
// Tokenizer utilities (inlined from tokenizer_utils.h/cpp)
// ============================================================================
namespace {

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

} // namespace

namespace parakeet::tokenizer_utils {

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

std::string decode_token_sequence(
    const std::vector<Token>& decoded_tokens,
    const tokenizers::Tokenizer& tokenizer) {
  std::vector<TokenId> token_ids;
  token_ids.reserve(decoded_tokens.size());
  for (const auto& tok : decoded_tokens) {
    token_ids.push_back(tok.id);
  }
  return decode_token_sequence(token_ids, tokenizer);
}

} // namespace parakeet::tokenizer_utils

// ============================================================================
// Timestamp utilities (inlined from timestamp_utils.h/cpp)
// ============================================================================
namespace parakeet::timestamp_utils {

std::vector<TokenWithTextInfo> get_tokens_with_text_info(
    const std::vector<Token>& tokens,
    const tokenizers::Tokenizer& tokenizer,
    const std::unordered_set<std::string>& supported_punctuation) {
  std::vector<TokenWithTextInfo> tokens_with_text;
  tokens_with_text.reserve(tokens.size());

  for (const auto& token : tokens) {
    auto piece_result = tokenizer.id_to_piece(token.id);
    if (!piece_result.ok()) {
      throw std::runtime_error(
          "id_to_piece failed for token=" + std::to_string(token.id));
    }

    auto text_result = tokenizer.decode(tokenizer.bos_tok(), token.id);
    if (!text_result.ok()) {
      throw std::runtime_error(
          "decode failed for token=" + std::to_string(token.id));
    }

    tokens_with_text.push_back(
        {token.id,
         piece_result.get(),
         text_result.get(),
         token.start_offset,
         token.start_offset + token.duration});
  }

  for (size_t i = 1; i < tokens_with_text.size(); i++) {
    if (supported_punctuation.count(tokens_with_text[i].decoded_text) > 0) {
      tokens_with_text[i].start_offset = tokens_with_text[i - 1].end_offset;
      tokens_with_text[i].end_offset = tokens_with_text[i].start_offset;
    }
  }

  return tokens_with_text;
}

std::vector<TextWithOffsets> get_words_offsets(
    const std::vector<TokenWithTextInfo>& tokens,
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

    const bool curr_punctuation = is_curr_punctuation(token.decoded_text);

    std::string next_non_delim_token;
    for (size_t j = i + 1; j < tokens.size(); j++) {
      if (tokens[j].decoded_text != word_delimiter_char) {
        next_non_delim_token = tokens[j].decoded_text;
        break;
      }
    }

    if (is_word_start(
            token.raw_piece, token.decoded_text, next_non_delim_token) &&
        !curr_punctuation) {
      if (!build_token_indices.empty()) {
        std::vector<TokenId> built_ids;
        built_ids.reserve(build_token_indices.size());
        for (size_t idx : build_token_indices) {
          built_ids.push_back(tokens[idx].id);
        }
        word_offsets.push_back(
            {tokenizer_utils::decode_token_sequence(built_ids, tokenizer),
             tokens[previous_token_index].start_offset,
             tokens[i - 1].end_offset});
      }

      build_token_indices.clear();

      if (token.decoded_text != word_delimiter_char) {
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
      last_built_word.text += token.decoded_text;
    } else if (curr_punctuation && !build_token_indices.empty()) {
      const auto& last = tokens[build_token_indices.back()].raw_piece;
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

  if (word_offsets.empty()) {
    if (!build_token_indices.empty()) {
      std::vector<TokenId> built_ids;
      built_ids.reserve(build_token_indices.size());
      for (const size_t idx : build_token_indices) {
        built_ids.push_back(tokens[idx].id);
      }
      word_offsets.push_back(
          {tokenizer_utils::decode_token_sequence(built_ids, tokenizer),
           tokens[0].start_offset,
           tokens.back().end_offset});
    }
  } else {
    word_offsets[0].start_offset = tokens[0].start_offset;

    if (!build_token_indices.empty()) {
      std::vector<TokenId> built_ids;
      built_ids.reserve(build_token_indices.size());
      for (size_t idx : build_token_indices) {
        built_ids.push_back(tokens[idx].id);
      }
      word_offsets.push_back(
          {tokenizer_utils::decode_token_sequence(built_ids, tokenizer),
           tokens[previous_token_index].start_offset,
           tokens.back().end_offset});
    }
  }

  return word_offsets;
}

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

} // namespace parakeet::timestamp_utils

// ============================================================================
// Parakeet decoding logic (converted from main.cpp)
// ============================================================================
namespace {

const std::vector<int> DURATIONS = {0, 1, 2, 3, 4};

struct TimestampOutputMode {
  bool token = false;
  bool word = false;
  bool segment = false;

  bool enabled() const {
    return token || word || segment;
  }
};

std::string to_lower_ascii(std::string s) {
  for (char& ch : s) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return s;
}

TimestampOutputMode parse_timestamp_output_mode(const std::string& raw_arg) {
  if (raw_arg.empty()) {
    return {false, false, true}; // default to segment
  }
  const std::string mode = to_lower_ascii(raw_arg);
  if (mode == "none") {
    return {false, false, false};
  }
  if (mode == "token") {
    return {true, false, false};
  }
  if (mode == "word") {
    return {false, true, false};
  }
  if (mode == "segment") {
    return {false, false, true};
  }
  if (mode == "all") {
    return {true, true, true};
  }
  return {false, false, true}; // default to segment
}

::executorch::runtime::Result<::executorch::aten::ScalarType>
get_input_scalar_type(
    Module& model,
    const char* method_name,
    size_t input_index) {
  auto method_meta_result = model.method_meta(method_name);
  if (!method_meta_result.ok()) {
    ET_LOG(Error, "Failed to get method metadata for %s", method_name);
    return method_meta_result.error();
  }
  auto method_meta = method_meta_result.get();
  if (method_meta.num_inputs() <= input_index) {
    ET_LOG(
        Error,
        "Method %s has %zu inputs, but requested index %zu",
        method_name,
        method_meta.num_inputs(),
        input_index);
    return ::executorch::runtime::Error::InvalidArgument;
  }
  auto input_meta_result = method_meta.input_tensor_meta(input_index);
  if (input_meta_result.error() != ::executorch::runtime::Error::Ok) {
    ET_LOG(
        Error,
        "Failed to get input tensor metadata for %s[%zu]",
        method_name,
        input_index);
    return input_meta_result.error();
  }
  return input_meta_result.get().scalar_type();
}

std::vector<parakeet::Token> greedy_decode_executorch(
    Module& model,
    const ::executorch::aten::Tensor& f_proj,
    int64_t encoder_len,
    int64_t blank_id,
    int64_t num_rnn_layers = 2,
    int64_t pred_hidden = 640,
    int64_t max_symbols_per_step = 10) {
  std::vector<parakeet::Token> hypothesis;

  size_t proj_dim = static_cast<size_t>(f_proj.sizes()[2]);

  auto h_dtype_result = get_input_scalar_type(model, "decoder_step", 1);
  if (!h_dtype_result.ok()) {
    return hypothesis;
  }
  auto c_dtype_result = get_input_scalar_type(model, "decoder_step", 2);
  if (!c_dtype_result.ok()) {
    return hypothesis;
  }
  auto h_dtype = h_dtype_result.get();
  auto c_dtype = c_dtype_result.get();

  ET_LOG(
      Info,
      "Decoder h dtype: %s, c dtype: %s",
      ::executorch::runtime::toString(h_dtype),
      ::executorch::runtime::toString(c_dtype));

  size_t h_elem_size = ::executorch::runtime::elementSize(h_dtype);
  size_t c_elem_size = ::executorch::runtime::elementSize(c_dtype);
  size_t num_elements =
      static_cast<size_t>(num_rnn_layers) * static_cast<size_t>(pred_hidden);

  std::vector<uint8_t> h_data(num_elements * h_elem_size, 0);
  std::vector<uint8_t> c_data(num_elements * c_elem_size, 0);

  auto h = from_blob(
      h_data.data(),
      {static_cast<::executorch::aten::SizesType>(num_rnn_layers),
       1,
       static_cast<::executorch::aten::SizesType>(pred_hidden)},
      h_dtype);
  auto c = from_blob(
      c_data.data(),
      {static_cast<::executorch::aten::SizesType>(num_rnn_layers),
       1,
       static_cast<::executorch::aten::SizesType>(pred_hidden)},
      c_dtype);

  std::vector<int64_t> sos_token_data = {blank_id};
  auto sos_token = from_blob(
      sos_token_data.data(), {1, 1}, ::executorch::aten::ScalarType::Long);
  auto decoder_init_result = model.execute(
      "decoder_step",
      std::vector<::executorch::runtime::EValue>{sos_token, h, c});
  if (!decoder_init_result.ok()) {
    ET_LOG(Error, "decoder_step (SOS) failed");
    return hypothesis;
  }
  auto& init_outputs = decoder_init_result.get();
  auto g_proj_init = init_outputs[0].toTensor();
  auto new_h_init = init_outputs[1].toTensor();
  auto new_c_init = init_outputs[2].toTensor();
  std::memcpy(h_data.data(), new_h_init.const_data_ptr(), h_data.size());
  std::memcpy(c_data.data(), new_c_init.const_data_ptr(), c_data.size());

  auto f_dtype_result = get_input_scalar_type(model, "joint", 0);
  if (!f_dtype_result.ok()) {
    return hypothesis;
  }
  auto g_dtype_result = get_input_scalar_type(model, "joint", 1);
  if (!g_dtype_result.ok()) {
    return hypothesis;
  }
  auto f_dtype = f_dtype_result.get();
  auto g_dtype = g_dtype_result.get();

  ET_LOG(
      Info,
      "Joint f dtype: %s, g dtype: %s",
      ::executorch::runtime::toString(f_dtype),
      ::executorch::runtime::toString(g_dtype));

  size_t f_elem_size = ::executorch::runtime::elementSize(f_dtype);
  size_t g_elem_size = ::executorch::runtime::elementSize(g_dtype);

  size_t g_proj_num_bytes =
      static_cast<size_t>(g_proj_init.numel()) * g_elem_size;
  std::vector<uint8_t> g_proj_data(g_proj_num_bytes);
  std::memcpy(
      g_proj_data.data(), g_proj_init.const_data_ptr(), g_proj_num_bytes);

  int64_t t = 0;
  int64_t symbols_on_frame = 0;
  const uint8_t* f_proj_ptr =
      static_cast<const uint8_t*>(f_proj.const_data_ptr());
  size_t f_t_num_bytes = proj_dim * f_elem_size;

  while (t < encoder_len) {
    std::vector<uint8_t> f_t_data(f_t_num_bytes);
    std::memcpy(
        f_t_data.data(),
        f_proj_ptr + static_cast<size_t>(t) * f_t_num_bytes,
        f_t_num_bytes);

    auto f_t = from_blob(
        f_t_data.data(),
        {1, 1, static_cast<::executorch::aten::SizesType>(proj_dim)},
        f_dtype);

    auto g_proj = from_blob(
        g_proj_data.data(),
        {1, 1, static_cast<::executorch::aten::SizesType>(proj_dim)},
        g_dtype);

    auto joint_result = model.execute(
        "joint", std::vector<::executorch::runtime::EValue>{f_t, g_proj});
    if (!joint_result.ok()) {
      ET_LOG(Error, "joint failed at t=%lld", static_cast<long long>(t));
      return hypothesis;
    }

    int64_t k = joint_result.get()[0].toTensor().const_data_ptr<int64_t>()[0];
    int64_t dur_idx =
        joint_result.get()[1].toTensor().const_data_ptr<int64_t>()[0];
    int64_t dur = DURATIONS[dur_idx];

    if (k == blank_id) {
      t += std::max(dur, static_cast<int64_t>(1));
      symbols_on_frame = 0;
    } else {
      hypothesis.push_back(
          {static_cast<parakeet::TokenId>(k), t, dur});

      std::vector<int64_t> token_data = {k};
      auto token = from_blob(
          token_data.data(), {1, 1}, ::executorch::aten::ScalarType::Long);

      auto decoder_result = model.execute(
          "decoder_step",
          std::vector<::executorch::runtime::EValue>{token, h, c});
      if (!decoder_result.ok()) {
        ET_LOG(Error, "decoder_step failed");
        return hypothesis;
      }
      auto& outputs = decoder_result.get();
      auto new_g_proj = outputs[0].toTensor();
      auto new_h = outputs[1].toTensor();
      auto new_c = outputs[2].toTensor();

      std::memcpy(h_data.data(), new_h.const_data_ptr(), h_data.size());
      std::memcpy(c_data.data(), new_c.const_data_ptr(), c_data.size());
      std::memcpy(
          g_proj_data.data(), new_g_proj.const_data_ptr(), g_proj_data.size());

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

std::string run_parakeet_tdt(
    const std::string& model_path,
    const std::string& audio_path,
    const std::string& tokenizer_path,
    const std::string& data_path,
    const std::string& timestamps) {
  TimestampOutputMode timestamp_mode = parse_timestamp_output_mode(timestamps);

  if (audio_path.empty()) {
    ET_LOG(Error, "audio_path must be provided.");
    return "";
  }

  ET_LOG(Info, "Loading model from: %s", model_path.c_str());
  std::unique_ptr<Module> model;
  if (!data_path.empty()) {
    ET_LOG(Info, "Loading data from: %s", data_path.c_str());
    model = std::make_unique<Module>(
        model_path, data_path, Module::LoadMode::Mmap);
  } else {
    model = std::make_unique<Module>(model_path, Module::LoadMode::Mmap);
  }
  auto model_load_error = model->load();
  if (model_load_error != Error::Ok) {
    ET_LOG(Error, "Failed to load model.");
    return "";
  }

  ET_LOG(Info, "Loading audio from: %s", audio_path.c_str());
  std::vector<float> audio_data =
      ::executorch::extension::llm::load_wav_audio_data(audio_path);
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
    return "";
  }
  auto& proc_outputs = proc_result.get();
  auto mel = proc_outputs[0].toTensor();
  auto mel_len_tensor_out = proc_outputs[1].toTensor();
  int64_t mel_len_value = mel_len_tensor_out.const_data_ptr<int64_t>()[0];

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

  ET_LOG(Info, "Running encoder...");
  auto enc_result = model->execute(
      "encoder", std::vector<::executorch::runtime::EValue>{mel, mel_len});
  if (!enc_result.ok()) {
    ET_LOG(Error, "Encoder forward failed.");
    return "";
  }

  auto& enc_outputs = enc_result.get();
  auto f_proj = enc_outputs[0].toTensor();
  int64_t encoded_len = enc_outputs[1].toTensor().const_data_ptr<int64_t>()[0];

  ET_LOG(
      Info,
      "Encoder output (f_proj) shape: [%ld, %ld, %ld], len=%ld",
      static_cast<long>(f_proj.sizes()[0]),
      static_cast<long>(f_proj.sizes()[1]),
      static_cast<long>(f_proj.sizes()[2]),
      static_cast<long>(encoded_len));

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
    return "";
  }

  int64_t blank_id = blank_id_result.get()[0].toInt();
  int64_t num_rnn_layers = num_rnn_layers_result.get()[0].toInt();
  int64_t pred_hidden = pred_hidden_result.get()[0].toInt();
  double window_stride = window_stride_result.get()[0].toDouble();
  int64_t encoder_subsampling_factor =
      encoder_subsampling_factor_result.get()[0].toInt();

  ET_LOG(Info, "Running TDT greedy decode...");
  auto decoded_tokens = greedy_decode_executorch(
      *model, f_proj, encoded_len, blank_id, num_rnn_layers, pred_hidden);

  ET_LOG(Info, "Decoded %zu tokens", decoded_tokens.size());

  ET_LOG(Info, "Loading tokenizer from: %s", tokenizer_path.c_str());
  auto tokenizer =
      ::executorch::extension::llm::load_tokenizer(tokenizer_path);
  if (!tokenizer || !tokenizer->is_loaded()) {
    ET_LOG(
        Error, "Failed to load tokenizer from: %s", tokenizer_path.c_str());
    return "";
  }

  std::string text = parakeet::tokenizer_utils::decode_token_sequence(
      decoded_tokens, *tokenizer);

  if (!timestamp_mode.enabled()) {
    return text;
  }

  ET_LOG(Info, "Computing timestamps...");
  std::unordered_set<std::string> supported_punctuation =
      parakeet::tokenizer_utils::derive_supported_punctuation(*tokenizer);
  ET_LOG(
      Info,
      "Derived supported_punctuation size=%zu",
      supported_punctuation.size());

  std::vector<parakeet::TokenWithTextInfo> tokens_with_text_info;
  try {
    tokens_with_text_info =
        parakeet::timestamp_utils::get_tokens_with_text_info(
            decoded_tokens, *tokenizer, supported_punctuation);
  } catch (const std::exception& e) {
    ET_LOG(Error, "Failed to get tokens with text info: %s", e.what());
    return text;
  }
  const auto word_offsets = parakeet::timestamp_utils::get_words_offsets(
      tokens_with_text_info, *tokenizer, supported_punctuation);
  const auto segment_offsets =
      parakeet::timestamp_utils::get_segment_offsets(word_offsets);

  const double frame_to_seconds =
      window_stride * static_cast<double>(encoder_subsampling_factor);

  std::string result;
  if (timestamp_mode.segment) {
    for (const auto& segment : segment_offsets) {
      const double start = segment.start_offset * frame_to_seconds;
      const double end = segment.end_offset * frame_to_seconds;
      result += std::to_string(start) + "s - " + std::to_string(end) + "s : " +
          segment.text + "\n";
    }
  } else if (timestamp_mode.word) {
    for (const auto& word : word_offsets) {
      const double start = word.start_offset * frame_to_seconds;
      const double end = word.end_offset * frame_to_seconds;
      result += std::to_string(start) + "s - " + std::to_string(end) + "s : " +
          word.text + "\n";
    }
  } else if (timestamp_mode.token) {
    for (const auto& token : tokens_with_text_info) {
      const double start = token.start_offset * frame_to_seconds;
      const double end = token.end_offset * frame_to_seconds;
      result += std::to_string(start) + "s - " + std::to_string(end) + "s : " +
          token.decoded_text + "\n";
    }
  }

  if (result.empty()) {
    return text;
  }
  return result;
}

// Helper to get a string from jstring
std::string jstringToString(JNIEnv* env, jstring jstr) {
  if (jstr == nullptr) {
    return "";
  }
  const char* chars = env->GetStringUTFChars(jstr, nullptr);
  std::string result(chars);
  env->ReleaseStringUTFChars(jstr, chars);
  return result;
}

// Handle struct that holds the Parakeet module state
struct ParakeetModuleHandle {
  std::string model_path;
  std::string tokenizer_path;
  std::string data_path;
  bool loaded = false;
};

} // namespace

// ============================================================================
// JNI Functions
// ============================================================================
extern "C" {

JNIEXPORT jlong JNICALL
Java_org_pytorch_executorch_extension_parakeet_ParakeetModule_nativeCreate(
    JNIEnv* env,
    jobject /* this */,
    jstring modelPath,
    jstring tokenizerPath,
    jstring dataPath) {
  std::string modelPathStr = jstringToString(env, modelPath);
  std::string tokenizerPathStr = jstringToString(env, tokenizerPath);
  std::string dataPathStr = jstringToString(env, dataPath);

  try {
    auto handle = std::make_unique<ParakeetModuleHandle>();
    handle->model_path = modelPathStr;
    handle->tokenizer_path = tokenizerPathStr;
    handle->data_path = dataPathStr;
    handle->loaded = true;

    return reinterpret_cast<jlong>(handle.release());
  } catch (const std::exception& e) {
    ET_LOG(Error, "Failed to create ParakeetModule: %s", e.what());
    env->ThrowNew(
        env->FindClass("java/lang/RuntimeException"),
        ("Failed to create ParakeetModule: " + std::string(e.what())).c_str());
    return 0;
  }
}

JNIEXPORT void JNICALL
Java_org_pytorch_executorch_extension_parakeet_ParakeetModule_nativeDestroy(
    JNIEnv* /* env */,
    jobject /* this */,
    jlong nativeHandle) {
  if (nativeHandle != 0) {
    auto* handle = reinterpret_cast<ParakeetModuleHandle*>(nativeHandle);
    delete handle;
  }
}

JNIEXPORT jstring JNICALL
Java_org_pytorch_executorch_extension_parakeet_ParakeetModule_nativeTranscribe(
    JNIEnv* env,
    jobject /* this */,
    jlong nativeHandle,
    jstring wavPath,
    jstring timestamps) {
  if (nativeHandle == 0) {
    env->ThrowNew(
        env->FindClass("java/lang/IllegalStateException"),
        "Module has been destroyed");
    return nullptr;
  }

  if (wavPath == nullptr) {
    env->ThrowNew(
        env->FindClass("java/lang/IllegalArgumentException"),
        "WAV path cannot be null");
    return nullptr;
  }

  auto* handle = reinterpret_cast<ParakeetModuleHandle*>(nativeHandle);
  std::string wavPathStr = jstringToString(env, wavPath);
  std::string timestampsStr = jstringToString(env, timestamps);

  if (timestampsStr.empty()) {
    timestampsStr = "segment";
  }

  try {
    std::string result = run_parakeet_tdt(
        handle->model_path,
        wavPathStr,
        handle->tokenizer_path,
        handle->data_path,
        timestampsStr);

    return env->NewStringUTF(result.c_str());
  } catch (const std::exception& e) {
    env->ThrowNew(
        env->FindClass("java/lang/RuntimeException"),
        ("Transcription failed: " + std::string(e.what())).c_str());
    return nullptr;
  }
}

} // extern "C"
