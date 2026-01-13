#include "timestamp_utils.h"

#include "tokenizer_utils.h"

#include <algorithm>
#include <stdexcept>

#include <pytorch/tokenizers/tokenizer.h>

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

  // NeMo TDT punctuation refinement pass: snap punctuation to the end of the
  // previous token.
  // https://github.com/NVIDIA-NeMo/NeMo/blob/bf583c9/nemo/collections/asr/parts/submodules/rnnt_decoding.py#L1169-L1189
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
    const std::string& word_delimiter_char) {
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

  // Match NeMo behavior: inject first start_offset and append any remaining
  // built tokens as the final word.
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
    const std::vector<std::string>& segment_delimiters,
    const std::optional<int64_t>& segment_gap_threshold) {
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

} // namespace parakeet::timestamp_utils
