/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "text_processor.h"

#include <CoreFoundation/CoreFoundation.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <unordered_set>

namespace supertonic {
namespace {

const std::unordered_set<std::string> kLanguages = {
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et",
    "fi", "fr", "hi", "hr", "hu", "id", "it", "lt", "lv", "nl", "pl",
    "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi", "na"};

std::u32string decode_utf8(const std::string& text) {
  std::u32string result;
  for (size_t index = 0; index < text.size();) {
    const auto first = static_cast<unsigned char>(text[index]);
    char32_t codepoint;
    size_t count;
    if (first < 0x80) {
      codepoint = first;
      count = 1;
    } else if ((first & 0xe0) == 0xc0) {
      codepoint = first & 0x1f;
      count = 2;
    } else if ((first & 0xf0) == 0xe0) {
      codepoint = first & 0x0f;
      count = 3;
    } else if ((first & 0xf8) == 0xf0) {
      codepoint = first & 0x07;
      count = 4;
    } else {
      throw std::runtime_error("text is not valid UTF-8");
    }
    if (index + count > text.size()) {
      throw std::runtime_error("text is not valid UTF-8");
    }
    for (size_t offset = 1; offset < count; ++offset) {
      const auto next = static_cast<unsigned char>(text[index + offset]);
      if ((next & 0xc0) != 0x80) {
        throw std::runtime_error("text is not valid UTF-8");
      }
      codepoint = (codepoint << 6) | (next & 0x3f);
    }
    result.push_back(codepoint);
    index += count;
  }
  return result;
}

std::string encode_utf8(const std::u32string& text) {
  std::string result;
  for (char32_t codepoint : text) {
    if (codepoint <= 0x7f) {
      result.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ff) {
      result.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
      result.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else if (codepoint <= 0xffff) {
      result.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
      result.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
      result.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else {
      result.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
      result.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
      result.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
      result.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
  }
  return result;
}

std::string normalize_nfkd(const std::string& text) {
  CFStringRef source = CFStringCreateWithBytes(
      kCFAllocatorDefault,
      reinterpret_cast<const UInt8*>(text.data()),
      text.size(),
      kCFStringEncodingUTF8,
      false);
  if (source == nullptr) {
    throw std::runtime_error("text is not valid UTF-8");
  }
  CFMutableStringRef normalized =
      CFStringCreateMutableCopy(kCFAllocatorDefault, 0, source);
  CFRelease(source);
  CFStringNormalize(normalized, kCFStringNormalizationFormKD);
  const CFIndex length = CFStringGetLength(normalized);
  const CFIndex capacity =
      CFStringGetMaximumSizeForEncoding(length, kCFStringEncodingUTF8) + 1;
  std::vector<char> output(static_cast<size_t>(capacity));
  if (!CFStringGetCString(
          normalized, output.data(), capacity, kCFStringEncodingUTF8)) {
    CFRelease(normalized);
    throw std::runtime_error("failed to encode normalized text");
  }
  CFRelease(normalized);
  return output.data();
}

bool is_emoji(char32_t value) {
  return (value >= 0x1f300 && value <= 0x1f64f) ||
      (value >= 0x1f680 && value <= 0x1f6ff) ||
      (value >= 0x1f700 && value <= 0x1faff) ||
      (value >= 0x2600 && value <= 0x27bf) ||
      (value >= 0x1f1e6 && value <= 0x1f1ff);
}

bool is_unicode_whitespace(char32_t value) {
  return CFCharacterSetIsLongCharacterMember(
      CFCharacterSetGetPredefined(kCFCharacterSetWhitespaceAndNewline),
      static_cast<UTF32Char>(value));
}

std::string canonicalize_whitespace(
    const std::string& value,
    bool preserve_newlines) {
  auto decoded = decode_utf8(value);
  for (char32_t& codepoint : decoded) {
    if (is_unicode_whitespace(codepoint) &&
        !(preserve_newlines && codepoint == U'\n')) {
      codepoint = U' ';
    }
  }
  return encode_utf8(decoded);
}

void replace_all(
    std::string& text,
    const std::string& old_value,
    const std::string& new_value) {
  for (size_t position = 0;
       (position = text.find(old_value, position)) != std::string::npos;
       position += new_value.size()) {
    text.replace(position, old_value.size(), new_value);
  }
}

std::string trim(const std::string& value) {
  const auto decoded = decode_utf8(value);
  size_t first = 0;
  while (first < decoded.size() && is_unicode_whitespace(decoded[first])) {
    ++first;
  }
  size_t last = decoded.size();
  while (last > first && is_unicode_whitespace(decoded[last - 1])) {
    --last;
  }
  return encode_utf8(decoded.substr(first, last - first));
}

std::string collapse_whitespace(const std::string& value) {
  std::u32string result;
  bool pending_space = false;
  for (char32_t character : decode_utf8(value)) {
    if (is_unicode_whitespace(character)) {
      pending_space = !result.empty();
    } else {
      if (pending_space) {
        result.push_back(U' ');
      }
      result.push_back(character);
      pending_space = false;
    }
  }
  return encode_utf8(result);
}

bool has_terminal_punctuation(const std::u32string& text) {
  static const std::u32string terminals = U".!?！？，;:,'\"')]}…。」』】〉》›»";
  return !text.empty() && terminals.find(text.back()) != std::u32string::npos;
}

bool is_abbreviation(const std::string& prefix) {
  static const std::vector<std::string> suffixes = {
      "Mr.",
      "Mrs.",
      "Ms.",
      "Dr.",
      "Prof.",
      "Sr.",
      "Jr.",
      "Ph.D.",
      "etc.",
      "e.g.",
      "i.e.",
      "vs.",
      "Inc.",
      "Ltd.",
      "Co.",
      "Corp.",
      "St.",
      "Ave.",
      "Blvd."};
  for (const auto& suffix : suffixes) {
    if (prefix.size() >= suffix.size() &&
        prefix.compare(prefix.size() - suffix.size(), suffix.size(), suffix) ==
            0) {
      return true;
    }
  }
  return prefix.size() >= 2 && prefix.back() == '.' &&
      std::isupper(static_cast<unsigned char>(prefix[prefix.size() - 2])) &&
      (prefix.size() == 2 ||
       !std::isalpha(static_cast<unsigned char>(prefix[prefix.size() - 3])));
}

bool is_cjk_sentence_terminal(char32_t character) {
  return character == U'。' || character == U'！' || character == U'？';
}

std::vector<std::string> split_sentences(const std::string& paragraph) {
  const std::u32string decoded = decode_utf8(paragraph);
  std::vector<std::string> result;
  size_t start = 0;
  for (size_t index = 0; index < decoded.size(); ++index) {
    const char32_t character = decoded[index];
    const bool ascii_terminal =
        character == U'.' || character == U'!' || character == U'?';
    const bool cjk_terminal = is_cjk_sentence_terminal(character);
    if (!ascii_terminal && !cjk_terminal) {
      continue;
    }
    const std::string prefix =
        encode_utf8(decoded.substr(start, index - start + 1));
    if (ascii_terminal && is_abbreviation(prefix)) {
      continue;
    }
    size_t next = index + 1;
    if (cjk_terminal && next < decoded.size() &&
        is_cjk_sentence_terminal(decoded[next])) {
      continue;
    }
    if (ascii_terminal &&
        (next == decoded.size() || !is_unicode_whitespace(decoded[next]))) {
      continue;
    }
    while (next < decoded.size() && is_unicode_whitespace(decoded[next])) {
      ++next;
    }
    result.push_back(trim(prefix));
    start = next;
    index = next == 0 ? 0 : next - 1;
  }
  if (start < decoded.size()) {
    result.push_back(trim(encode_utf8(decoded.substr(start))));
  }
  return result;
}

} // namespace

void validate_language(const std::string& language) {
  if (kLanguages.count(language) == 0) {
    throw std::invalid_argument("Invalid language: " + language);
  }
}

std::string preprocess_text(
    const std::string& text,
    const std::string& language) {
  std::u32string values = decode_utf8(normalize_nfkd(text));
  std::u32string cleaned;
  for (char32_t value : values) {
    if (is_emoji(value) || value == U'♥' || value == U'☆' || value == U'♡' ||
        value == U'©' || value == U'\\') {
      continue;
    }
    switch (value) {
      case U'–':
      case U'‑':
      case U'—':
        cleaned.push_back(U'-');
        break;
      case U'_':
      case U'[':
      case U']':
      case U'|':
      case U'/':
      case U'#':
      case U'→':
      case U'←':
        cleaned.push_back(U' ');
        break;
      case U'“':
      case U'”':
        cleaned.push_back(U'"');
        break;
      case U'‘':
      case U'’':
      case U'´':
      case U'`':
        cleaned.push_back(U'\'');
        break;
      default:
        cleaned.push_back(value);
    }
  }
  std::string result = encode_utf8(cleaned);
  replace_all(result, "@", " at ");
  replace_all(result, "e.g.,", "for example, ");
  replace_all(result, "i.e.,", "that is, ");
  for (const char punctuation : std::string(",.!?;:'")) {
    replace_all(
        result, std::string(" ") + punctuation, std::string(1, punctuation));
  }
  while (result.find("\"\"") != std::string::npos) {
    replace_all(result, "\"\"", "\"");
  }
  while (result.find("''") != std::string::npos) {
    replace_all(result, "''", "'");
  }
  result = collapse_whitespace(result);
  if (!has_terminal_punctuation(decode_utf8(result))) {
    result.push_back('.');
  }
  validate_language(language);
  return "<" + language + ">" + result + "</" + language + ">";
}

std::vector<std::string> chunk_text(
    const std::string& text,
    size_t max_length) {
  const std::string normalized_text = canonicalize_whitespace(text, true);
  std::vector<std::string> paragraphs;
  size_t start = 0;
  for (size_t index = 0; index < normalized_text.size();) {
    if (normalized_text[index] != '\n') {
      ++index;
      continue;
    }
    size_t next = index;
    int newlines = 0;
    while (next < normalized_text.size() &&
           (normalized_text[next] == '\n' || normalized_text[next] == ' ')) {
      if (normalized_text[next] == '\n') {
        ++newlines;
      }
      ++next;
    }
    if (newlines >= 2) {
      const auto paragraph = trim(normalized_text.substr(start, index - start));
      if (!paragraph.empty()) {
        paragraphs.push_back(paragraph);
      }
      start = next;
    }
    index = next;
  }
  const auto final_paragraph = trim(normalized_text.substr(start));
  if (!final_paragraph.empty()) {
    paragraphs.push_back(final_paragraph);
  }

  std::vector<std::string> chunks;
  for (const auto& paragraph : paragraphs) {
    std::string current;
    for (const auto& sentence : split_sentences(paragraph)) {
      const size_t packed_length =
          decode_utf8(current).size() + decode_utf8(sentence).size() + 1;
      if (packed_length <= max_length) {
        current += (current.empty() ? "" : " ") + sentence;
      } else {
        if (!current.empty()) {
          chunks.push_back(trim(current));
        }
        current = sentence;
      }
    }
    if (!current.empty()) {
      chunks.push_back(trim(current));
    }
  }
  return chunks;
}

std::vector<std::string> chunk_text_for_language(
    const std::string& text,
    const std::string& language) {
  return chunk_text(text, language == "ko" || language == "ja" ? 120 : 300);
}

UnicodeProcessor::UnicodeProcessor(const std::string& indexer_path) {
  std::ifstream file(indexer_path);
  if (!file) {
    throw std::runtime_error("failed to open Unicode indexer: " + indexer_path);
  }
  const nlohmann::json data = nlohmann::json::parse(file);
  if (!data.is_array()) {
    throw std::runtime_error("Unicode indexer must be a JSON array");
  }
  indexer_.reserve(data.size());
  for (const auto& token : data) {
    if (!token.is_number_integer()) {
      throw std::runtime_error("Unicode indexer tokens must be integers");
    }
    indexer_.push_back(token.get<int64_t>());
  }
}

void UnicodeProcessor::configure_vocabulary(int64_t vocabulary_size) {
  if (vocabulary_size <= 0) {
    throw std::invalid_argument("text vocabulary size must be positive");
  }
  for (int64_t token_id : indexer_) {
    if (token_id < -1 || token_id >= vocabulary_size) {
      throw std::runtime_error(
          "Unicode indexer contains a token outside the text vocabulary");
    }
  }
  vocabulary_size_ = vocabulary_size;
}

TextBatch UnicodeProcessor::process(
    const std::vector<std::string>& texts,
    const std::vector<std::string>& languages) const {
  if (vocabulary_size_ <= 0) {
    throw std::runtime_error("Unicode processor vocabulary is not configured");
  }
  if (texts.size() != languages.size()) {
    throw std::invalid_argument(
        "texts and languages must have the same cardinality");
  }
  if (texts.empty()) {
    throw std::invalid_argument("expected at least one text and language");
  }
  std::vector<std::u32string> processed;
  size_t max_length = 0;
  for (size_t index = 0; index < texts.size(); ++index) {
    processed.push_back(
        decode_utf8(preprocess_text(texts[index], languages[index])));
    max_length = std::max(max_length, processed.back().size());
  }
  TextBatch batch;
  batch.shape = {
      static_cast<int64_t>(texts.size()), static_cast<int64_t>(max_length)};
  batch.ids.assign(texts.size() * max_length, 0);
  batch.mask.assign(texts.size() * max_length, 0.0f);
  for (size_t row = 0; row < processed.size(); ++row) {
    for (size_t column = 0; column < processed[row].size(); ++column) {
      const auto codepoint = static_cast<size_t>(processed[row][column]);
      if (codepoint >= indexer_.size()) {
        throw std::runtime_error(
            "Unicode indexer has no entry for codepoint " +
            std::to_string(codepoint));
      }
      const int64_t token_id = indexer_[codepoint];
      if (token_id < 0) {
        throw std::invalid_argument(
            "unsupported Unicode codepoint " + std::to_string(codepoint));
      }
      batch.ids[row * max_length + column] = token_id;
      batch.mask[row * max_length + column] = 1.0f;
    }
  }
  return batch;
}

} // namespace supertonic
