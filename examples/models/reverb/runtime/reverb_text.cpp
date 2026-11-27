/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "reverb_text.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>

namespace reverb {
namespace {

const char* const kLanguages[] = {"de", "en", "es", "fr", "it", "pl", "pt"};

// Decodes one UTF-8 code point at `i`, advancing it. A malformed sequence
// decodes as its first byte alone.
uint32_t next_code_point(const std::string& text, size_t& i) {
  const auto byte = [&](size_t at) { return static_cast<uint8_t>(text[at]); };
  const uint8_t lead = byte(i);
  int extra = 0;
  uint32_t code = lead;
  if (lead >= 0xC0 && lead < 0xE0) {
    extra = 1;
    code = lead & 0x1F;
  } else if (lead >= 0xE0 && lead < 0xF0) {
    extra = 2;
    code = lead & 0x0F;
  } else if (lead >= 0xF0 && lead < 0xF8) {
    extra = 3;
    code = lead & 0x07;
  }
  if (i + extra >= text.size() && extra > 0) {
    ++i;
    return lead;
  }
  for (int k = 1; k <= extra; ++k) {
    if ((byte(i + k) & 0xC0) != 0x80) {
      ++i;
      return lead;
    }
    code = (code << 6) | (byte(i + k) & 0x3F);
  }
  i += extra + 1;
  return code;
}

bool in(uint32_t c, uint32_t lo, uint32_t hi) {
  return c >= lo && c <= hi;
}

// The weight of one code point by its Unicode general category, covering the
// scripts of Reverb's languages. Anything not listed counts as a letter.
double code_point_weight(uint32_t c) {
  // Z: separators.
  if (c == 0x20 || c == 0xA0 || c == 0x1680 || in(c, 0x2000, 0x200A) ||
      c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000) {
    return 0.2;
  }
  // M: combining marks.
  if (in(c, 0x0300, 0x036F) || in(c, 0x1AB0, 0x1AFF) || in(c, 0x1DC0, 0x1DFF) ||
      in(c, 0x20D0, 0x20FF) || in(c, 0xFE20, 0xFE2F)) {
    return 0.0;
  }
  // N: numbers.
  if (in(c, '0', '9') || c == 0xB2 || c == 0xB3 || c == 0xB9 ||
      in(c, 0xBC, 0xBE) || in(c, 0x2070, 0x2079) || in(c, 0x2080, 0x2089) ||
      in(c, 0x2150, 0x2189) || in(c, 0x2460, 0x249B) || in(c, 0x24EA, 0x24FF)) {
    return 3.5;
  }
  // P and S: punctuation and symbols.
  if (c > 0x20 && c < 0x7F && !std::isalnum(static_cast<int>(c))) {
    return 0.5;
  }
  if ((in(c, 0xA1, 0xBF) && c != 0xAA && c != 0xAD && c != 0xB5 && c != 0xBA) ||
      c == 0xD7 || c == 0xF7 || in(c, 0x2010, 0x2027) ||
      in(c, 0x2030, 0x205E) || in(c, 0x207A, 0x207E) || in(c, 0x208A, 0x208E) ||
      in(c, 0x20A0, 0x20C0) || in(c, 0x2190, 0x23FF) || in(c, 0x2500, 0x27BF) ||
      in(c, 0x2E00, 0x2E5D) || in(c, 0x3001, 0x3003) || in(c, 0x3008, 0x3011)) {
    return 0.5;
  }
  return 1.0;
}

bool is_space(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
      c == '\v';
}

// Splits after every run of whitespace that follows '.', '!' or '?'.
std::vector<std::string> split_sentences(const std::string& text) {
  std::vector<std::string> sentences;
  size_t start = 0;
  size_t i = 0;
  while (i < text.size()) {
    const char c = text[i];
    if ((c == '.' || c == '!' || c == '?') && i + 1 < text.size() &&
        is_space(text[i + 1])) {
      size_t end = i + 1;
      size_t next = end;
      while (next < text.size() && is_space(text[next])) {
        ++next;
      }
      sentences.push_back(text.substr(start, end - start));
      start = next;
      i = next;
      continue;
    }
    ++i;
  }
  sentences.push_back(text.substr(start));
  return sentences;
}

} // namespace

bool is_supported_language(const std::string& language) {
  return std::any_of(
      std::begin(kLanguages), std::end(kLanguages), [&](const char* code) {
        return language == code;
      });
}

std::string build_prompt(
    const std::string& text,
    const std::string& reference_text,
    const std::string& language) {
  return "<|denoise|><|lang_start|>" + language +
      "<|lang_end|><|instruct_start|>None<|instruct_end|><|text_start|>" +
      reference_text + " " + text + "<|text_end|>";
}

double spoken_weight(const std::string& text) {
  double weight = 0.0;
  size_t i = 0;
  while (i < text.size()) {
    weight += code_point_weight(next_code_point(text, i));
  }
  return weight;
}

int estimate_frames(
    const std::string& text,
    const std::string& reference_text,
    int reference_frames) {
  const double reference_weight = spoken_weight(reference_text);
  if (reference_weight <= 0.0 || reference_frames <= 0) {
    return 1;
  }
  double frames = spoken_weight(text) / (reference_weight / reference_frames);
  if (frames < 50.0) {
    frames = 50.0 * std::cbrt(frames / 50.0);
  }
  return std::max(1, static_cast<int>(frames));
}

std::vector<std::string> split_text(
    const std::string& text,
    const std::string& reference_text,
    int reference_frames,
    int longest,
    int piece) {
  if (estimate_frames(text, reference_text, reference_frames) <= longest) {
    return {text};
  }
  std::vector<std::string> pieces;
  for (const auto& sentence : split_sentences(text)) {
    if (!pieces.empty()) {
      const std::string joined = pieces.back() + " " + sentence;
      if (estimate_frames(joined, reference_text, reference_frames) <= piece) {
        pieces.back() = joined;
        continue;
      }
    }
    pieces.push_back(sentence);
  }
  return pieces;
}

} // namespace reverb
