/*
 * Copyright (c) 2024 MediaTek Inc.
 *
 * Licensed under the BSD License (the "License"); you may not use this file
 * except in compliance with the License. See the license file in the root
 * directory of this source tree for more details.
 */

#pragma once

#include "llm_helper/include/llm_types.h"

#include <chrono>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace example {
namespace utils {

class Timer {
 public:
  explicit Timer(std::function<void(double)> callback) : mCallback(callback) {}

  void Start() {
    mTimeStart = std::chrono::high_resolution_clock::now();
  }

  void End() {
    const auto time_end = std::chrono::high_resolution_clock::now();
    const double elapsed_time_sec =
        std::chrono::duration_cast<std::chrono::microseconds>(
            time_end - mTimeStart)
            .count() /
        1000000.0;
    mCallback(elapsed_time_sec);
  }

 private:
  std::chrono::high_resolution_clock::time_point mTimeStart;
  std::function<void(double)> mCallback;
};

// Split string via a separator character
static std::vector<std::string> split(const std::string& str, const char sep) {
  std::vector<std::string> tokens;
  std::ostringstream match_pattern;
  match_pattern << "([^" << sep << "]+)";
  const std::regex token_pattern(match_pattern.str());
  std::smatch match;
  auto cur = str.cbegin();
  while (std::regex_search(cur, str.cend(), match, token_pattern)) {
    tokens.push_back(match[0].str());
    cur = match.suffix().first;
  }
  return tokens;
}

static std::string read_file(const std::string& filepath) {
  std::ifstream file(filepath);
  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

template <typename LogitsType>
static uint64_t argmax(const void* logits_buffer, const size_t vocab_size) {
  auto logits = reinterpret_cast<const LogitsType*>(logits_buffer);
  LogitsType max = logits[0];
  uint64_t index = 0;
  for (size_t i = 1; i < vocab_size; i++) {
    if (logits[i] > max) {
      max = logits[i];
      index = i;
    }
  }
  return index;
}

static uint64_t argmax(
    const llm_helper::LLMType logitsType,
    const void* logits_buffer,
    const size_t vocab_size) {
  switch (logitsType) {
    case llm_helper::LLMType::INT16:
      return argmax<int16_t>(logits_buffer, vocab_size);
    case llm_helper::LLMType::FP16:
      return argmax<__fp16>(logits_buffer, vocab_size);
    case llm_helper::LLMType::FP32:
      return argmax<float>(logits_buffer, vocab_size);
    default:
      ET_LOG(
          Error,
          "Unsupported logits type for argmax: %s",
          getLLMTypeName(logitsType));
      return 0;
  }
}

template <typename T>
static std::string to_string(const std::vector<T> vec) {
  std::ostringstream ss;
  auto iter = vec.cbegin();
  ss << "{" << *iter++;
  while (iter != vec.cend()) {
    ss << ", " << *iter++;
  }
  ss << "}";
  return ss.str();
}

} // namespace utils
} // namespace example
