/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/named_data_map.h>
#include <executorch/runtime/core/result.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace executorch {
namespace extension {
namespace llm {
namespace metadata {

// Type tag constants (1-byte prefix on every encoded value).
// Must stay in sync with Python (extension/llm/export/metadata.py).
inline constexpr uint8_t kTagInt = 0x01;
inline constexpr uint8_t kTagFloat = 0x02;
inline constexpr uint8_t kTagString = 0x03;
inline constexpr uint8_t kTagIntList = 0x04;
inline constexpr uint8_t kTagBytes = 0x05;

inline constexpr const char* kPrefix = "metadata.";

inline constexpr const char* kBosId = "metadata.tokenizer.bos_id";
inline constexpr const char* kEosIds = "metadata.tokenizer.eos_ids";
inline constexpr const char* kMaxSeqLen = "metadata.context.max_seq_len";
inline constexpr const char* kMaxContextLen =
    "metadata.context.max_context_len";
inline constexpr const char* kVocabSize = "metadata.model.vocab_size";
inline constexpr const char* kUseKVCache = "metadata.model.use_kv_cache";
inline constexpr const char* kNLayers = "metadata.model.n_layers";
inline constexpr const char* kChatTemplate = "metadata.tokenizer.chat_template";

namespace detail {

inline runtime::Error check_tag(
    const void* data,
    size_t size,
    uint8_t expected_tag) {
  if (size == 0) {
    return runtime::Error::InvalidArgument;
  }
  uint8_t actual_tag;
  std::memcpy(&actual_tag, data, 1);
  if (actual_tag != expected_tag) {
    return runtime::Error::InvalidArgument;
  }
  return runtime::Error::Ok;
}

} // namespace detail

inline runtime::Result<int64_t> get_int(
    const runtime::NamedDataMap& map,
    const char* key) {
  auto result = map.get_data(key);
  if (!result.ok()) {
    return result.error();
  }
  auto buffer = std::move(result.get());
  if (buffer.size() != 1 + sizeof(int64_t)) {
    return runtime::Error::InvalidArgument;
  }
  auto err = detail::check_tag(buffer.data(), buffer.size(), kTagInt);
  if (err != runtime::Error::Ok) {
    return err;
  }
  int64_t value;
  std::memcpy(
      &value, static_cast<const char*>(buffer.data()) + 1, sizeof(int64_t));
  return value;
}

inline runtime::Result<double> get_float(
    const runtime::NamedDataMap& map,
    const char* key) {
  auto result = map.get_data(key);
  if (!result.ok()) {
    return result.error();
  }
  auto buffer = std::move(result.get());
  if (buffer.size() != 1 + sizeof(double)) {
    return runtime::Error::InvalidArgument;
  }
  auto err = detail::check_tag(buffer.data(), buffer.size(), kTagFloat);
  if (err != runtime::Error::Ok) {
    return err;
  }
  double value;
  std::memcpy(
      &value, static_cast<const char*>(buffer.data()) + 1, sizeof(double));
  return value;
}

inline runtime::Result<std::string> get_string(
    const runtime::NamedDataMap& map,
    const char* key) {
  auto result = map.get_data(key);
  if (!result.ok()) {
    return result.error();
  }
  auto buffer = std::move(result.get());
  if (buffer.size() < 1) {
    return runtime::Error::InvalidArgument;
  }
  auto err = detail::check_tag(buffer.data(), buffer.size(), kTagString);
  if (err != runtime::Error::Ok) {
    return err;
  }
  std::string value(
      static_cast<const char*>(buffer.data()) + 1, buffer.size() - 1);
  return value;
}

inline runtime::Result<std::vector<uint8_t>> get_bytes(
    const runtime::NamedDataMap& map,
    const char* key) {
  auto result = map.get_data(key);
  if (!result.ok()) {
    return result.error();
  }
  auto buffer = std::move(result.get());
  if (buffer.size() < 1) {
    return runtime::Error::InvalidArgument;
  }
  auto err = detail::check_tag(buffer.data(), buffer.size(), kTagBytes);
  if (err != runtime::Error::Ok) {
    return err;
  }
  const auto* begin = static_cast<const uint8_t*>(buffer.data()) + 1;
  return std::vector<uint8_t>(begin, begin + buffer.size() - 1);
}

inline runtime::Result<std::vector<int64_t>> get_int_list(
    const runtime::NamedDataMap& map,
    const char* key) {
  auto result = map.get_data(key);
  if (!result.ok()) {
    return result.error();
  }
  auto buffer = std::move(result.get());
  if (buffer.size() < 1 + sizeof(uint32_t)) {
    return runtime::Error::InvalidArgument;
  }
  auto err = detail::check_tag(buffer.data(), buffer.size(), kTagIntList);
  if (err != runtime::Error::Ok) {
    return err;
  }
  const char* base = static_cast<const char*>(buffer.data()) + 1;
  size_t payload_size = buffer.size() - 1;

  uint32_t count;
  std::memcpy(&count, base, sizeof(uint32_t));
  size_t data_size = payload_size - sizeof(uint32_t);
  if (data_size / sizeof(int64_t) != count || data_size % sizeof(int64_t) != 0) {
    return runtime::Error::InvalidArgument;
  }
  std::vector<int64_t> values(count);
  if (count > 0) {
    std::memcpy(values.data(), base + sizeof(uint32_t), count * sizeof(int64_t));
  }
  return values;
}

} // namespace metadata
} // namespace llm
} // namespace extension
} // namespace executorch
