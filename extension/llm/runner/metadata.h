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

inline constexpr const char* kPrefix = "metadata.";

inline constexpr const char* kBosId = "metadata.tokenizer.bos_id";
inline constexpr const char* kEosIds = "metadata.tokenizer.eos_ids";
inline constexpr const char* kMaxSeqLen = "metadata.context.max_seq_len";
inline constexpr const char* kMaxContextLen = "metadata.context.max_context_len";
inline constexpr const char* kVocabSize = "metadata.model.vocab_size";
inline constexpr const char* kUseKVCache = "metadata.model.use_kv_cache";
inline constexpr const char* kEnableDynamicShape =
    "metadata.model.enable_dynamic_shape";
inline constexpr const char* kUseSDPAWithKVCache =
    "metadata.model.use_sdpa_with_kv_cache";
inline constexpr const char* kNLayers = "metadata.model.n_layers";
inline constexpr const char* kChatTemplate = "metadata.tokenizer.chat_template";

// Type tags for self-describing metadata values (must match Python _TAG_* constants)
enum class ValueType : uint8_t {
  Int = 0x01,
  Float = 0x02,
  String = 0x03,
  IntList = 0x04,
  Bytes = 0x05,
};

inline runtime::Result<int64_t> get_int(
    const runtime::NamedDataMap& map,
    const char* key) {
  auto result = map.get_data(key);
  if (!result.ok()) {
    return result.error();
  }
  auto buffer = std::move(result.get());

  const auto* data = static_cast<const uint8_t*>(buffer.data());
  size_t size = buffer.size();

  // New format: 1 byte tag + 8 bytes value
  if (size == 1 + sizeof(int64_t) &&
      static_cast<ValueType>(data[0]) == ValueType::Int) {
    int64_t value;
    std::memcpy(&value, data + 1, sizeof(int64_t));
    buffer.Free();
    return value;
  }
  // Legacy format: 8 bytes value (no tag)
  if (size == sizeof(int64_t)) {
    int64_t value;
    std::memcpy(&value, data, sizeof(int64_t));
    buffer.Free();
    return value;
  }
  buffer.Free();
  return runtime::Error::InvalidArgument;
}

inline runtime::Result<std::string> get_string(
    const runtime::NamedDataMap& map,
    const char* key) {
  auto result = map.get_data(key);
  if (!result.ok()) {
    return result.error();
  }
  auto buffer = std::move(result.get());

  const auto* data = static_cast<const uint8_t*>(buffer.data());
  size_t size = buffer.size();

  // New format: 1 byte tag + string bytes
  if (size >= 1 && static_cast<ValueType>(data[0]) == ValueType::String) {
    std::string value(reinterpret_cast<const char*>(data + 1), size - 1);
    buffer.Free();
    return value;
  }
  // Legacy format: entire buffer is the string
  std::string value(static_cast<const char*>(buffer.data()), size);
  buffer.Free();
  return value;
}

inline runtime::Result<std::vector<int64_t>> get_int_list(
    const runtime::NamedDataMap& map,
    const char* key) {
  auto result = map.get_data(key);
  if (!result.ok()) {
    return result.error();
  }
  auto buffer = std::move(result.get());

  const auto* data = static_cast<const uint8_t*>(buffer.data());
  size_t size = buffer.size();
  size_t offset = 0;

  // New format: 1 byte tag + count(u32) + int64[]
  if (size >= 1 && static_cast<ValueType>(data[0]) == ValueType::IntList) {
    offset = 1;
  }

  if (size - offset < sizeof(uint32_t)) {
    buffer.Free();
    return runtime::Error::InvalidArgument;
  }
  uint32_t count;
  std::memcpy(&count, data + offset, sizeof(uint32_t));
  size_t expected = offset + sizeof(uint32_t) + count * sizeof(int64_t);
  if (size != expected) {
    buffer.Free();
    return runtime::Error::InvalidArgument;
  }
  std::vector<int64_t> values(count);
  std::memcpy(
      values.data(),
      data + offset + sizeof(uint32_t),
      count * sizeof(int64_t));
  buffer.Free();
  return values;
}

} // namespace metadata
} // namespace llm
} // namespace extension
} // namespace executorch
