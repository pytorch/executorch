/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/extension/llm/runner/constants.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/compiler.h>
#include <stdio.h>
#include <time.h>
#include <algorithm>
#include <cctype>
#include <string>
#include <vector>
#if defined(__linux__) || defined(__ANDROID__) || defined(__unix__)
#include <sys/resource.h>
#endif
#if defined(__linux__) || defined(__ANDROID__)
#include <unistd.h>
#endif

#define ET_UNWRAP_TOKENIZER(result__)                       \
  ({                                                        \
    auto tk_result__ = (result__);                          \
    if (!tk_result__.ok()) {                                \
      ET_LOG(                                               \
          Error,                                            \
          "Tokenizers error code %d",                       \
          static_cast<int>(tk_result__.error()));           \
      return ::executorch::runtime::Error::InvalidArgument; \
    }                                                       \
    std::move(*tk_result__);                                \
  })

// Portable (MSVC-safe) statement form of ET_UNWRAP_TOKENIZER. Declares var__
// in the current scope and assigns the unwrapped value to it. The internal
// result variable is named et_assign_result_##var__ rather than a fixed name
// so that multiple calls in the same scope do not collide with each other.
#define ET_ASSIGN_OR_RETURN_TOKENIZER(var__, result__)       \
  auto et_assign_result_##var__ = (result__);                \
  if (!et_assign_result_##var__.ok()) {                      \
    ET_LOG(                                                  \
        Error,                                               \
        "Tokenizers error code %d",                          \
        static_cast<int>(et_assign_result_##var__.error())); \
    return ::executorch::runtime::Error::InvalidArgument;    \
  }                                                          \
  auto var__ = std::move(*et_assign_result_##var__)

#define ET_CHECK_TK_OK_OR_RETURN_ERROR(result__, ...)                        \
  do {                                                                       \
    auto tk_result__ = (result__);                                           \
    if (tk_result__ != ::tokenizers::Error::Ok) {                            \
      ET_LOG(                                                                \
          Error, "Tokenizer error: %d", static_cast<uint32_t>(tk_result__)); \
      ET_CHECK_OK_OR_RETURN_ERROR(                                           \
          ::executorch::runtime::Error::InvalidArgument, ##__VA_ARGS__);     \
    }                                                                        \
  } while (0)

namespace executorch {
namespace extension {
namespace llm {

ET_EXPERIMENTAL void inline safe_printf(const char* piece) {
  // piece might be a raw byte token, and we only want to print printable chars
  // or whitespace because some of the other bytes can be various control codes,
  // backspace, etc.
  if (piece == nullptr) {
    return;
  }
  if (piece[0] == '\0') {
    return;
  }
  if (piece[1] == '\0') {
    unsigned char byte_val = piece[0];
    if (!(isprint(byte_val) || isspace(byte_val))) {
      return; // bad byte, don't print it
    }
  }
  printf("%s", piece);
}

// Length of the longest prefix of `s` that does not end in the middle of a
// UTF-8 multi-byte sequence. A byte-level tokenizer can emit a token that is
// only part of a character (e.g. one byte of a 3-byte CJK codepoint or emoji),
// so a caller streaming text must hold the incomplete tail until it completes
// rather than decode the partial bytes. An invalid lead byte counts as length 1
// (emitted, so the caller can replace it) rather than stalling output.
ET_EXPERIMENTAL size_t inline utf8_complete_prefix_len(const std::string& s) {
  size_t i = 0;
  const size_t n = s.size();
  while (i < n) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    size_t len;
    if (c < 0x80) {
      len = 1;
    } else if ((c >> 5) == 0x6) {
      len = 2;
    } else if ((c >> 4) == 0xE) {
      len = 3;
    } else if ((c >> 3) == 0x1E) {
      len = 4;
    } else {
      len = 1; // invalid lead byte; emit it and let the caller replace it
    }
    if (i + len > n) {
      break; // incomplete trailing sequence: hold it for more bytes
    }
    i += len;
  }
  return i;
}

ET_EXPERIMENTAL size_t inline utf8_safe_prefix_len(
    const std::string& s,
    size_t len) {
  len = std::min(len, s.size());
  if (len == 0) {
    return 0;
  }
  const auto* data = reinterpret_cast<const unsigned char*>(s.data());
  size_t i = len;
  while (i > 0 && (data[i - 1] & 0xC0) == 0x80) {
    --i;
  }
  if (i == 0) {
    return 0;
  }
  const size_t lead_pos = i - 1;
  const unsigned char lead = data[lead_pos];
  size_t need = 0;
  if (lead < 0x80) {
    need = 1;
  } else if ((lead & 0xE0) == 0xC0) {
    need = 2;
  } else if ((lead & 0xF0) == 0xE0) {
    need = 3;
  } else if ((lead & 0xF8) == 0xF0) {
    need = 4;
  } else {
    return lead_pos;
  }
  return len - lead_pos == need ? len : lead_pos;
}

// How many leading bytes of `text` a streaming consumer may safely emit given a
// set of `stops` strings, and whether a stop was hit (`stop_hit`).
//   * If any stop occurs, returns the byte offset of the EARLIEST occurrence
//   and
//     sets stop_hit=true — text before it is safe; the stop and everything
//     after are dropped (the stop is excluded from output).
//   * Otherwise returns the length minus the longest possible partial-stop tail
//     (max(len(stop))-1 bytes), snapped DOWN to a UTF-8 boundary so a
//     multi-byte character is never split; stop_hit=false. Holding back that
//     conservative tail lets a stop that straddles the next piece still be
//     caught without suffix-prefix matching each stop.
// `text` is expected to be complete-UTF-8 (e.g. the assembled output of
// utf8_complete_prefix_len) and stops are expected to be real text, so a found
// stop offset cannot split a UTF-8 character. Empty `stops` => emit everything,
// no hold-back.
ET_EXPERIMENTAL size_t inline stop_safe_prefix_len(
    const std::string& text,
    const std::vector<std::string>& stops,
    bool& stop_hit) {
  stop_hit = false;
  if (stops.empty()) {
    return text.size();
  }
  size_t earliest = std::string::npos;
  size_t max_len = 0;
  for (const auto& s : stops) {
    if (s.empty()) {
      continue;
    }
    max_len = std::max(max_len, s.size());
    const size_t p = text.find(s);
    if (p != std::string::npos &&
        (earliest == std::string::npos || p < earliest)) {
      earliest = p;
    }
  }
  if (earliest != std::string::npos) {
    stop_hit = true;
    return earliest;
  }
  const size_t hold = max_len > 0 ? max_len - 1 : 0;
  if (text.size() <= hold) {
    return 0;
  }
  return utf8_safe_prefix_len(text, text.size() - hold);
}

// ----------------------------------------------------------------------------
// utilities: time

ET_EXPERIMENTAL long inline time_in_ms() {
  // return time in milliseconds, for benchmarking the model speed
  struct timespec time;
  // The `timespec_get` function is for windows time access. Some AOSP OS does
  // not have timespec_get support.
#if defined(__ANDROID_API__)
  clock_gettime(CLOCK_REALTIME, &time);
#else
  timespec_get(&time, TIME_UTC);
#endif
  return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// utilities: memory usage

// Returns the current RSS in bytes. Returns 0 if not supported.
// RSS: Resident Set Size, the amount of memory currently in the RAM for this
// process. These values are approximate, and are only used for logging
// purposes.
ET_EXPERIMENTAL size_t inline get_rss_bytes() {
#if defined(__linux__) || defined(__ANDROID__)
  // Read current VmRSS from /proc/self/statm (page count in field 1).
  FILE* f = fopen("/proc/self/statm", "r");
  if (f) {
    size_t vm_size = 0, resident = 0;
    if (fscanf(f, "%zu %zu", &vm_size, &resident) == 2) {
      fclose(f);
      return resident * sysconf(_SC_PAGESIZE);
    }
    fclose(f);
  }
#endif // __linux__ || __ANDROID__
  return 0;
}

// Returns the cache position tensor, which can be either a single start_pos
// (when the method_name [`text_decoder` or `forward`] expects a tensor with
// size 1 because model will populate the cache position tensor underneath), or
// a populated tensor for cache position, for the given start_pos and seq_len.
inline runtime::Result<TensorPtr> populate_start_pos_or_cache_position(
    Module* module,
    int64_t& start_pos,
    std::vector<int64_t>& cache_positions_vec,
    int seq_len,
    const char* method_name = "forward") {
  // Get expected shape of cache position tensor, which should be the second
  // argument
  auto method_meta_result = module->method_meta(method_name);
  if (!method_meta_result.ok()) {
    return method_meta_result.error();
  }
  auto method_meta = std::move(*method_meta_result);
  auto second_input_info_result = method_meta.input_tensor_meta(1);
  if (!second_input_info_result.ok()) {
    return second_input_info_result.error();
  }
  auto second_input_info = std::move(*second_input_info_result);
  auto second_input_sizes = second_input_info.sizes();
  auto numel = second_input_sizes[0];

  TensorPtr start_pos_tensor;
  if (numel > 1) {
    // `cache_position` goes from start_pos to start_pos +
    // encoder_output.size(1). e.g. if start_pos = 2 and encoder_output.size(1)
    // = 5, cache_position_tensor should be [2, 3, 4, 5, 6].
    cache_positions_vec.resize(seq_len);
    for (int64_t i = 0; i < seq_len; ++i) {
      cache_positions_vec[i] = start_pos + i;
    }
    return ::executorch::extension::from_blob(
        cache_positions_vec.data(),
        {static_cast<int>(seq_len)},
        executorch::aten::ScalarType::Long);
  } else {
    // Cache position is size 1.
    return ::executorch::extension::from_blob(
        &start_pos, {1}, executorch::aten::ScalarType::Long);
  }
}

/**
 * Helper function to convert a float tensor to bfloat16.
 * Creates a new tensor with bfloat16 dtype and copies/converts the data.
 */
inline ::executorch::runtime::Result<::executorch::extension::TensorPtr>
convert_to_bfloat16(const ::executorch::extension::TensorPtr& src_tensor) {
  ET_CHECK_OR_RETURN_ERROR(
      src_tensor->scalar_type() == ::executorch::aten::ScalarType::Float,
      InvalidArgument,
      "BFloat16 conversion only supported from Float source data");

  const auto num_elements = static_cast<size_t>(src_tensor->numel());
  const float* float_data = src_tensor->const_data_ptr<float>();

  auto bf16_tensor = ::executorch::extension::empty_like(
      src_tensor, ::executorch::aten::ScalarType::BFloat16);
  auto* bf16_data =
      bf16_tensor->mutable_data_ptr<::executorch::aten::BFloat16>();
  for (size_t i = 0; i < num_elements; ++i) {
    bf16_data[i] = ::executorch::aten::BFloat16(float_data[i]);
  }

  return bf16_tensor;
}

} // namespace llm
} // namespace extension
} // namespace executorch

namespace torch {
namespace executor {
namespace util {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::extension::llm::get_rss_bytes;
using ::executorch::extension::llm::safe_printf;
using ::executorch::extension::llm::time_in_ms;
} // namespace util
} // namespace executor
} // namespace torch
