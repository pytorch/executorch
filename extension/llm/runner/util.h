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
#include <cctype>
#include <vector>
#if defined(__linux__) || defined(__ANDROID__) || defined(__unix__)
#include <sys/resource.h>
#endif

#define ET_UNWRAP_TOKENIZER(result__)                       \
  ({                                                        \
    auto tk_result__ = (result__);                          \
    if (!tk_result__.ok()) {                                \
      ET_LOG(                                               \
          Error,                                            \
          "Tokenizers error code %d",                       \
          static_cast<uint32_t>(tk_result__.error()));      \
      return ::executorch::runtime::Error::InvalidArgument; \
    }                                                       \
    std::move(*tk_result__);                                \
  })

#define ET_CHECK_TK_OK_OR_RETURN_ERROR(result__, ...)                        \
  ({                                                                         \
    auto tk_result__ = (result__);                                           \
    if (tk_result__ != ::tokenizers::Error::Ok) {                            \
      ET_LOG(                                                                \
          Error, "Tokenizer error: %d", static_cast<uint32_t>(tk_result__)); \
      ET_CHECK_OK_OR_RETURN_ERROR(                                           \
          ::executorch::runtime::Error::InvalidArgument, ##__VA_ARGS__);     \
    }                                                                        \
  })

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
#if defined(__linux__) || defined(__ANDROID__) || defined(__unix__)
  struct rusage r_usage;
  if (getrusage(RUSAGE_SELF, &r_usage) == 0) {
    return r_usage.ru_maxrss * 1024;
  }
#endif // __linux__ || __ANDROID__ || __unix__
  // Unsupported platform like Windows, or getrusage() failed.
  // __APPLE__ and __MACH__ are not supported because r_usage.ru_maxrss does not
  // consistently return kbytes on macOS. On older versions of macOS, it
  // returns bytes, but on newer versions it returns kbytes. Need to figure out
  // when this changed.
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
