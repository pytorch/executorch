/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cinttypes>
#include <cstdint>
#include <limits>
#include <optional>

#include <executorch/extension/llm/runner/constants.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/log.h>

namespace executorch {
namespace extension {
namespace llm {

enum class LogitsToKeepMode : std::int64_t {
  Full = 0,
  Last = 1,
  Selected = 2,
};

// Readers for the metadata a program publishes as constant methods. Each
// required field's reader returns Error::InvalidProgram (and logs which method)
// if it is absent or malformed (non-positive size, unknown enum). A genuinely
// optional field added later should read through detail::read_int_method
// (nullopt when absent) so old programs stay readable; there are none today.
// vocab_size is returned as published (int64); check_vocab_size() narrows it to
// int32 after cross-checking the forward output.

namespace detail {

// Read a named int constant method: nullopt if absent, the value if present,
// Error::InvalidProgram if it does not evaluate to a single int.
inline runtime::Result<std::optional<std::int64_t>> read_int_method(
    Module& module,
    const char* name) {
  const auto names = ET_UNWRAP(module.method_names());
  if (names.count(name) == 0) {
    return std::optional<std::int64_t>{};
  }
  const auto result = module.execute(name);
  if (!result.ok()) {
    return result.error();
  }
  ET_CHECK_OR_RETURN_ERROR(
      result->size() == 1 && result->at(0).isInt(),
      InvalidProgram,
      "metadata %s must evaluate to a single int",
      name);
  return std::optional<std::int64_t>{result->at(0).toInt()};
}

// A required int constant that must be present and positive.
inline runtime::Result<std::int64_t> read_required_positive_int(
    Module& module,
    const char* name) {
  const auto value = ET_UNWRAP(read_int_method(module, name));
  ET_CHECK_OR_RETURN_ERROR(
      value.has_value(), InvalidProgram, "metadata %s is required", name);
  ET_CHECK_OR_RETURN_ERROR(
      *value > 0,
      InvalidProgram,
      "metadata %s must be positive, got %" PRId64,
      name,
      *value);
  return *value;
}

} // namespace detail

// One reader per constant: the name, its encoding, and its validation together.
// Each rejection logs which constant method was at fault.

inline runtime::Result<std::int64_t> read_max_context_length(Module& module) {
  return detail::read_required_positive_int(module, kMaxContextLen);
}

inline runtime::Result<std::int64_t> read_vocab_size(Module& module) {
  return detail::read_required_positive_int(module, kVocabSize);
}

inline runtime::Result<aten::ScalarType> read_activation_dtype(Module& module) {
  const auto value =
      ET_UNWRAP(detail::read_int_method(module, kActivationDtype));
  ET_CHECK_OR_RETURN_ERROR(
      value.has_value(),
      InvalidProgram,
      "metadata %s is required",
      kActivationDtype);
  switch (*value) {
    case static_cast<std::int64_t>(aten::ScalarType::Half):
      return aten::ScalarType::Half;
    case static_cast<std::int64_t>(aten::ScalarType::Float):
      return aten::ScalarType::Float;
    case static_cast<std::int64_t>(aten::ScalarType::BFloat16):
      return aten::ScalarType::BFloat16;
    default:
      ET_LOG(
          Error,
          "metadata %s has unsupported value %" PRId64,
          kActivationDtype,
          *value);
      return runtime::Error::InvalidProgram;
  }
}

inline runtime::Result<LogitsToKeepMode> read_logits_to_keep_mode(
    Module& module) {
  const auto value =
      ET_UNWRAP(detail::read_int_method(module, kLogitsToKeepMode));
  ET_CHECK_OR_RETURN_ERROR(
      value.has_value(),
      InvalidProgram,
      "metadata %s is required",
      kLogitsToKeepMode);
  switch (*value) {
    case static_cast<std::int64_t>(LogitsToKeepMode::Full):
      return LogitsToKeepMode::Full;
    case static_cast<std::int64_t>(LogitsToKeepMode::Last):
      return LogitsToKeepMode::Last;
    case static_cast<std::int64_t>(LogitsToKeepMode::Selected):
      return LogitsToKeepMode::Selected;
    default:
      ET_LOG(
          Error,
          "metadata %s has unsupported value %" PRId64,
          kLogitsToKeepMode,
          *value);
      return runtime::Error::InvalidProgram;
  }
}

inline runtime::Result<std::int64_t> read_max_seq_len(Module& module) {
  return detail::read_required_positive_int(module, kMaxSeqLen);
}

// Check the published vocab size against the model's actual forward output
// width: reject a disagreement or an out-of-int32 width, and hand back the
// (now int32) value the sampler takes.
inline runtime::Result<std::int32_t> check_vocab_size(
    std::int64_t published_vocab_size,
    std::int64_t output_vocab_size) {
  ET_CHECK_OR_RETURN_ERROR(
      output_vocab_size > 0 &&
          output_vocab_size <= std::numeric_limits<std::int32_t>::max(),
      InvalidProgram,
      "forward output vocab width %" PRId64 " is out of range",
      output_vocab_size);
  ET_CHECK_OR_RETURN_ERROR(
      published_vocab_size == output_vocab_size,
      InvalidProgram,
      "published %s %" PRId64 " disagrees with forward output width %" PRId64,
      kVocabSize,
      published_vocab_size,
      output_vocab_size);
  // Equal to output_vocab_size, already checked to fit int32.
  return static_cast<std::int32_t>(published_vocab_size);
}

} // namespace llm
} // namespace extension
} // namespace executorch
