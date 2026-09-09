/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <limits>
#include <optional>

#include <executorch/extension/llm/runner/constants.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/result.h>

namespace executorch {
namespace extension {
namespace llm {

enum class LogitsToKeepMode : std::int64_t {
  Full = 0,
  Last = 1,
  Selected = 2,
};

// Typed view of the metadata a program publishes as constant methods.
//
// Fields are grouped by how read_model_metadata() treats a missing constant
// method:
//   - Required core: read_model_metadata() returns Error::InvalidProgram if the
//     method is absent, so on success these are always populated.
//   - Optional: absent method leaves the field at its default (nullopt here).
//     Add new fields as optional so old programs stay readable; promote to
//     required only as a deliberate, breaking change.
// A present-but-malformed value (non-positive size, out-of-range vocab, unknown
// enum) is always rejected, whether the field is required or optional.
struct ModelMetadata {
  // Required core.
  std::int64_t max_context_length;
  std::int32_t vocab_size;
  aten::ScalarType activation_dtype;
  LogitsToKeepMode logits_to_keep_mode;
  // Optional (defaults to nullopt when the method is absent).
  std::optional<std::int64_t> max_seq_len;
};

inline runtime::Result<ModelMetadata> read_model_metadata(Module& module) {
  const auto names = module.method_names();
  if (!names.ok()) {
    return names.error();
  }

  const auto read_int =
      [&module, &names](
          const char* name) -> runtime::Result<std::optional<std::int64_t>> {
    if (names->count(name) == 0) {
      return std::optional<std::int64_t>{};
    }
    const auto result = module.execute(name);
    if (!result.ok()) {
      return result.error();
    }
    if (result->size() != 1 || !result->at(0).isInt()) {
      return runtime::Error::InvalidProgram;
    }
    return std::optional<std::int64_t>{result->at(0).toInt()};
  };

  ModelMetadata metadata;
  auto max_context_length = read_int(kMaxContextLen);
  if (!max_context_length.ok()) {
    return max_context_length.error();
  }
  if (!*max_context_length || **max_context_length <= 0) {
    return runtime::Error::InvalidProgram;
  }
  metadata.max_context_length = **max_context_length;

  auto vocab_size = read_int(kVocabSize);
  if (!vocab_size.ok()) {
    return vocab_size.error();
  }
  if (!*vocab_size || **vocab_size <= 0 ||
      **vocab_size > std::numeric_limits<std::int32_t>::max()) {
    return runtime::Error::InvalidProgram;
  }
  metadata.vocab_size = static_cast<std::int32_t>(**vocab_size);

  auto activation_dtype = read_int(kActivationDtype);
  if (!activation_dtype.ok()) {
    return activation_dtype.error();
  }
  if (!*activation_dtype) {
    return runtime::Error::InvalidProgram;
  }
  switch (**activation_dtype) {
    case static_cast<std::int64_t>(aten::ScalarType::Half):
      metadata.activation_dtype = aten::ScalarType::Half;
      break;
    case static_cast<std::int64_t>(aten::ScalarType::Float):
      metadata.activation_dtype = aten::ScalarType::Float;
      break;
    case static_cast<std::int64_t>(aten::ScalarType::BFloat16):
      metadata.activation_dtype = aten::ScalarType::BFloat16;
      break;
    default:
      return runtime::Error::InvalidProgram;
  }

  auto logits_to_keep_mode = read_int(kLogitsToKeepMode);
  if (!logits_to_keep_mode.ok()) {
    return logits_to_keep_mode.error();
  }
  if (!*logits_to_keep_mode) {
    return runtime::Error::InvalidProgram;
  }
  switch (**logits_to_keep_mode) {
    case static_cast<std::int64_t>(LogitsToKeepMode::Full):
      metadata.logits_to_keep_mode = LogitsToKeepMode::Full;
      break;
    case static_cast<std::int64_t>(LogitsToKeepMode::Last):
      metadata.logits_to_keep_mode = LogitsToKeepMode::Last;
      break;
    case static_cast<std::int64_t>(LogitsToKeepMode::Selected):
      metadata.logits_to_keep_mode = LogitsToKeepMode::Selected;
      break;
    default:
      return runtime::Error::InvalidProgram;
  }

  auto max_seq_len = read_int(kMaxSeqLen);
  if (!max_seq_len.ok()) {
    return max_seq_len.error();
  }
  if (*max_seq_len && **max_seq_len <= 0) {
    return runtime::Error::InvalidProgram;
  }
  metadata.max_seq_len = *max_seq_len;

  return metadata;
}

inline runtime::Result<std::int32_t> resolve_vocab_size(
    const ModelMetadata& metadata,
    std::int64_t output_vocab_size) {
  if (output_vocab_size <= 0 ||
      output_vocab_size > std::numeric_limits<std::int32_t>::max()) {
    return runtime::Error::InvalidProgram;
  }
  if (metadata.vocab_size != output_vocab_size) {
    return runtime::Error::InvalidProgram;
  }
  return metadata.vocab_size;
}

} // namespace llm
} // namespace extension
} // namespace executorch
