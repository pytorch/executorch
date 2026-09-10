/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_set>

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <pytorch/tokenizers/tokenizer.h>

namespace executorch {
namespace backends {
namespace mlx {
namespace examples {
namespace llm {

struct StopTokens {
  std::unordered_set<uint64_t> ids;
  std::optional<uint64_t> turn_end_id;
};

inline const char* turn_end_piece(const std::string& chat) {
  if (chat == "llama3") {
    return "<|eot_id|>";
  }
  if (chat == "gemma") {
    return "<end_of_turn>";
  }
  if (chat == "gemma4") {
    return "<turn|>";
  }
  return nullptr;
}

inline bool resolve_stop_tokens(
    tokenizers::Tokenizer& tokenizer,
    ::executorch::extension::Module& module,
    const std::string& chat,
    StopTokens& out) {
  out.ids = ::executorch::extension::llm::get_eos_ids(&tokenizer, &module);
  out.turn_end_id.reset();
  if (chat == "0") {
    return true;
  }
  const char* piece = turn_end_piece(chat);
  if (piece == nullptr) {
    return false;
  }
  auto id = tokenizer.piece_to_id(piece);
  if (!id.ok()) {
    return false;
  }
  out.turn_end_id = *id;
  out.ids.insert(*id);
  return true;
}

inline bool wrap_turn(
    const std::string& chat,
    const std::string& prompt,
    bool with_bos,
    std::string& out) {
  if (chat == "0") {
    out = prompt;
  } else if (chat == "llama3") {
    out = std::string(with_bos ? "<|begin_of_text|>" : "") +
        "<|start_header_id|>user<|end_header_id|>\n\n" + prompt +
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
  } else if (chat == "gemma") {
    out = std::string(with_bos ? "<bos>" : "") + "<start_of_turn>user\n" +
        prompt + "<end_of_turn>\n<start_of_turn>model\n";
  } else if (chat == "gemma4") {
    out = std::string(with_bos ? "<bos>" : "") + "<|turn>user\n" + prompt +
        "<turn|>\n<|turn>model\n";
  } else {
    return false;
  }
  return true;
}

inline int storage_dtype(const std::string& name) {
  using ScalarType = ::executorch::runtime::etensor::ScalarType;
  if (name == "bf16") {
    return static_cast<int>(ScalarType::BFloat16);
  }
  if (name == "fp16") {
    return static_cast<int>(ScalarType::Half);
  }
  if (name == "fp32") {
    return static_cast<int>(ScalarType::Float);
  }
  return -1;
}

inline int resolve_kv_storage_dtype(
    const std::string& override_name,
    ::executorch::aten::ScalarType activation_dtype) {
  if (!override_name.empty()) {
    return storage_dtype(override_name);
  }
  return static_cast<int>(activation_dtype);
}

} // namespace llm
} // namespace examples
} // namespace mlx
} // namespace backends
} // namespace executorch
