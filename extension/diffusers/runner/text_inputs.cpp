/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/diffusers/runner/text_inputs.h>

#include <executorch/runtime/platform/log.h>

#include <nlohmann/json.hpp>
#include <pytorch/tokenizers/hf_tokenizer.h>
#include <pytorch/tokenizers/sentencepiece.h>

#include <algorithm>
#include <cinttypes>
#include <filesystem>
#include <fstream>

namespace executorch::extension::diffusers {

using executorch::runtime::Error;
using executorch::runtime::Result;
using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

json read_json(const fs::path& path) {
  std::ifstream file(path);
  if (!file) {
    return json::object();
  }
  return json::parse(file, nullptr, /*allow_exceptions=*/false);
}

// A token entry is either a string or an AddedToken object.
std::string token_content(const json& token) {
  if (token.is_string()) {
    return token.get<std::string>();
  }
  if (token.is_object() && token.contains("content")) {
    return token.at("content").get<std::string>();
  }
  return "";
}

} // namespace

PromptTokenizer::PromptTokenizer(
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    bool sentencepiece,
    bool add_bos,
    uint64_t pad_id,
    int64_t max_length,
    bool pad_left)
    : tokenizer_(std::move(tokenizer)),
      sentencepiece_(sentencepiece),
      add_bos_(add_bos),
      pad_id_(pad_id),
      max_length_(max_length),
      pad_left_(pad_left) {}

Result<std::unique_ptr<PromptTokenizer>> PromptTokenizer::load(
    const std::string& directory,
    int64_t max_length) {
  const fs::path root(directory);
  std::unique_ptr<::tokenizers::Tokenizer> tokenizer;
  const bool sentencepiece = fs::exists(root / "spiece.model");
  if (sentencepiece) {
    tokenizer = std::make_unique<::tokenizers::SPTokenizer>();
    if (tokenizer->load((root / "spiece.model").string()) !=
        ::tokenizers::Error::Ok) {
      ET_LOG(Error, "Failed to load %s/spiece.model", directory.c_str());
      return Error::InvalidArgument;
    }
  } else if (fs::exists(root / "tokenizer.json")) {
    tokenizer = std::make_unique<::tokenizers::HFTokenizer>();
    if (tokenizer->load(directory) != ::tokenizers::Error::Ok) {
      ET_LOG(Error, "Failed to load %s/tokenizer.json", directory.c_str());
      return Error::InvalidArgument;
    }
  } else {
    ET_LOG(
        Error,
        "%s has no spiece.model or tokenizer.json. A CLIP tokenizer can be "
        "converted with: python -c \"from transformers import "
        "CLIPTokenizerFast as T; T.from_pretrained('%s').save_pretrained('%s')\"",
        directory.c_str(),
        directory.c_str(),
        directory.c_str());
    return Error::InvalidArgument;
  }

  const auto config = read_json(root / "tokenizer_config.json");
  const auto special_tokens = read_json(root / "special_tokens_map.json");
  if (config.is_discarded() || special_tokens.is_discarded()) {
    ET_LOG(Error, "Invalid tokenizer configuration in %s", directory.c_str());
    return Error::InvalidArgument;
  }

  std::string pad_token = token_content(config.value("pad_token", json()));
  if (pad_token.empty()) {
    pad_token = token_content(special_tokens.value("pad_token", json()));
  }
  if (pad_token.empty()) {
    ET_LOG(Error, "%s defines no pad token", directory.c_str());
    return Error::InvalidArgument;
  }
  const auto pad_id = tokenizer->piece_to_id(pad_token);
  if (!pad_id.ok()) {
    ET_LOG(
        Error,
        "Pad token %s is not in the %s vocabulary",
        pad_token.c_str(),
        directory.c_str());
    return Error::InvalidArgument;
  }

  const auto model_max_length = config.find("model_max_length");
  if (model_max_length != config.end() && model_max_length->is_number() &&
      model_max_length->get<double>() < static_cast<double>(max_length)) {
    ET_LOG(
        Error,
        "%s supports %.0f tokens, expected %" PRId64,
        directory.c_str(),
        model_max_length->get<double>(),
        max_length);
    return Error::InvalidArgument;
  }

  const auto add_bos = config.find("add_bos_token");
  const auto padding_side = config.find("padding_side");
  return std::unique_ptr<PromptTokenizer>(new PromptTokenizer(
      std::move(tokenizer),
      sentencepiece,
      add_bos != config.end() && add_bos->is_boolean() && add_bos->get<bool>(),
      pad_id.get(),
      max_length,
      padding_side != config.end() && *padding_side == "left"));
}

Result<std::vector<uint64_t>> PromptTokenizer::tokenize(
    const std::string& text,
    bool add_special_tokens) const {
  // A Hugging Face tokenizer adds what its post-processor template declares;
  // a SentencePiece tokenizer is T5-style and appends the end token.
  const int8_t bos =
      add_special_tokens && (!sentencepiece_ || add_bos_) ? 1 : 0;
  const int8_t eos = add_special_tokens ? 1 : 0;
  auto tokens = tokenizer_->encode(text, bos, eos);
  if (!tokens.ok()) {
    ET_LOG(Error, "Failed to tokenize prompt");
    return Error::InvalidArgument;
  }
  return std::move(tokens.get());
}

Result<TextEncoderInputs> PromptTokenizer::encode(
    const std::string& text) const {
  auto ids = ET_UNWRAP(tokenize(text, /*add_special_tokens=*/true));
  if (static_cast<int64_t>(ids.size()) > max_length_) {
    // Keep the special tokens and drop content tokens from the end. The
    // content starts where the plain encoding lines up with the full one.
    const auto content = ET_UNWRAP(tokenize(text, false));
    const size_t added = ids.size() - content.size();
    ET_CHECK_OR_RETURN_ERROR(
        static_cast<int64_t>(added) < max_length_,
        InvalidArgument,
        "Special tokens do not fit in %" PRId64 " tokens",
        max_length_);
    size_t prefix = 0;
    while (prefix <= added &&
           !std::equal(content.begin(), content.end(), ids.begin() + prefix)) {
      ++prefix;
    }
    ET_CHECK_OR_RETURN_ERROR(
        prefix <= added,
        Internal,
        "Could not locate the prompt between its special tokens");
    std::vector<uint64_t> truncated(ids.begin(), ids.begin() + prefix);
    truncated.insert(
        truncated.end(),
        content.begin(),
        content.begin() + (max_length_ - added));
    truncated.insert(
        truncated.end(), ids.begin() + prefix + content.size(), ids.end());
    ids = std::move(truncated);
  }

  const size_t padding = max_length_ - ids.size();
  TextEncoderInputs inputs;
  inputs.input_ids.assign(ids.begin(), ids.end());
  inputs.attention_mask.assign(ids.size(), 1);
  const auto pad_at =
      pad_left_ ? inputs.input_ids.begin() : inputs.input_ids.end();
  inputs.input_ids.insert(pad_at, padding, static_cast<int64_t>(pad_id_));
  const auto mask_at =
      pad_left_ ? inputs.attention_mask.begin() : inputs.attention_mask.end();
  inputs.attention_mask.insert(mask_at, padding, 0);
  return inputs;
}

} // namespace executorch::extension::diffusers
