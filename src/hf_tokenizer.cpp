/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#include <pytorch/tokenizers/hf_tokenizer.h>

// Standard
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Third Party
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace tokenizers {

// -------------------------private method end-------------------------------
// -------------------------public method start-------------------------------

Error HFTokenizer::load(const std::string& path) {
  // If this is a directory, look for tokenizer.json and tokenizer_config.json
  std::string model_json = path;
  std::string model_config_json = "";
  if (fs::is_directory(path)) {
    const fs::path root(path);
    model_json = root / "tokenizer.json";
    if (!fs::exists(model_json)) {
      fprintf(stderr, "no tokenizer.json found in %s\n", path.c_str());
      return Error::LoadFailure;
    }
    const auto model_config_json_path = root / "tokenizer_config.json";
    if (fs::exists(model_config_json_path)) {
      model_config_json = model_config_json_path;
    }
  }

  // Load the tokenizer.json file
  std::ifstream file(model_json);
  if (!file) {
    fprintf(stderr, "failed to open encoder file: %s\n", path.c_str());
    return Error::LoadFailure;
  }
  std::string contents(
      (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  json parsed_json;
  try {
    parsed_json = json::parse(contents);
  } catch (const json::exception& e) {
    std::cerr << "Error parsing json file: " << e.what() << std::endl;
    return Error::LoadFailure;
  }

  // Parse the special tokens
  try {
    std::vector<std::pair<std::string, std::uint64_t>> special_token_pairs;
    const auto& special_tokens = parsed_json.at("added_tokens");
    auto special_token_map = TK_UNWRAP(detail::buildTokenMap(
        special_tokens,
        [](const auto& it) -> std::string { return it.at("content"); },
        [](const auto& it) -> std::uint64_t { return it.at("id"); }));

    // Create special token regex to help later with encoding.
    special_token_regex_ =
        TK_UNWRAP(detail::build_special_token_regex(special_token_map));

    // Store for future use.
    special_token_map_.emplace(std::move(special_token_map));
  } catch (const json::out_of_range& e) {
    fprintf(stderr, "Could not parse special tokens: %s\n", e.what());
    return Error::LoadFailure;
  }

  // Parse the standard tokens
  try {
    std::vector<std::pair<std::string, std::uint64_t>> token_pairs;
    const auto& vocab = parsed_json.at("/model/vocab"_json_pointer);
    for (const auto& entry : vocab.items()) {
      const std::string token = entry.key();
      const uint64_t token_id = entry.value();
      // Skip adding special tokens to the standard encoder/decoder
      if (!special_token_map_->tryGetString(token_id)) {
        token_pairs.emplace_back(token, token_id);
      }
    }

    auto token_map = TK_UNWRAP(detail::buildTokenMap(std::move(token_pairs)));
    token_map_.emplace(std::move(token_map));
  } catch (const json::out_of_range& e) {
    fprintf(stderr, "Could not parse tokens: %s\n", e.what());
    return Error::LoadFailure;
  }

  // Set the vocab size to include special tokens
  vocab_size_ = token_map_->size() + special_token_map_->size();

  // Set up the pre-tokenizer
  try {
    std::cout << "Setting up pretokenizer..." << std::endl;
    _pretokenizer = PreTokenizerConfig()
                        .parse_json(parsed_json.at("pre_tokenizer"))
                        .create();
    std::cout << "Pretokenizer set up" << std::endl;
  } catch (const json::out_of_range& e) {
    fprintf(stderr, "Could not parse pre_tokenizer: %s\n", e.what());
    return Error::LoadFailure;
  }

  // Set up the decoder (optional)
  try {
    _decoder =
        TokenDecoderConfig().parse_json(parsed_json.at("decoder")).create();
  } catch (const json::out_of_range& e) {
    // No decoder specified
  }

  // TODO: Do we need to parse the merges?

  // If a tokenizer config file is found, parse it to look up the eos/bos tokens
  if (!model_config_json.empty()) {
    // Load it and parse it as json
    std::ifstream config_file(model_config_json);
    if (!config_file) {
      fprintf(stderr, "failed to open encoder file: %s\n", path.c_str());
      return Error::LoadFailure;
    }
    std::string config_contents(
        (std::istreambuf_iterator<char>(config_file)),
        std::istreambuf_iterator<char>());
    json parsed_config_json;
    try {
      parsed_config_json = json::parse(config_contents);
    } catch (const json::exception& e) {
      std::cerr << "Error parsing model config json json file: " << e.what()
                << std::endl;
      return Error::LoadFailure;
    }

    // Pull out the token strings
    try {
      const std::string bos_token = parsed_config_json.contains("bos_token") &&
              !parsed_config_json["bos_token"].is_null()
          ? parsed_config_json["bos_token"].get<std::string>()
          : "";

      const std::string eos_token = parsed_config_json.contains("eos_token") &&
              !parsed_config_json["eos_token"].is_null()
          ? parsed_config_json["eos_token"].get<std::string>()
          : "";
      const auto bos_res = special_token_map_->tryGetInteger(bos_token);
      const auto eos_res = special_token_map_->tryGetInteger(eos_token);
      if (!bos_res) {
        fprintf(
            stderr, "BOS token %s not in special tokens\n", bos_token.c_str());
        return Error::LoadFailure;
      }
      if (!eos_res) {
        fprintf(
            stderr, "EOS token %s not in special tokens\n", eos_token.c_str());
        return Error::LoadFailure;
      }
      bos_tok_ = *bos_res;
      eos_tok_ = *eos_res;
    } catch (const json::out_of_range& e) {
      fprintf(
          stderr, "Could not eos/bos from tokenizer config: %s\n", e.what());
      return Error::LoadFailure;
    }
  }

  // Otherwise, make an educated guess with the following logic:
  // 1. Look for special tokens with "bos"/"begin" or "eos"/"end" in them
  // 2. Sub-qualify with the word "text" if needed
  // 3. If EOS found, but BOS is not (or vice versa), assume they are the same
  else {
    std::vector<std::string_view> bos_candidates;
    std::vector<std::string_view> eos_candidates;
    for (std::size_t token_idx = 0; token_idx < special_token_map_->size();
         ++token_idx) {
      const auto [token, _] = special_token_map_->getElement(token_idx);
      if (token.find("bos") != std::string::npos ||
          token.find("begin") != std::string::npos) {
        bos_candidates.push_back(token);
      }
      if (token.find("eos") != std::string::npos ||
          token.find("end") != std::string::npos) {
        eos_candidates.push_back(token);
      }
    }

    if (bos_candidates.size() > 1) {
      const auto orig_candidates = std::move(bos_candidates);
      bos_candidates.clear();
      for (const auto& cand : orig_candidates) {
        if (cand.find("text") != std::string::npos) {
          bos_candidates.push_back(cand);
        }
      }
    }
    if (eos_candidates.size() > 1) {
      const auto orig_candidates = std::move(eos_candidates);
      eos_candidates.clear();
      for (const auto& cand : orig_candidates) {
        if (cand.find("text") != std::string::npos) {
          eos_candidates.push_back(cand);
        }
      }
    }

    // Use if a single candidate
    bool bos_found = false;
    bool eos_found = false;
    if (bos_candidates.size() == 1) {
      bos_found = true;
      bos_tok_ = *(special_token_map_->tryGetInteger(bos_candidates[0]));
    }
    if (eos_candidates.size() == 1) {
      eos_found = true;
      eos_tok_ = *(special_token_map_->tryGetInteger(eos_candidates[0]));
    }

    // Make them the same if only one found
    if (bos_found && !eos_found) {
      eos_tok_ = bos_tok_;
    } else if (!bos_found && eos_found) {
      bos_tok_ = eos_tok_;
    }
  }

  // Mark initialized once everything is done
  initialized_ = true;

  return Error::Ok;
}
// -------------------------public method end-----------------------------------
// -------------------------private method start--------------------------------

Error HFTokenizer::_encode(
    const std::string& input,
    std::vector<uint64_t>& ret,
    uint64_t& last_piece_token_len) const {
  for (const auto& piece : _pretokenizer->pre_tokenize(input)) {
    const auto result = token_map_->tryGetInteger(piece);
    if (result) {
      last_piece_token_len = 1;
      ret.push_back(*result);
      continue;
    }
    auto tokens = TK_UNWRAP(byte_pair_encode_(piece, *token_map_));

    last_piece_token_len = tokens.size();
    ret.insert(ret.end(), tokens.begin(), tokens.end());
  }
  return Error::Ok;
}

void HFTokenizer::_decode(const std::string& input, std::string& ret) const {
  if (_decoder) {
    ret += _decoder->decode(input);
  } else {
    ret += input;
  }
}

} // namespace tokenizers
