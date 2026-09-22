/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "kev.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <regex>
#include <unordered_set>
#include <utility>

#include <executorch/extension/tensor/tensor.h>
#include <pytorch/tokenizers/tokenizer.h>

namespace kev {

using executorch::aten::ScalarType;
using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::runtime::Error;
using executorch::runtime::Result;

namespace {

Result<OptionScores> score_options(std::vector<double> logits) {
  if (logits.empty() ||
      !std::all_of(logits.begin(), logits.end(), [](double x) {
        return std::isfinite(x);
      })) {
    return Error::InvalidArgument;
  }
  const auto best = std::max_element(logits.begin(), logits.end());
  const auto selected = static_cast<size_t>(best - logits.begin());
  std::vector<double> probabilities;
  probabilities.reserve(logits.size());
  double total = 0;
  for (const auto logit : logits) {
    probabilities.push_back(std::exp(logit - *best));
    total += probabilities.back();
  }
  std::transform(
      probabilities.begin(),
      probabilities.end(),
      probabilities.begin(),
      [total](double probability) { return probability / total; });
  return OptionScores{std::move(logits), std::move(probabilities), selected};
}

Result<int64_t> metadata(Module& module, const char* name) {
  auto result = module.execute(name);
  if (!result.ok()) {
    return result.error();
  }
  ET_CHECK_OR_RETURN_ERROR(
      result->size() == 1 && result->front().isInt(),
      InvalidProgram,
      "Expected integer metadata: %s",
      name);
  const auto value = result->front().toInt();
  ET_CHECK_OR_RETURN_ERROR(
      value >= 0 && value <= std::numeric_limits<int32_t>::max(),
      InvalidProgram,
      "Invalid metadata: %s",
      name);
  return value;
}

Result<std::vector<int64_t>> user_tokens(
    const tokenizers::Tokenizer& tokenizer,
    const std::string& text) {
  static const std::regex control_token(R"(<\|([A-Za-z0-9_]+)\|>)");
  try {
    const auto escaped = std::regex_replace(text, control_token, "<¦$1¦>");
    auto encoded = tokenizer.encode(escaped);
    if (!encoded.ok()) {
      return Error::InvalidExternalData;
    }
    return std::vector<int64_t>(encoded->begin(), encoded->end());
  } catch (const std::exception&) {
    return Error::InvalidArgument;
  }
}

struct Row {
  std::vector<int64_t> tokens;
  std::vector<int64_t> options;
};

} // namespace

Result<Prefix> prefill(
    Module& module,
    const tokenizers::Tokenizer& tokenizer,
    const std::string& state) {
  ET_ASSIGN_OR_RETURN(version, metadata(module, "get_kev_version"));
  ET_CHECK_OR_RETURN_ERROR(
      version == 1, InvalidProgram, "Expected a Kev export");
  ET_ASSIGN_OR_RETURN(max_prefix, metadata(module, "get_max_prefix"));
  ET_ASSIGN_OR_RETURN(max_context, metadata(module, "get_max_context"));
  ET_ASSIGN_OR_RETURN(max_questions, metadata(module, "get_max_questions"));
  ET_ASSIGN_OR_RETURN(max_options, metadata(module, "get_max_options"));
  ET_ASSIGN_OR_RETURN(pad_id, metadata(module, "get_pad_id"));
  ET_CHECK_OR_RETURN_ERROR(
      max_prefix > 0 && max_context > max_prefix && max_questions > 0 &&
          max_options > 0 && max_options <= 255,
      InvalidProgram,
      "Invalid Kev limits");
  Prefix prefix;
  prefix.module_ = &module;
  prefix.tokenizer_ = &tokenizer;
  prefix.max_context_ = max_context;
  prefix.max_questions_ = max_questions;
  prefix.max_options_ = max_options;
  prefix.pad_id_ = pad_id;
  constexpr std::array special_tokens{
      "<|fim_prefix|>",
      "<|fim_middle|>",
      "<|box_start|>",
      "<|box_end|>",
      "<|fim_suffix|>"};
  for (size_t i = 0; i < special_tokens.size(); ++i) {
    const auto name = "get_special_" + std::to_string(i);
    ET_ASSIGN_OR_RETURN(id, metadata(module, name.c_str()));
    const auto encoded = tokenizer.encode(special_tokens[i]);
    ET_CHECK_OR_RETURN_ERROR(
        encoded.ok() && encoded->size() == 1 && encoded->front() == id,
        InvalidExternalData,
        "Tokenizer does not match Kev delimiter %s",
        special_tokens[i]);
    prefix.special_[i] = id;
  }
  ET_ASSIGN_OR_RETURN(tokens, user_tokens(tokenizer, state));
  tokens.insert(tokens.begin(), prefix.special_[0]);
  ET_CHECK_OR_RETURN_ERROR(
      tokens.size() <= static_cast<size_t>(max_prefix),
      InvalidArgument,
      "State has %zu tokens; limit is %lld",
      tokens.size(),
      static_cast<long long>(max_prefix));
  prefix.length_ = tokens.size();
  auto input = from_blob(
      tokens.data(),
      {1, static_cast<int32_t>(tokens.size())},
      ScalarType::Long);
  auto outputs = module.execute("prefill", input);
  if (!outputs.ok()) {
    return outputs.error();
  }
  ET_CHECK_OR_RETURN_ERROR(
      outputs->size() == 3 && (*outputs)[0].isTensor() &&
          (*outputs)[1].isTensor() && (*outputs)[2].isTensor(),
      InvalidProgram,
      "Expected convolution, DeltaNet, and attention state");
  const auto& conv = (*outputs)[0].toTensor();
  const auto& recurrent = (*outputs)[1].toTensor();
  const auto& kv = (*outputs)[2].toTensor();
  ET_CHECK_OR_RETURN_ERROR(
      conv.dim() == 4 && conv.size(1) == 1 && recurrent.dim() == 5 &&
          recurrent.size(0) == conv.size(0) && recurrent.size(1) == 1 &&
          recurrent.scalar_type() == ScalarType::Float && kv.dim() == 6 &&
          kv.size(1) == 2 && kv.size(2) == 1 && kv.size(4) == prefix.length_ &&
          kv.scalar_type() == conv.scalar_type() &&
          (conv.scalar_type() == ScalarType::Float ||
           conv.scalar_type() == ScalarType::BFloat16),
      InvalidProgram,
      "Invalid Kev prefix tensors");
  for (size_t i = 0; i < prefix.state_.size(); ++i) {
    prefix.state_[i] =
        executorch::extension::clone_tensor_ptr((*outputs)[i].toTensor());
  }
  return prefix;
}

Result<std::vector<Answer>> evaluate(
    const Prefix& prefix,
    const std::vector<Question>& questions) {
  ET_CHECK_OR_RETURN_ERROR(
      prefix.module_ && prefix.tokenizer_ && prefix.state_[0] &&
          prefix.state_[1] && prefix.state_[2],
      InvalidArgument,
      "Invalid or moved prefix");
  ET_CHECK_OR_RETURN_ERROR(
      !questions.empty() && questions.size() <= prefix.max_questions_,
      InvalidArgument,
      "Provide 1-%zu questions",
      prefix.max_questions_);
  std::vector<Row> rows;
  std::unordered_set<std::string> ids;
  size_t max_length = 0, max_options = 0;
  for (const auto& question : questions) {
    ET_CHECK_OR_RETURN_ERROR(
        !question.id.empty() && ids.insert(question.id).second &&
            question.instructions.find_first_not_of(" \t\n\r\f\v") !=
                std::string::npos &&
            !question.options.empty() &&
            question.options.size() <= prefix.max_options_,
        InvalidArgument,
        "Question requires a unique ID, instructions, and 1-%zu options",
        prefix.max_options_);
    ET_ASSIGN_OR_RETURN(
        instructions, user_tokens(*prefix.tokenizer_, question.instructions));
    Row row{{prefix.special_[1]}, {}};
    row.tokens.insert(
        row.tokens.end(), instructions.begin(), instructions.end());
    std::unordered_set<std::string> labels;
    for (const auto& option : question.options) {
      ET_CHECK_OR_RETURN_ERROR(
          !option.label.empty() && labels.insert(option.label).second,
          InvalidArgument,
          "Option labels must be nonempty and unique");
      std::string text = option.label;
      if (option.description && !option.description->empty()) {
        text += ": " + *option.description;
      }
      ET_ASSIGN_OR_RETURN(tokens, user_tokens(*prefix.tokenizer_, text));
      row.tokens.push_back(prefix.special_[2]);
      row.tokens.insert(row.tokens.end(), tokens.begin(), tokens.end());
      row.options.push_back(row.tokens.size());
      row.tokens.push_back(prefix.special_[3]);
    }
    row.tokens.push_back(prefix.special_[4]);
    ET_CHECK_OR_RETURN_ERROR(
        row.tokens.size() <= prefix.max_context_ - prefix.length_,
        InvalidArgument,
        "Question '%s' exceeds the %zu-token context limit",
        question.id.c_str(),
        prefix.max_context_);
    max_length = std::max(max_length, row.tokens.size());
    max_options = std::max(max_options, row.options.size());
    rows.push_back(std::move(row));
  }
  const auto batch = static_cast<int32_t>(rows.size());
  std::vector<int64_t> tokens(rows.size() * max_length, prefix.pad_id_);
  std::vector<int64_t> options(rows.size() * max_options, 0);
  std::vector<int64_t> decide(rows.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    std::copy(
        rows[i].tokens.begin(),
        rows[i].tokens.end(),
        tokens.begin() + i * max_length);
    std::copy(
        rows[i].options.begin(),
        rows[i].options.end(),
        options.begin() + i * max_options);
    decide[i] = rows[i].tokens.size() - 1;
  }
  auto input = from_blob(
      tokens.data(),
      {batch, static_cast<int32_t>(max_length)},
      ScalarType::Long);
  auto decide_input = from_blob(decide.data(), {batch}, ScalarType::Long);
  auto option_input = from_blob(
      options.data(),
      {batch, static_cast<int32_t>(max_options)},
      ScalarType::Long);
  auto outputs = prefix.module_->execute(
      "score",
      {input,
       decide_input,
       option_input,
       prefix.state_[0],
       prefix.state_[1],
       prefix.state_[2]});
  if (!outputs.ok()) {
    return outputs.error();
  }
  ET_CHECK_OR_RETURN_ERROR(
      outputs->size() == 1 && outputs->front().isTensor(),
      InvalidProgram,
      "Expected Kev pointer logits");
  const auto& logits = outputs->front().toTensor();
  ET_CHECK_OR_RETURN_ERROR(
      logits.scalar_type() == ScalarType::Float && logits.dim() == 2 &&
          logits.size(0) == batch && logits.size(1) == max_options &&
          logits.strides()[1] == 1,
      InvalidProgram,
      "Expected float logits with shape [questions, options]");
  std::vector<Answer> answers;
  for (size_t i = 0; i < questions.size(); ++i) {
    const auto* begin =
        logits.const_data_ptr<float>() + i * logits.strides()[0];
    auto scores = score_options({begin, begin + questions[i].options.size()});
    if (!scores.ok()) {
      return Error::InvalidExternalData;
    }
    Answer answer{questions[i].id, {}, std::move(*scores)};
    for (const auto& option : questions[i].options) {
      answer.labels.push_back(option.label);
    }
    answers.push_back(std::move(answer));
  }
  return answers;
}

} // namespace kev
