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

Result<std::vector<double>> probabilities(const float* logits, size_t count) {
  if (count == 0 || !std::all_of(logits, logits + count, [](float x) {
        return std::isfinite(x);
      })) {
    return Error::InvalidExternalData;
  }
  const double best = *std::max_element(logits, logits + count);
  std::vector<double> probabilities;
  probabilities.reserve(count);
  double total = 0;
  for (size_t i = 0; i < count; ++i) {
    probabilities.push_back(std::exp(static_cast<double>(logits[i]) - best));
    total += probabilities.back();
  }
  std::transform(
      probabilities.begin(),
      probabilities.end(),
      probabilities.begin(),
      [total](double probability) { return probability / total; });
  return probabilities;
}

Result<Answer>
make_answer(const Question& question, const float* logits, size_t count) {
  ET_ASSIGN_OR_RETURN(p, probabilities(logits, count));
  const auto selected =
      static_cast<size_t>(std::max_element(p.begin(), p.end()) - p.begin());
  if (const auto* choice = std::get_if<Choice>(&question)) {
    ChoiceAnswer answer{choice->criteria[selected].first, {}, 1.0};
    for (size_t j = 0; j < p.size(); ++j) {
      answer.probabilities.emplace(choice->criteria[j].first, p[j]);
    }
    if (p.size() > 1) {
      const double uniform = 1.0 / p.size();
      answer.confidence = (p[selected] - uniform) / (1.0 - uniform);
    }
    return Answer{std::move(answer)};
  } else if (std::holds_alternative<Noul>(question)) {
    return Answer{NoulAnswer{p[1]}};
  } else {
    const auto& score = std::get<Score>(question);
    ScoreAnswer answer{0.0, score.criteria, std::move(p), 1.0};
    double distance = 0.0;
    for (size_t j = 0; j < count; ++j) {
      answer.score += j * answer.probabilities[j];
      distance +=
          answer.probabilities[j] * std::abs(static_cast<double>(j) - selected);
    }
    if (count > 1) {
      answer.confidence = 1.0 - distance / (count - 1);
    }
    return Answer{std::move(answer)};
  }
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

Kev::Kev(Module& module, const tokenizers::Tokenizer& tokenizer)
    : module_(module), tokenizer_(tokenizer) {}

Result<Answers> Kev::system_one(
    const std::string& state,
    const Questions& questions) {
  ET_ASSIGN_OR_RETURN(prefix, prefill(state));
  return evaluate(prefix, questions);
}

Result<Prefix> Kev::prefill(const std::string& state) {
  ET_ASSIGN_OR_RETURN(version, metadata(module_, "get_kev_version"));
  ET_CHECK_OR_RETURN_ERROR(
      version == 1, InvalidProgram, "Expected a Kev export");
  ET_ASSIGN_OR_RETURN(max_prefix, metadata(module_, "get_max_prefix"));
  ET_ASSIGN_OR_RETURN(max_context, metadata(module_, "get_max_context"));
  ET_ASSIGN_OR_RETURN(max_questions, metadata(module_, "get_max_questions"));
  ET_ASSIGN_OR_RETURN(max_options, metadata(module_, "get_max_options"));
  ET_ASSIGN_OR_RETURN(pad_id, metadata(module_, "get_pad_id"));
  ET_CHECK_OR_RETURN_ERROR(
      max_prefix > 0 && max_context > max_prefix && max_questions > 0 &&
          max_options > 0 && max_options <= 255,
      InvalidProgram,
      "Invalid Kev limits");
  Prefix prefix;
  prefix.owner_ = this;
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
    ET_ASSIGN_OR_RETURN(id, metadata(module_, name.c_str()));
    const auto encoded = tokenizer_.encode(special_tokens[i]);
    ET_CHECK_OR_RETURN_ERROR(
        encoded.ok() && encoded->size() == 1 && encoded->front() == id,
        InvalidExternalData,
        "Tokenizer does not match Kev delimiter %s",
        special_tokens[i]);
    prefix.special_[i] = id;
  }
  ET_ASSIGN_OR_RETURN(tokens, user_tokens(tokenizer_, state));
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
  auto outputs = module_.execute("prefill", input);
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

Result<Answers> Kev::evaluate(
    const Prefix& prefix,
    const Questions& questions) {
  ET_CHECK_OR_RETURN_ERROR(
      prefix.owner_ == this,
      InvalidArgument,
      "Prefix belongs to a different Kev instance");
  ET_CHECK_OR_RETURN_ERROR(
      prefix.state_[0] && prefix.state_[1] && prefix.state_[2],
      InvalidArgument,
      "Invalid or moved prefix");
  ET_CHECK_OR_RETURN_ERROR(
      !questions.empty(), InvalidArgument, "Provide at least one question");
  std::vector<Row> rows;
  rows.reserve(questions.size());
  std::unordered_set<std::string> ids;
  for (const auto& [id, question] : questions) {
    ET_CHECK_OR_RETURN_ERROR(
        ids.insert(id).second, InvalidArgument, "Question IDs must be unique");
    std::vector<std::string> options;
    if (const auto* choice = std::get_if<Choice>(&question)) {
      std::unordered_set<std::string> names;
      for (const auto& [name, description] : choice->criteria) {
        ET_CHECK_OR_RETURN_ERROR(
            names.insert(name).second,
            InvalidArgument,
            "Choice criteria names must be unique");
        options.push_back(name);
        if (description && !description->empty()) {
          options.back() += ": " + *description;
        }
      }
    } else if (const auto* noul = std::get_if<Noul>(&question)) {
      for (const auto outcome : {NoulOutcome::False, NoulOutcome::True}) {
        options.emplace_back(outcome == NoulOutcome::True ? "yes" : "no");
        const auto description = noul->criteria.find(outcome);
        if (description != noul->criteria.end() &&
            !description->second.empty()) {
          options.back() += ": " + description->second;
        }
      }
    } else {
      options = std::get<Score>(question).criteria;
    }
    ET_CHECK_OR_RETURN_ERROR(
        !options.empty() && options.size() <= prefix.max_options_,
        InvalidArgument,
        "Question requires 1-%zu criteria",
        prefix.max_options_);
    const auto& text = std::visit(
        [](const auto& q) -> const std::string& { return q.instructions; },
        question);
    ET_ASSIGN_OR_RETURN(instructions, user_tokens(tokenizer_, text));
    Row row{{prefix.special_[1]}, {}};
    row.tokens.insert(
        row.tokens.end(), instructions.begin(), instructions.end());
    for (const auto& option : options) {
      ET_ASSIGN_OR_RETURN(tokens, user_tokens(tokenizer_, option));
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
        id.c_str(),
        prefix.max_context_);
    rows.push_back(std::move(row));
  }
  Answers answers;
  answers.reserve(questions.size());
  for (size_t start = 0; start < rows.size();) {
    const auto count = std::min(prefix.max_questions_, rows.size() - start);
    size_t max_length = 0, max_options = 0;
    for (size_t i = start; i < start + count; ++i) {
      max_length = std::max(max_length, rows[i].tokens.size());
      max_options = std::max(max_options, rows[i].options.size());
    }
    const auto batch = static_cast<int32_t>(count);
    std::vector<int64_t> tokens(count * max_length, prefix.pad_id_);
    std::vector<int64_t> options(count * max_options, 0);
    std::vector<int64_t> decide(count);
    for (size_t i = 0; i < count; ++i) {
      const auto& row = rows[start + i];
      std::copy(
          row.tokens.begin(),
          row.tokens.end(),
          tokens.begin() + i * max_length);
      std::copy(
          row.options.begin(),
          row.options.end(),
          options.begin() + i * max_options);
      decide[i] = row.tokens.size() - 1;
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
    auto outputs = module_.execute(
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
    for (size_t i = 0; i < count; ++i) {
      const auto& [id, question] = questions[start + i];
      const auto* begin =
          logits.const_data_ptr<float>() + i * logits.strides()[0];
      ET_ASSIGN_OR_RETURN(
          answer, make_answer(question, begin, rows[start + i].options.size()));
      answers.emplace_back(id, std::move(answer));
    }
    start += count;
  }
  return answers;
}

} // namespace kev
