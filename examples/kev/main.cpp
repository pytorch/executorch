/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pytorch/tokenizers/hf_tokenizer.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string_view>

#include "kev.h"

using executorch::extension::Module;
using kev::Question;

int main(int argc, char** argv) {
  const bool benchmark =
      argc > 3 && std::string_view(argv[argc - 1]) == "--benchmark";
  const int num_args = argc - benchmark;
  if (num_args < 3 || num_args > 4) {
    std::cerr
        << "Usage: kev_runner MODEL.pte TOKENIZER.json [TEXT] [--benchmark]\n";
    return 1;
  }
  Module module(argv[1], Module::LoadMode::Mmap);
  tokenizers::HFTokenizer tokenizer;
  if (tokenizer.load(argv[2]) != tokenizers::Error::Ok) {
    std::cerr << "Cannot load tokenizer.json\n";
    return 1;
  }
  const std::string state = num_args == 4
      ? argv[3]
      : "I was charged twice for invoice 4411. Please refund the duplicate charge.";
  const std::vector<std::vector<Question>> batches{
      {{"department",
        "Which team should handle this ticket?",
        {{"billing", "Payments, invoices, and refunds"},
         {"technical", "Bugs, outages, and integrations"},
         {"sales", "Pricing, upgrades, and new accounts"}}},
       {"refund_requested",
        "Does the customer explicitly ask for a refund?",
        {{"no", std::nullopt}, {"yes", std::nullopt}}}},
      {{"duplicate_charge",
        "Was the customer charged more than once?",
        {{"no", std::nullopt}, {"yes", std::nullopt}}}}};
  using Clock = std::chrono::steady_clock;
  using Milliseconds = std::chrono::duration<double, std::milli>;
  constexpr int warmups = 2;
  constexpr int runs = 5;
  std::array<double, runs> prefill_times{}, evaluation_times{}, total_times{};
  for (int i = 0; i < (benchmark ? warmups + runs : 1); ++i) {
    const auto start = Clock::now();
    auto prefix = kev::prefill(module, tokenizer, state);
    if (!prefix.ok()) {
      std::cerr << executorch::runtime::to_string(prefix.error()) << '\n';
      return 1;
    }
    const auto prefilled = Clock::now();
    for (const auto& questions : batches) {
      auto answers = kev::evaluate(*prefix, questions);
      if (!answers.ok()) {
        std::cerr << executorch::runtime::to_string(answers.error()) << '\n';
        return 1;
      }
      if (!benchmark) {
        for (const auto& answer : *answers) {
          std::cout << answer.question_id << ": "
                    << answer.labels[answer.scores.selected_index] << '\n';
          for (size_t j = 0; j < answer.labels.size(); ++j) {
            std::cout << "  " << answer.labels[j] << ": "
                      << answer.scores.probabilities[j] << '\n';
          }
        }
      }
    }
    const auto end = Clock::now();
    if (benchmark && i >= warmups) {
      prefill_times[i - warmups] = Milliseconds(prefilled - start).count();
      evaluation_times[i - warmups] = Milliseconds(end - prefilled).count();
      total_times[i - warmups] = Milliseconds(end - start).count();
    }
  }
  if (benchmark) {
    std::cout << "Median latency (ms; " << warmups << " warmups, " << runs
              << " runs; 3 questions in 2 calls)\n"
              << std::fixed << std::setprecision(2);
    const auto print_median = [](const char* label, auto times) {
      std::sort(times.begin(), times.end());
      std::cout << label << ": " << times[times.size() / 2] << '\n';
    };
    print_median("prefill", prefill_times);
    print_median("cached evaluation", evaluation_times);
    print_median("prefill + evaluation", total_times);
  }
  return 0;
}
