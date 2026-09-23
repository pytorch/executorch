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

#include "kev.h"

using executorch::extension::Module;
using kev::Choice;
using kev::Noul;
using kev::Questions;

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "Usage: kev_benchmark MODEL.pte TOKENIZER.json [TEXT]\n";
    return 1;
  }
  Module module(argv[1], Module::LoadMode::Mmap);
  tokenizers::HFTokenizer tokenizer;
  if (tokenizer.load(argv[2]) != tokenizers::Error::Ok) {
    std::cerr << "Cannot load tokenizer.json\n";
    return 1;
  }
  kev::Kev model(module, tokenizer);
  const std::string state = argc == 4
      ? argv[3]
      : "I was charged twice for invoice 4411. Please refund the duplicate charge.";
  const std::vector<Questions> batches{
      {{"department",
        Choice{
            "Which team should handle this ticket?",
            {{"billing", "Payments, invoices, and refunds"},
             {"technical", "Bugs, outages, and integrations"},
             {"sales", "Pricing, upgrades, and new accounts"}}}},
       {"refund_requested",
        Noul{"Does the customer explicitly ask for a refund?", {}}}},
      {{"duplicate_charge",
        Noul{"Was the customer charged more than once?", {}}}}};
  using Clock = std::chrono::steady_clock;
  using Milliseconds = std::chrono::duration<double, std::milli>;
  constexpr int warmups = 2;
  constexpr int runs = 5;
  std::array<double, runs> prefill_times{}, evaluation_times{}, total_times{};
  for (int i = 0; i < warmups + runs; ++i) {
    const auto start = Clock::now();
    auto prefix = model.prefill(state);
    if (!prefix.ok()) {
      std::cerr << executorch::runtime::to_string(prefix.error()) << '\n';
      return 1;
    }
    const auto prefilled = Clock::now();
    for (const auto& questions : batches) {
      auto answers = model.evaluate(*prefix, questions);
      if (!answers.ok()) {
        std::cerr << executorch::runtime::to_string(answers.error()) << '\n';
        return 1;
      }
    }
    const auto end = Clock::now();
    if (i >= warmups) {
      prefill_times[i - warmups] = Milliseconds(prefilled - start).count();
      evaluation_times[i - warmups] = Milliseconds(end - prefilled).count();
      total_times[i - warmups] = Milliseconds(end - start).count();
    }
  }
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
  return 0;
}
