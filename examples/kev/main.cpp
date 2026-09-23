/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pytorch/tokenizers/hf_tokenizer.h>
#include <iostream>

#include "kev.h"

using executorch::extension::Module;
using kev::Choice;
using kev::Noul;
using kev::Questions;

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "Usage: kev_runner MODEL.pte TOKENIZER.json [TEXT]\n";
    return 1;
  }
  Module module(argv[1], Module::LoadMode::Mmap);
  tokenizers::HFTokenizer tokenizer;
  if (tokenizer.load(argv[2]) != tokenizers::Error::Ok) {
    std::cerr << "Cannot load tokenizer.json\n";
    return 1;
  }
  const std::string state = argc == 4
      ? argv[3]
      : "I was charged twice for invoice 4411. Please refund the duplicate charge.";
  const Questions questions{
      {"department",
       Choice{
           "Which team should handle this ticket?",
           {{"billing", "Payments, invoices, and refunds"},
            {"technical", "Bugs, outages, and integrations"},
            {"sales", "Pricing, upgrades, and new accounts"}}}},
      {"refund_requested",
       Noul{"Does the customer explicitly ask for a refund?", {}}},
      {"duplicate_charge",
       Noul{"Was the customer charged more than once?", {}}}};
  kev::Kev model(module, tokenizer);
  auto answers = model.system_one(state, questions);
  if (!answers.ok()) {
    std::cerr << executorch::runtime::to_string(answers.error()) << '\n';
    return 1;
  }
  for (const auto& [id, answer] : *answers) {
    std::cout << id << ": ";
    if (const auto* choice = std::get_if<kev::ChoiceAnswer>(&answer)) {
      std::cout << choice->choice << " (confidence " << choice->confidence
                << ")\n";
      for (const auto& [name, probability] : choice->probabilities) {
        std::cout << "  " << name << ": " << probability << '\n';
      }
    } else if (const auto* noul = std::get_if<kev::NoulAnswer>(&answer)) {
      std::cout << noul->noul << '\n';
    } else {
      const auto& score = std::get<kev::ScoreAnswer>(answer);
      std::cout << score.score << " (confidence " << score.confidence << ")\n";
      for (size_t level = 0; level < score.probabilities.size(); ++level) {
        std::cout << "  " << level << " (" << score.legend.at(level)
                  << "): " << score.probabilities[level] << '\n';
      }
    }
  }
  return 0;
}
