/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// CUDA integration proof for one loaded Gemma4 31B model with isolated
// sessions. Set GEMMA_MODEL_PATH, GEMMA_DATA_PATH, and GEMMA_TOKENIZER_PATH to
// run it.

#include <executorch/examples/models/gemma4_31b/gemma4_31b_engine.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace llm = ::executorch::extension::llm;
using ::executorch::runtime::Error;

namespace {

int g_failures = 0;

void check(const std::string& name, bool ok) {
  printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name.c_str());
  if (!ok) {
    ++g_failures;
  }
}

const char* env(const char* k) {
  const char* v = std::getenv(k);
  return (v && *v) ? v : nullptr;
}

std::vector<uint64_t> encode_prompt(
    llm::Gemma4_31BEngine& engine,
    const std::string& prompt) {
  const std::string rendered = "<|turn>user\n" + prompt +
      "<turn|>\n<|turn>model\n"
      "<|channel>thought\n<channel|>";
  auto encoded = engine.tokenizer()->encode(rendered, /*bos=*/0, /*eos=*/0);
  if (!encoded.ok()) {
    return {};
  }
  std::vector<uint64_t> ids;
  ids.reserve(encoded->size() + 1);
  ids.push_back(2);
  ids.insert(ids.end(), encoded->begin(), encoded->end());
  return ids;
}

std::vector<uint64_t> encode_prompt_at_least(
    llm::Gemma4_31BEngine& engine,
    const std::string& seed,
    const std::string& filler,
    size_t min_tokens) {
  std::string prompt = seed;
  std::vector<uint64_t> ids = encode_prompt(engine, prompt);
  while (!ids.empty() && ids.size() < min_tokens) {
    prompt.append(" ");
    prompt.append(filler);
    ids = encode_prompt(engine, prompt);
  }
  return ids;
}

std::vector<uint64_t> decode_tokens(
    llm::LLMSession& session,
    const std::vector<uint64_t>& prompt,
    int n) {
  llm::SamplingConfig sampling;
  std::vector<uint64_t> out;
  if (session.prefill_tokens(prompt, &sampling) != Error::Ok) {
    return out;
  }
  for (int i = 0; i < n; ++i) {
    auto step = session.decode_one(sampling);
    if (step.error() != Error::Ok || step.get().is_terminal) {
      break;
    }
    out.push_back(step.get().token_id);
  }
  return out;
}

void run_no_bleed_case(
    llm::Gemma4_31BEngine& engine,
    const std::string& label,
    const std::vector<uint64_t>& prompt_a,
    const std::vector<uint64_t>& prompt_b,
    int decode_steps) {
  printf(
      "\n%s (A=%zu tokens, B=%zu tokens)\n",
      label.c_str(),
      prompt_a.size(),
      prompt_b.size());
  check(label + ": prompts encoded", !prompt_a.empty() && !prompt_b.empty());

  std::vector<uint64_t> solo;
  auto solo_session = engine.create_session();
  check(label + ": create solo session", solo_session.error() == Error::Ok);
  if (solo_session.error() == Error::Ok) {
    solo = decode_tokens(*solo_session.get(), prompt_a, decode_steps);
  }
  check(label + ": solo produced tokens", !solo.empty());

  std::vector<uint64_t> interleaved_a;
  std::vector<uint64_t> interleaved_b;
  auto a_result = engine.create_session();
  auto b_result = engine.create_session();
  check(
      label + ": create interleaved sessions",
      a_result.error() == Error::Ok && b_result.error() == Error::Ok);
  if (a_result.error() == Error::Ok && b_result.error() == Error::Ok) {
    auto a = std::move(a_result.get());
    auto b = std::move(b_result.get());
    llm::SamplingConfig sampling;
    bool ok = a->prefill_tokens(prompt_a, &sampling) == Error::Ok &&
        b->prefill_tokens(prompt_b, &sampling) == Error::Ok;
    check(label + ": interleaved prefills", ok);
    bool a_done = false;
    bool b_done = false;
    for (int i = 0; i < decode_steps && ok; ++i) {
      if (!a_done) {
        auto step = a->decode_one(sampling);
        if (step.error() != Error::Ok || step.get().is_terminal) {
          a_done = true;
        } else {
          interleaved_a.push_back(step.get().token_id);
        }
      }
      if (!b_done) {
        auto step = b->decode_one(sampling);
        if (step.error() != Error::Ok || step.get().is_terminal) {
          b_done = true;
        } else {
          interleaved_b.push_back(step.get().token_id);
        }
      }
    }
  }

  check(
      label + ": A interleaved == A solo (bit-identical)",
      interleaved_a == solo);
  check(
      label + ": B ran a distinct conversation",
      !interleaved_b.empty() && interleaved_b != solo);
}

int64_t gpu_free() {
  size_t free = 0;
  size_t total = 0;
  return cudaMemGetInfo(&free, &total) == cudaSuccess
      ? static_cast<int64_t>(free)
      : -1;
}

} // namespace

int main() {
  const char* model = env("GEMMA_MODEL_PATH");
  const char* tokenizer = env("GEMMA_TOKENIZER_PATH");
  if (!model || !tokenizer) {
    printf(
        "SKIP: integration proof needs GEMMA_MODEL_PATH / "
        "GEMMA_TOKENIZER_PATH (+ GEMMA_DATA_PATH) on a CUDA box.\n");
    return 0;
  }

  llm::Gemma4_31BConfig config;
  config.model_path = model;
  config.data_path = env("GEMMA_DATA_PATH") ? env("GEMMA_DATA_PATH") : "";
  config.tokenizer_path = tokenizer;
  config.max_sessions = 4;

  auto engine_result = llm::Gemma4_31BEngine::create(config);
  if (engine_result.error() != Error::Ok) {
    printf("SKIP: engine create failed (no CUDA device / bad paths).\n");
    return 0;
  }
  auto engine = std::move(engine_result.get());

  printf("Gemma4 31B no-bleed integration proof:\n");
  auto prompt_a = encode_prompt(*engine, "List three colors.");
  auto prompt_b = encode_prompt(*engine, "Name two countries in Europe.");
  run_no_bleed_case(*engine, "short-context", prompt_a, prompt_b, 24);

  auto long_prompt_a = encode_prompt_at_least(
      *engine,
      "After reading these notes, answer with one concise sentence.",
      "alpha beta gamma delta epsilon zeta eta theta iota kappa",
      1152);
  auto long_prompt_b = encode_prompt_at_least(
      *engine,
      "After reading this inventory, answer with one concise sentence.",
      "red orange yellow green blue indigo violet black white silver",
      1152);
  run_no_bleed_case(
      *engine,
      "long-context-crosses-sliding-window",
      long_prompt_a,
      long_prompt_b,
      8);

  const int64_t est = engine->serving_capacity().estimated_bytes_per_session;
  int64_t free_before = gpu_free();
  {
    auto extra_result = engine->create_session();
    if (extra_result.error() == Error::Ok) {
      auto extra = std::move(extra_result.get());
      llm::SamplingConfig sampling;
      extra->prefill_tokens(prompt_a, &sampling);
      int64_t free_after = gpu_free();
      if (free_before > 0 && free_after > 0) {
        const int64_t delta = free_before - free_after;
        printf(
            "    extra-session GPU delta=%lld bytes (est/session=%lld)\n",
            static_cast<long long>(delta),
            static_cast<long long>(est));
        check(
            "extra session is state-sized, not another model load",
            delta > 0 && delta < (16LL << 30));
        if (est > 0) {
          check(
              "memory delta within 2x of estimated_bytes_per_session",
              delta <= est * 2 + (512LL << 20));
        }
      }
    }
  }

  std::vector<std::unique_ptr<llm::LLMSession>> held;
  const int capacity = engine->serving_capacity()
                           .max_physical_sessions_without_weight_duplication;
  while (true) {
    auto session = engine->create_session();
    if (session.error() != Error::Ok) {
      break;
    }
    held.push_back(std::move(session.get()));
    if (held.size() > static_cast<size_t>(capacity) + 2) {
      break;
    }
  }
  check(
      "capacity enforced: create_session fails past serving_capacity",
      held.size() <= static_cast<size_t>(capacity));

  printf(
      "\n%s (%d failure(s))\n",
      g_failures ? "FAILURES" : "ALL PASS",
      g_failures);
  return g_failures ? 1 : 0;
}
