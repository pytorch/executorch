/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// CUDA integration proof for one loaded Qwen model with isolated sessions.
// Set QWEN_MODEL_PATH, QWEN_DATA_PATH, and QWEN_TOKENIZER_PATH to run it.

#include <executorch/examples/models/qwen3_5_moe/qwen35_moe_engine.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace llm = ::executorch::extension::llm;
using ::executorch::runtime::Error;

namespace {
int g_failures = 0;
void check(const char* name, bool ok) {
  printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) {
    ++g_failures;
  }
}

const char* env(const char* k) {
  const char* v = std::getenv(k);
  return (v && *v) ? v : nullptr;
}

std::vector<uint64_t> encode(llm::Qwen35MoEEngine& e, const std::string& s) {
  auto r = e.tokenizer()->encode(s);
  return r.ok() ? std::move(*r) : std::vector<uint64_t>{};
}

std::vector<uint64_t>
solo_decode(llm::LLMSession& s, std::vector<uint64_t> prompt, int n) {
  llm::SamplingConfig samp;
  std::vector<uint64_t> out;
  if (s.prefill_tokens(prompt, &samp) != Error::Ok) {
    return out;
  }
  for (int i = 0; i < n; ++i) {
    auto r = s.decode_one(samp);
    if (r.error() != Error::Ok || r.get().is_terminal) {
      break;
    }
    out.push_back(r.get().token_id);
  }
  return out;
}

int64_t gpu_free() {
  size_t free = 0, total = 0;
  return cudaMemGetInfo(&free, &total) == cudaSuccess
      ? static_cast<int64_t>(free)
      : -1;
}

} // namespace

int main() {
  const char* model = env("QWEN_MODEL_PATH");
  const char* tok = env("QWEN_TOKENIZER_PATH");
  if (!model || !tok) {
    printf(
        "SKIP: integration proof needs QWEN_MODEL_PATH / QWEN_TOKENIZER_PATH "
        "(+ QWEN_DATA_PATH) on a CUDA box.\n");
    return g_failures ? 1 : 0;
  }
  llm::Qwen35MoEConfig config;
  config.model_path = model;
  config.data_path = env("QWEN_DATA_PATH") ? env("QWEN_DATA_PATH") : "";
  config.tokenizer_path = tok;
  config.max_sessions = 4;

  auto engine_r = llm::Qwen35MoEEngine::create(config);
  if (engine_r.error() != Error::Ok) {
    printf("SKIP: engine create failed (no CUDA device / bad paths).\n");
    return 0;
  }
  auto engine = std::move(engine_r.get());
  printf("no-bleed integration proof:\n");

  const int kN = 24;
  auto prompt_a = encode(*engine, "List three colors:");
  auto prompt_b =
      encode(*engine, "Name two countries in Europe and explain why.");
  check("prompts encoded", !prompt_a.empty() && !prompt_b.empty());

  auto sa_r = engine->create_session();
  check("create session A", sa_r.error() == Error::Ok);
  std::vector<uint64_t> ids_solo;
  if (sa_r.error() == Error::Ok) {
    auto sa = std::move(sa_r.get());
    ids_solo = solo_decode(*sa, prompt_a, kN);
  }
  check("solo produced tokens", !ids_solo.empty());

  auto a2_r = engine->create_session();
  auto b_r = engine->create_session();
  check("create A2 + B", a2_r.error() == Error::Ok && b_r.error() == Error::Ok);
  std::vector<uint64_t> ids_a2, ids_b;
  if (a2_r.error() == Error::Ok && b_r.error() == Error::Ok) {
    auto a2 = std::move(a2_r.get());
    auto b = std::move(b_r.get());
    llm::SamplingConfig samp;
    bool ok = a2->prefill_tokens(prompt_a, &samp) == Error::Ok &&
        b->prefill_tokens(prompt_b, &samp) == Error::Ok;
    check("interleaved prefills", ok);
    bool a_done = false, b_done = false;
    for (int i = 0; i < kN && ok; ++i) {
      if (!a_done) {
        auto r = a2->decode_one(samp);
        if (r.error() != Error::Ok || r.get().is_terminal) {
          a_done = true;
        } else {
          ids_a2.push_back(r.get().token_id);
        }
      }
      if (!b_done) {
        auto r = b->decode_one(samp);
        if (r.error() != Error::Ok || r.get().is_terminal) {
          b_done = true;
        } else {
          ids_b.push_back(r.get().token_id);
        }
      }
    }
  }

  check(
      "no bleed: A interleaved == A solo (bit-identical)", ids_a2 == ids_solo);
  check("B ran a distinct conversation", !ids_b.empty() && ids_b != ids_solo);

  const int64_t est = engine->serving_capacity().estimated_bytes_per_session;
  {
    int64_t free_before = gpu_free();
    auto extra_r = engine->create_session();
    if (extra_r.error() == Error::Ok) {
      auto extra = std::move(extra_r.get());
      llm::SamplingConfig samp;
      extra->prefill_tokens(prompt_a, &samp);
      int64_t free_after = gpu_free();
      if (free_before > 0 && free_after > 0) {
        const int64_t delta = free_before - free_after;
        printf(
            "    extra-session GPU delta=%lld bytes (est/session=%lld)\n",
            (long long)delta,
            (long long)est);
        check(
            "extra session is state-sized (>0, < 4 GB, not an ~18 GB reload)",
            delta > 0 && delta < (4LL << 30));
        if (est > 0) {
          check(
              "memory delta within 2x of estimated_bytes_per_session",
              delta <= est * 2 + (256LL << 20));
        }
      }
    }
  }

  std::vector<std::unique_ptr<llm::LLMSession>> held;
  while (true) {
    auto r = engine->create_session();
    if (r.error() != Error::Ok) {
      break;
    }
    held.push_back(std::move(r.get()));
    if (held.size() > (size_t)config.max_sessions + 2) {
      break;
    }
  }
  check(
      "capacity enforced: create_session fails past max_sessions",
      held.size() <= (size_t)config.max_sessions);

  printf(
      "\n%s (%d failure(s))\n",
      g_failures ? "FAILURES" : "ALL PASS",
      g_failures);
  return g_failures ? 1 : 0;
}
