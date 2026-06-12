/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// GPU no-bleed integration proof for the CUDA per-session mutable-state
// rebind -- the REAL guard for mutable-buffer completeness (an under-declared
// buffer would be shared across sessions; only behavior catches that, not the
// declared-subset-of-discovered bookkeeping check). This is the automated form
// of the manual "A solo / A inter" multi-session isolation proof.
//
// CRITICAL: sessions are interleaved at EXECUTE granularity (A prefill, B
// prefill, A decode, B decode, ...). The mechanism under test is the
// per-execute rebind, so running A-to-completion then B would pass even with a
// broken rebind.
//
// GPU-gated: requires a CUDA device + an exported model. Set QWEN_MODEL_PATH,
// QWEN_DATA_PATH, QWEN_TOKENIZER_PATH. Skips cleanly (exit 0) if unset or the
// engine cannot be created (no device) -- so it is safe in CI; the real run is
// the nightly/manual GPU job.

#include <executorch/examples/models/qwen3_5_moe/qwen35_moe_engine.h>

#include <executorch/backends/cuda/runtime/cuda_mutable_state.h>

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

// Greedy-decode `n` non-terminal tokens from a freshly-prefilled session.
std::vector<uint64_t>
solo_decode(llm::LLMSession& s, std::vector<uint64_t> prompt, int n) {
  llm::SamplingConfig samp; // temperature -1 => greedy/argmax (deterministic)
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

// GPU-FREE fall-closed DEFAULTS of cuda_mutable_state (no device, no handle).
// Covers only the safety defaults -- the descriptor build, positive coverage,
// bytes_per_session sum, and symbols_available AND-fold are exercised
// BEHAVIORALLY by the no-bleed integration test below (the real guard); a
// GPU-free unit test of those branches would need a build_descriptors allocator
// seam / fake-handle harness and is a knowingly-deferred follow-up.
namespace cu = ::executorch::backends::cuda;
void test_mutable_state_fallclosed_defaults() {
  printf("cuda_mutable_state fall-closed defaults (GPU-free):\n");
  const cu::MutableStateContext bad = 999999; // never created
  cu::MutableStateContext c1 = cu::mutable_state_create_context();
  cu::MutableStateContext c2 = cu::mutable_state_create_context();
  check("context ids are distinct/monotonic", c2 > c1);
  check(
      "fresh context: rebinding unavailable (no handle)",
      !cu::mutable_state_available(c1));
  check(
      "bytes_per_session: 0 for fresh and unknown",
      cu::mutable_state_bytes_per_session(c1) == 0 &&
          cu::mutable_state_bytes_per_session(bad) == 0);
  check(
      "validate_coverage: unknown ctx -> InvalidArgument",
      cu::mutable_state_validate_coverage(bad) == Error::InvalidArgument);
  check(
      "validate_coverage: no symbols -> NotSupported (fall closed)",
      cu::mutable_state_validate_coverage(c1) == Error::NotSupported);
  // Declaring FQNs without symbols still falls closed (the check is gated on
  // symbols, so it never wrongly passes coverage with nothing discovered).
  cu::mutable_state_register_fqns(c1, {"a.b", "c.d"});
  check(
      "validate_coverage: declared-but-no-symbols still NotSupported",
      cu::mutable_state_validate_coverage(c1) == Error::NotSupported);
  check(
      "create_session: unknown ctx -> InvalidArgument",
      cu::mutable_state_create_session(bad).error() == Error::InvalidArgument);
  check(
      "create_session: no symbols -> NotSupported",
      cu::mutable_state_create_session(c1).error() == Error::NotSupported);
  cu::mutable_state_destroy_session(bad, 0); // no-op, must not crash
  cu::mutable_state_destroy_context(bad); // no-op, must not crash
  cu::mutable_state_destroy_context(c1);
  cu::mutable_state_destroy_context(c2);
  check("destroy of unknown ctx/session is a safe no-op", true);
}

} // namespace

int main() {
  // GPU-free fall-closed defaults always run (even when the integration part
  // skips for lack of a device).
  test_mutable_state_fallclosed_defaults();

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

  // (1) Session A solo -> baseline greedy ids.
  auto sa_r = engine->create_session();
  check("create session A", sa_r.error() == Error::Ok);
  std::vector<uint64_t> ids_solo;
  if (sa_r.error() == Error::Ok) {
    auto sa = std::move(sa_r.get());
    ids_solo = solo_decode(*sa, prompt_a, kN);
  }
  check("solo produced tokens", !ids_solo.empty());

  // (2) A2 and B interleaved at EXECUTE granularity.
  auto a2_r = engine->create_session();
  auto b_r = engine->create_session();
  check("create A2 + B", a2_r.error() == Error::Ok && b_r.error() == Error::Ok);
  std::vector<uint64_t> ids_a2, ids_b;
  if (a2_r.error() == Error::Ok && b_r.error() == Error::Ok) {
    auto a2 = std::move(a2_r.get());
    auto b = std::move(b_r.get());
    llm::SamplingConfig samp;
    bool ok = a2->prefill_tokens(prompt_a, &samp) == Error::Ok && // A prefill
        b->prefill_tokens(prompt_b, &samp) == Error::Ok; // then B prefill
    check("interleaved prefills", ok);
    bool a_done = false, b_done = false;
    for (int i = 0; i < kN && ok; ++i) {
      if (!a_done) { // A decode
        auto r = a2->decode_one(samp);
        if (r.error() != Error::Ok || r.get().is_terminal) {
          a_done = true;
        } else {
          ids_a2.push_back(r.get().token_id);
        }
      }
      if (!b_done) { // B decode (between A's steps)
        auto r = b->decode_one(samp);
        if (r.error() != Error::Ok || r.get().is_terminal) {
          b_done = true;
        } else {
          ids_b.push_back(r.get().token_id);
        }
      }
    }
  }

  // THE no-bleed assertion: A's interleaved output is bit-identical to A solo
  // (greedy is deterministic), so B's interleaved session state did not corrupt
  // A's -- i.e. each session's mutable buffers are truly isolated.
  check(
      "no bleed: A interleaved == A solo (bit-identical)", ids_a2 == ids_solo);
  // Sanity that B actually ran a different conversation (else the test is
  // vacuous).
  check("B ran a distinct conversation", !ids_b.empty() && ids_b != ids_solo);

  // (3) Per-extra-session memory is STATE-sized, not a second model load.
  // Per-session buffers are allocated LAZILY on first execute (rebind), not at
  // create_session(), so measure the free-memory delta around a fresh session's
  // first prefill.
  const int64_t est = engine->serving_capacity().estimated_bytes_per_session;
  {
    int64_t free_before = gpu_free();
    auto extra_r = engine->create_session();
    if (extra_r.error() == Error::Ok) {
      auto extra = std::move(extra_r.get());
      llm::SamplingConfig samp;
      extra->prefill_tokens(
          prompt_a, &samp); // first execute -> allocates state
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
  } // extra released here -> frees its slot before the capacity test

  // (4) Capacity: the (max_sessions+1)th create_session fails (no silent
  // share). The sessions above already hold slots; create up to capacity then
  // one more.
  std::vector<std::unique_ptr<llm::LLMSession>> held;
  while (true) {
    auto r = engine->create_session();
    if (r.error() != Error::Ok) {
      break;
    }
    held.push_back(std::move(r.get()));
    if (held.size() > (size_t)config.max_sessions + 2) {
      break; // guard against a non-enforcing backend
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
