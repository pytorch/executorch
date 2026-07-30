/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Hermetic tests for worker_loop.h (worker_handle_request + WorkerSessions),
// the highest-risk serving logic. A scriptable fake LLMSession / Tokenizer /
// LLMEngine drives the real loop with NO model, tokenizer, or GPU. worker_emit
// writes to std::cout, so each test captures stdout and parses the JSON events.
// Self-contained assertions (no gtest) to match test_worker_prefill_plan.

#include <executorch/examples/llm_server/cpp/worker_loop.h>

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using executorch::extension::llm::DecodeResult;
using executorch::extension::llm::LLMEngine;
using executorch::extension::llm::LLMServingCapacity;
using executorch::extension::llm::LLMSession;
using executorch::extension::llm::SamplingConfig;
using executorch::extension::llm::worker_handle_request;
using executorch::extension::llm::WorkerSessions;
using executorch::extension::llm::WorkerSessionState;
using ETError = ::executorch::runtime::Error;
template <typename T>
using ETResult = ::executorch::runtime::Result<T>;

namespace {
int g_failures = 0;

void check(const char* name, bool ok) {
  printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) {
    ++g_failures;
  }
}

// ---- Fake LLMSession: scriptable decode stream + injectable failures --------
class FakeSession : public LLMSession {
 public:
  struct Step {
    uint64_t id;
    std::string piece;
    bool is_eos;
    bool is_terminal;
  };
  std::vector<Step> steps;
  size_t step_i = 0;
  int64_t pos = 0; // models the session's KV position

  int prefill_calls = 0;
  std::vector<size_t> prefill_sizes; // size of each prefill_tokens() call
  std::vector<std::vector<uint64_t>> prefill_batches;
  int fail_prefill_on = -1; // 0-based call index to fail (-1 = never)
  int decode_calls = 0;
  int fail_decode_on = -1;
  int reset_calls = 0;
  bool fail_reset = false;

  ETError prefill_tokens(
      const std::vector<uint64_t>& tokens,
      const SamplingConfig* /*initial_sampling*/ = nullptr) override {
    prefill_sizes.push_back(tokens.size());
    prefill_batches.push_back(tokens);
    if (prefill_calls++ == fail_prefill_on) {
      return ETError::Internal; // failed AFTER (notionally) mutating state
    }
    pos += static_cast<int64_t>(tokens.size());
    return ETError::Ok;
  }

  ETResult<DecodeResult> decode_one(const SamplingConfig& /*s*/) override {
    if (decode_calls++ == fail_decode_on) {
      return ETError::Internal;
    }
    if (step_i >= steps.size()) {
      return DecodeResult{0, "", true, true}; // default: EOS/terminal
    }
    const Step s = steps[step_i++];
    if (!s.is_terminal) {
      pos += 1; // a forwarded token advances the cache position
    }
    return DecodeResult{s.id, s.piece, s.is_eos, s.is_terminal};
  }

  int64_t position() const override {
    return pos;
  }
  ETError reset() override {
    ++reset_calls;
    if (fail_reset) {
      return ETError::Internal;
    }
    pos = 0;
    step_i = 0;
    return ETError::Ok;
  }
  void stop() override {}
};

// ---- Fake Tokenizer: only needed to satisfy the signature; tests use {ids}
// segments so encode() is not exercised on the hot paths. -------------------
class FakeTokenizer : public ::tokenizers::Tokenizer {
 public:
  ::tokenizers::Error load(const std::string&) override {
    initialized_ = true;
    return ::tokenizers::Error::Ok;
  }
  ::tokenizers::Result<std::vector<uint64_t>> encode(
      const std::string& input,
      int8_t /*bos*/ = 0,
      int8_t /*eos*/ = 0) const override {
    std::vector<uint64_t>
        out; // 1 id per byte (deterministic; unused by ids tests)
    out.reserve(input.size());
    std::transform(
        input.begin(),
        input.end(),
        std::back_inserter(out),
        [](unsigned char c) { return static_cast<uint64_t>(c); });
    return out;
  }
  ::tokenizers::Result<std::string> decode(
      uint64_t /*prev*/,
      uint64_t /*token*/,
      bool /*skip_special_tokens*/ = false) const override {
    return std::string("");
  }
  ::tokenizers::Result<std::string> id_to_piece(uint64_t /*t*/) const override {
    return std::string("");
  }
  ::tokenizers::Result<uint64_t> piece_to_id(
      const std::string& /*t*/) const override {
    return static_cast<uint64_t>(0);
  }
  bool is_loaded() const override {
    return true;
  }
};

class FakeEngine : public LLMEngine {
 public:
  int32_t capacity = 4;
  ETResult<std::unique_ptr<LLMSession>> create_session() override {
    return std::unique_ptr<LLMSession>(new FakeSession());
  }
  LLMServingCapacity serving_capacity() const override {
    return LLMServingCapacity{capacity, 0};
  }
  const std::unordered_map<std::string, int64_t>& metadata() const override {
    return md_;
  }

 private:
  std::unordered_map<std::string, int64_t> md_;
};

// ---- stdout-capturing driver ------------------------------------------------
struct Emitted {
  std::string text; // concatenated {"token": ...} pieces
  nlohmann::json done; // the {"done": true, ...} event
  int token_events = 0;
  bool threw = false;
};

Emitted run(
    WorkerSessionState& st,
    bool warm,
    const nlohmann::json& req,
    const std::unordered_map<std::string, int64_t>& md = {},
    const std::vector<uint64_t>& prefix = {}) {
  static FakeTokenizer tok;
  std::ostringstream cap;
  std::streambuf* old = std::cout.rdbuf(cap.rdbuf());
  Emitted em;
  try {
    worker_handle_request(st, warm, tok, md, req, prefix);
  } catch (const std::exception&) {
    em.threw = true;
  }
  std::cout.rdbuf(old);
  std::istringstream iss(cap.str());
  std::string line;
  while (std::getline(iss, line)) {
    if (line.empty()) {
      continue;
    }
    auto j = nlohmann::json::parse(line);
    if (j.contains("token")) {
      em.text += j["token"].get<std::string>();
      ++em.token_events;
    }
    if (j.contains("done")) {
      em.done = j;
    }
  }
  return em;
}

WorkerSessionState makeState() {
  WorkerSessionState st;
  st.session.reset(new FakeSession());
  return st;
}
FakeSession& fake(WorkerSessionState& st) {
  return *static_cast<FakeSession*>(st.session.get());
}
nlohmann::json idsReq(std::vector<uint64_t> ids, int64_t max_new = 8) {
  return {{"max_new_tokens", max_new}, {"prompt_segments", {{{"ids", ids}}}}};
}

void test_new_full_prefill() {
  auto st = makeState();
  fake(st).steps = {{10, "a", false, false}, {0, "", true, true}};
  auto em = run(st, /*warm=*/true, idsReq({1, 2, 3}));
  check("new: reason=new", em.done["session_reset_reason"] == "new");
  check("new: reset called once", fake(st).reset_calls == 1);
  check(
      "new: full prefill (3)",
      fake(st).prefill_sizes == std::vector<size_t>{3});
  check(
      "new: reused=0 prefilled=3",
      em.done["reused_prompt_tokens"] == 0 &&
          em.done["prefilled_prompt_tokens"] == 3);
  check(
      "new: resident.size()==position()",
      st.resident_token_ids.size() == (size_t)st.session->position());
}

void test_exact_prefix_warm_suffix() {
  auto st = makeState();
  // First turn establishes resident [1,2].
  fake(st).steps = {{0, "", true, true}};
  run(st, true, idsReq({1, 2}));
  size_t resets_after_first = fake(st).reset_calls;
  fake(st).steps = {{0, "", true, true}};
  fake(st).prefill_sizes.clear();
  // Second turn extends to [1,2,3] -> warm suffix prefill of just [3].
  auto em = run(st, true, idsReq({1, 2, 3}));
  check(
      "warm: reason=exact_prefix",
      em.done["session_reset_reason"] == "exact_prefix");
  check(
      "warm: prefill suffix only ([3])",
      fake(st).prefill_sizes == std::vector<size_t>{1});
  check(
      "warm: reused=2 prefilled=1",
      em.done["reused_prompt_tokens"] == 2 &&
          em.done["prefilled_prompt_tokens"] == 1);
  check("warm: no extra reset", fake(st).reset_calls == resets_after_first);
  check(
      "warm: resident.size()==position()",
      st.resident_token_ids.size() == (size_t)st.session->position());
}

void test_mismatch_full_reset() {
  auto st = makeState();
  fake(st).steps = {{0, "", true, true}};
  run(st, true, idsReq({1, 2}));
  fake(st).steps = {{0, "", true, true}};
  fake(st).prefill_sizes.clear();
  auto em = run(st, true, idsReq({1, 9})); // divergent token
  check(
      "mismatch: reason=mismatch",
      em.done["session_reset_reason"] == "mismatch");
  check(
      "mismatch: full prefill (2)",
      fake(st).prefill_sizes == std::vector<size_t>{2});
}

void test_equal_prompt_no_empty_prefill() {
  auto st = makeState();
  fake(st).steps = {{0, "", true, true}};
  run(st, true, idsReq({1, 2, 3}));
  fake(st).steps = {{0, "", true, true}};
  fake(st).prefill_sizes.clear();
  auto em = run(st, true, idsReq({1, 2, 3})); // identical prompt
  check("equal: reason=equal", em.done["session_reset_reason"] == "equal");
  bool any_empty = false;
  for (size_t s : fake(st).prefill_sizes) {
    any_empty = any_empty || (s == 0);
  }
  check("equal: prefill_tokens never called with []", !any_empty);
  check(
      "equal: full reprefill (3)",
      fake(st).prefill_sizes == std::vector<size_t>{3});
}

void test_anonymous_never_warm() {
  auto st = makeState();
  fake(st).steps = {{0, "", true, true}};
  run(st, /*warm=*/false, idsReq({1, 2}));
  fake(st).steps = {{0, "", true, true}};
  fake(st).prefill_sizes.clear();
  // Even though resident now matches a prefix, warm=false forces a full reset.
  auto em = run(st, /*warm=*/false, idsReq({1, 2, 3}));
  check(
      "scratch: reason=new (warm disabled)",
      em.done["session_reset_reason"] == "new");
  check(
      "scratch: full prefill (3)",
      fake(st).prefill_sizes == std::vector<size_t>{3});
}

void test_generated_token_ids_excludes_terminal() {
  auto st = makeState();
  fake(st).steps = {
      {10, "a", false, false}, {11, "b", false, false}, {0, "", true, true}};
  auto em = run(st, true, idsReq({1, 2}));
  check("genids: text=ab", em.text == "ab");
  check("genids: completion_tokens=2", em.done["completion_tokens"] == 2);
  std::vector<uint64_t> ids =
      em.done["generated_token_ids"].get<std::vector<uint64_t>>();
  check(
      "genids: ==[10,11] (terminal EOS excluded)",
      ids == std::vector<uint64_t>{10, 11});
  check("genids: finish=stop (EOS)", em.done["finish_reason"] == "stop");
  check(
      "genids: resident.size()==position()",
      st.resident_token_ids.size() == (size_t)st.session->position());
}

void test_prompt_prefix_ids_prepend_text_prompt_once() {
  auto st = makeState();
  fake(st).steps = {{0, "", true, true}};
  auto em =
      run(st,
          /*warm=*/true,
          {{"max_new_tokens", 1}, {"prompt", "ab"}},
          {},
          {2});
  check("prefix: prompt_tokens includes prefix", em.done["prompt_tokens"] == 3);
  check(
      "prefix: prefilled ids == [2,'a','b']",
      fake(st).prefill_batches ==
          std::vector<std::vector<uint64_t>>{
              {2, static_cast<uint64_t>('a'), static_cast<uint64_t>('b')}});
}

void test_stop_string_marks_dirty_and_omits_ids() {
  auto st = makeState();
  fake(st).steps = {
      {10, "a", false, false},
      {11, "b", false, false},
      {12, "X", false, false}};
  nlohmann::json req = idsReq({1, 2});
  req["stop"] = {"X"};
  auto em = run(st, true, req);
  check("stop: text=ab (stop trimmed)", em.text == "ab");
  check("stop: finish=stop", em.done["finish_reason"] == "stop");
  check(
      "stop: no generated_token_ids", !em.done.contains("generated_token_ids"));
  check("stop: session marked dirty", st.dirty);
}

void test_prefill_failure_marks_dirty() {
  auto st = makeState();
  fake(st).fail_prefill_on = 0;
  auto em = run(st, true, idsReq({1, 2, 3}));
  check("prefill-fail: threw", em.threw);
  check("prefill-fail: dirty", st.dirty);
}

void test_decode_failure_marks_dirty() {
  auto st = makeState();
  fake(st).fail_decode_on = 0;
  auto em = run(st, true, idsReq({1, 2, 3}));
  check("decode-fail: threw", em.threw);
  check("decode-fail: dirty", st.dirty);
}

void test_utf8_split_across_pieces_emits_once_intact() {
  auto st = makeState();
  // "é" = 0xC3 0xA9, split across two decode pieces; must emit once, intact.
  fake(st).steps = {
      {10, std::string("\xC3"), false, false},
      {11, std::string("\xA9"), false, false},
      {0, "", true, true}};
  auto em = run(st, true, idsReq({1}));
  check(
      "utf8: emitted bytes == C3 A9 intact",
      em.text == std::string("\xC3\xA9"));
  check("utf8: not emitted as a partial first byte", em.token_events == 1);
}

void test_stop_straddles_pieces() {
  auto st = makeState();
  // stop "ab" arrives across two pieces "a","b": nothing should be emitted.
  fake(st).steps = {
      {10, "a", false, false},
      {11, "b", false, false},
      {12, "c", false, false}};
  nlohmann::json req = idsReq({1});
  req["stop"] = {"ab"};
  auto em = run(st, true, req);
  check("stop-straddle: nothing emitted", em.text.empty());
  check("stop-straddle: finish=stop", em.done["finish_reason"] == "stop");
  check("stop-straddle: dirty", st.dirty);
}

void test_reset_named_only_clears_on_success() {
  FakeEngine engine;
  WorkerSessions sessions(engine);
  std::string code;
  WorkerSessionState* st = sessions.open_named("s", code);
  check("reset_named: session opened", st != nullptr);
  if (st == nullptr) {
    return;
  }
  st->resident_token_ids = {1, 2, 3};
  auto& s = *static_cast<FakeSession*>(st->session.get());

  // Failed reset: must report error AND leave resident state intact (lockstep).
  s.fail_reset = true;
  ETError err = sessions.reset_named("s");
  check("reset_named: failed reset reports error", err != ETError::Ok);
  check(
      "reset_named: resident intact after failed reset",
      st->resident_token_ids.size() == 3);

  // Successful reset: clears resident state.
  s.fail_reset = false;
  err = sessions.reset_named("s");
  check("reset_named: success reports Ok", err == ETError::Ok);
  check(
      "reset_named: resident cleared after success",
      st->resident_token_ids.empty());

  // Absent session is an idempotent no-op (Ok).
  check(
      "reset_named: absent id is Ok",
      sessions.reset_named("nope") == ETError::Ok);
}

} // namespace

int main() {
  printf("worker_loop.h harness:\n");
  test_new_full_prefill();
  test_exact_prefix_warm_suffix();
  test_mismatch_full_reset();
  test_equal_prompt_no_empty_prefill();
  test_anonymous_never_warm();
  test_generated_token_ids_excludes_terminal();
  test_prompt_prefix_ids_prepend_text_prompt_once();
  test_stop_string_marks_dirty_and_omits_ids();
  test_prefill_failure_marks_dirty();
  test_decode_failure_marks_dirty();
  test_utf8_split_across_pieces_emits_once_intact();
  test_stop_straddles_pieces();
  test_reset_named_only_clears_on_success();
  printf(
      "\n%s (%d failure(s))\n",
      g_failures ? "FAILURES" : "ALL PASS",
      g_failures);
  return g_failures ? 1 : 0;
}
