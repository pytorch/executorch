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
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#endif

using executorch::extension::llm::DecodeResult;
using executorch::extension::llm::LLMEngine;
using executorch::extension::llm::LLMServingCapacity;
using executorch::extension::llm::LLMSession;
using executorch::extension::llm::SamplingConfig;
using executorch::extension::llm::worker_cancel_request_id;
using executorch::extension::llm::worker_handle_request;
using executorch::extension::llm::WorkerCancellationController;
using executorch::extension::llm::WorkerCancellationState;
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
  std::vector<SamplingConfig> prefill_sampling;
  std::vector<SamplingConfig> decode_sampling;
  int reset_calls = 0;
  bool fail_reset = false;
  std::atomic<int> stop_calls{0};
  std::atomic<bool> stop_requested{false};
  std::atomic<bool> block_prefill{false};
  std::atomic<bool> prefill_entered{false};
  std::atomic<bool> release_prefill{false};
  std::atomic<bool> decode_saw_stop{false};

  ETError prefill_tokens(
      const std::vector<uint64_t>& tokens,
      const SamplingConfig* initial_sampling = nullptr) override {
    prefill_sizes.push_back(tokens.size());
    if (initial_sampling != nullptr) {
      prefill_sampling.push_back(*initial_sampling);
    }
    prefill_batches.push_back(tokens);
    if (prefill_calls++ == fail_prefill_on) {
      return ETError::Internal; // failed AFTER (notionally) mutating state
    }
    prefill_entered.store(true, std::memory_order_release);
    while (block_prefill.load(std::memory_order_acquire) &&
           !release_prefill.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    pos += static_cast<int64_t>(tokens.size());
    // Real sessions may clear a cooperative stop while setting up prefill.
    stop_requested.store(false, std::memory_order_release);
    return ETError::Ok;
  }

  ETResult<DecodeResult> decode_one(const SamplingConfig& sampling) override {
    decode_sampling.push_back(sampling);
    if (stop_requested.exchange(false, std::memory_order_acq_rel)) {
      decode_saw_stop.store(true, std::memory_order_release);
      return DecodeResult{0, "", false, true};
    }
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
    stop_requested.store(false, std::memory_order_release);
    return ETError::Ok;
  }
  void stop() override {
    stop_calls.fetch_add(1, std::memory_order_acq_rel);
    stop_requested.store(true, std::memory_order_release);
  }
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
    const std::vector<uint64_t>& prefix = {},
    const nlohmann::json& additional_terminal_stats = nlohmann::json::object(),
    WorkerCancellationState* cancellation = nullptr) {
  static FakeTokenizer tok;
  std::ostringstream cap;
  std::streambuf* old = std::cout.rdbuf(cap.rdbuf());
  Emitted em;
  try {
    worker_handle_request(
        st,
        warm,
        tok,
        md,
        req,
        prefix,
        additional_terminal_stats,
        cancellation);
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

#if defined(__unix__) || defined(__APPLE__)
std::array<uint8_t, sizeof(uint64_t)> cancelFrame(uint64_t request_id) {
  std::array<uint8_t, sizeof(uint64_t)> bytes{};
  for (size_t index = 0; index < bytes.size(); ++index) {
    bytes[index] = static_cast<uint8_t>(request_id >> (index * 8));
  }
  return bytes;
}

bool writeBytes(int descriptor, const uint8_t* bytes, size_t count) {
  return ::write(descriptor, bytes, count) == static_cast<ssize_t>(count);
}

bool writeCancelFrame(int descriptor, uint64_t request_id) {
  const auto bytes = cancelFrame(request_id);
  return writeBytes(descriptor, bytes.data(), bytes.size());
}

template <typename Predicate>
bool waitFor(Predicate predicate) {
  for (int attempt = 0; attempt < 200; ++attempt) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return predicate();
}
#endif

bool sameSampling(
    const SamplingConfig& sampling,
    float temperature,
    float top_p,
    int32_t top_k,
    uint64_t seed) {
  return sampling.temperature == temperature && sampling.top_p == top_p &&
      sampling.top_k == top_k && sampling.seed == seed;
}

void test_sampling_config_forwarded() {
  auto st = makeState();
  fake(st).steps = {{10, "a", false, false}, {0, "", true, true}};
  auto req = idsReq({1, 2, 3});
  req["temperature"] = 0.7;
  req["top_p"] = 0.8;
  req["top_k"] = 32;
  req["seed"] = 123;
  run(st, /*warm=*/false, req);

  check(
      "sampling: explicit config reaches prefill",
      fake(st).prefill_sampling.size() == 1 &&
          sameSampling(fake(st).prefill_sampling[0], 0.7f, 0.8f, 32, 123));
  check(
      "sampling: explicit config reaches every decode",
      fake(st).decode_sampling.size() == 2 &&
          std::all_of(
              fake(st).decode_sampling.begin(),
              fake(st).decode_sampling.end(),
              [](const SamplingConfig& sampling) {
                return sameSampling(sampling, 0.7f, 0.8f, 32, 123);
              }));

  auto defaults = makeState();
  fake(defaults).steps = {{0, "", true, true}};
  run(defaults, /*warm=*/false, idsReq({1}));
  check(
      "sampling: omitted fields use worker defaults",
      fake(defaults).prefill_sampling.size() == 1 &&
          sameSampling(fake(defaults).prefill_sampling[0], 0.0f, 1.0f, 0, 0));
}

void test_invalid_sampling_config_rejected() {
  for (auto invalid : std::vector<nlohmann::json>{
           {{"top_p", 0.0}},
           {{"top_k", -1}},
           {{"top_k",
             static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1}},
           {{"seed", -1}},
       }) {
    auto st = makeState();
    auto req = idsReq({1});
    req.update(invalid);
    const auto em = run(st, /*warm=*/false, req);
    check(
        "sampling: invalid direct worker value rejected before prefill",
        em.threw && fake(st).prefill_calls == 0);
  }
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

void test_cancel_request_id_validation() {
  check(
      "cancel id: legacy request accepted without capability",
      !worker_cancel_request_id(idsReq({1}), false).has_value());
  check(
      "cancel id: positive signed integer accepted",
      worker_cancel_request_id({{"cancel_request_id", 7}}, true) ==
          std::optional<uint64_t>(7));
  check(
      "cancel id: full uint64 range accepted",
      worker_cancel_request_id(
          {{"cancel_request_id", std::numeric_limits<uint64_t>::max()}},
          true) ==
          std::optional<uint64_t>(std::numeric_limits<uint64_t>::max()));

  for (const auto& request : std::vector<nlohmann::json>{
           {{"cancel_request_id", 0}},
           {{"cancel_request_id", -1}},
           {{"cancel_request_id", 1.0}},
           {{"cancel_request_id", "1"}},
           {{"cancel_request_id", true}},
       }) {
    bool threw = false;
    try {
      (void)worker_cancel_request_id(request, true);
    } catch (const std::exception&) {
      threw = true;
    }
    check("cancel id: invalid value rejected", threw);
  }

  bool unsupported_threw = false;
  try {
    (void)worker_cancel_request_id({{"cancel_request_id", 1}}, false);
  } catch (const std::exception&) {
    unsupported_threw = true;
  }
  check("cancel id: capability required", unsupported_threw);
}

void test_cancellation_state_seals_once() {
  auto cancelled_session = makeState();
  WorkerCancellationState cancelled;
  check(
      "cancel state: first request wins",
      cancelled.request_cancel(fake(cancelled_session)));
  check(
      "cancel state: duplicate ignored",
      !cancelled.request_cancel(fake(cancelled_session)));
  check(
      "cancel state: pre-decode request is only latched",
      fake(cancelled_session).stop_calls == 0);
  cancelled.enter_decode(fake(cancelled_session));
  check(
      "cancel state: decode entry delivers stop once",
      fake(cancelled_session).stop_calls == 1);
  check("cancel state: seal reports cancellation", cancelled.seal());
  check(
      "cancel state: late request ignored",
      !cancelled.request_cancel(fake(cancelled_session)));
  check("cancel state: second seal is not cancelled", !cancelled.seal());

  auto completed_session = makeState();
  WorkerCancellationState completed;
  completed.enter_decode(fake(completed_session));
  check("cancel state: clean seal reports completion", !completed.seal());
  check(
      "cancel state: completed request stays sealed",
      !completed.request_cancel(fake(completed_session)));
}

void test_cancelled_terminal_metadata_and_dirty_reset() {
  auto st = makeState();
  fake(st).steps = {{10, "a", false, false}};
  WorkerCancellationState cancellation;
  check(
      "cancel terminal: cancellation requested",
      cancellation.request_cancel(fake(st)));
  auto em =
      run(st,
          true,
          idsReq({1, 2}),
          {},
          {},
          nlohmann::json::object(),
          &cancellation);
  check("cancel terminal: request succeeds", !em.threw);
  check("cancel terminal: stop reasserted once", fake(st).stop_calls == 1);
  check("cancel terminal: decode observes stop", fake(st).decode_saw_stop);
  check(
      "cancel terminal: metadata emitted",
      em.done["finish_reason"] == "stop" && em.done["cancelled"] == true);
  check(
      "cancel terminal: generated ids omitted",
      !em.done.contains("generated_token_ids"));
  check("cancel terminal: session marked dirty", st.dirty);

  fake(st).steps = {{0, "", true, true}};
  fake(st).prefill_sizes.clear();
  auto next = run(st, true, idsReq({1, 2, 3}));
  check(
      "cancel terminal: next request resets dirty session",
      next.done["session_reset_reason"] == "dirty" &&
          fake(st).prefill_sizes == std::vector<size_t>{3} && !st.dirty);
}

#if defined(__unix__) || defined(__APPLE__)
void test_controller_partial_zero_duplicate_and_conflicting_frames() {
  int descriptors[2] = {-1, -1};
  const bool pipe_ok = ::pipe(descriptors) == 0;
  check("controller frames: pipe created", pipe_ok);
  if (!pipe_ok) {
    return;
  }
  {
    WorkerCancellationController controller(descriptors[0]);
    auto active_state = makeState();
    auto other_state = makeState();
    WorkerCancellationState cancellation;
    auto active = controller.activate(42, active_state, cancellation);
    cancellation.enter_decode(fake(active_state));
    check(
        "controller frames: valid descriptor supported",
        controller.supported());

    check(
        "controller frames: zero written", writeCancelFrame(descriptors[1], 0));
    const auto conflicting_frame = cancelFrame(43);
    check(
        "controller frames: seven partial bytes written",
        writeBytes(
            descriptors[1],
            conflicting_frame.data(),
            conflicting_frame.size() - 1));
    check(
        "controller frames: zero consumed before partial assertion",
        waitFor([&]() {
          return controller.processed_frame_count_for_testing() == 1;
        }));
    check(
        "controller frames: zero and partial frame ignored",
        fake(active_state).stop_calls == 0 && !cancellation.cancelled());

    check(
        "controller frames: final conflicting byte written",
        writeBytes(
            descriptors[1],
            conflicting_frame.data() + conflicting_frame.size() - 1,
            1));
    check(
        "controller frames: matching frame written as FIFO barrier",
        writeCancelFrame(descriptors[1], 42));
    check(
        "controller frames: conflict and match consumed in FIFO order",
        waitFor([&]() {
          return controller.processed_frame_count_for_testing() == 3;
        }));
    check(
        "controller frames: only matching frame cancels selected session",
        fake(active_state).stop_calls == 1 && cancellation.cancelled() &&
            fake(other_state).stop_calls == 0);
    active.reset();

    WorkerCancellationState conflicting_cancellation;
    auto conflicting =
        controller.activate(43, other_state, conflicting_cancellation);
    conflicting_cancellation.enter_decode(fake(other_state));
    check(
        "controller frames: active conflict was not queued",
        !conflicting_cancellation.cancelled() &&
            fake(other_state).stop_calls == 0);
    check(
        "controller frames: duplicate matching frames written",
        writeCancelFrame(descriptors[1], 43) &&
            writeCancelFrame(descriptors[1], 43));
    check(
        "controller frames: duplicate matching frames consumed", waitFor([&]() {
          return controller.processed_frame_count_for_testing() == 5;
        }));
    check(
        "controller frames: duplicate matching frames stop exactly once",
        fake(other_state).stop_calls == 1 &&
            conflicting_cancellation.cancelled());
  }
  if (descriptors[1] >= 0) {
    ::close(descriptors[1]);
  }
}

void test_controller_bounded_pending_and_preactivation_match() {
  int descriptors[2] = {-1, -1};
  const bool pipe_ok = ::pipe(descriptors) == 0;
  check("controller pending: pipe created", pipe_ok);
  if (!pipe_ok) {
    return;
  }

  std::array<uint8_t, sizeof(uint64_t) * 8> frames{};
  for (uint64_t request_id = 100; request_id < 108; ++request_id) {
    const auto frame = cancelFrame(request_id);
    const size_t offset = static_cast<size_t>(request_id - 100) * frame.size();
    std::copy(frame.begin(), frame.end(), frames.begin() + offset);
  }
  check(
      "controller pending: burst written before construction",
      writeBytes(descriptors[1], frames.data(), frames.size()));

  {
    WorkerCancellationController controller(descriptors[0]);
    auto first_state = makeState();
    WorkerCancellationState first_cancellation;
    auto first = controller.activate(100, first_state, first_cancellation);
    check(
        "controller pending: prequeued match is latched before decode",
        first_cancellation.cancelled() && fake(first_state).stop_calls == 0);
    first_cancellation.enter_decode(fake(first_state));
    check(
        "controller pending: decode entry invokes stop once",
        fake(first_state).stop_calls == 1);
    first.reset();
    bool stale_request_threw = false;
    try {
      controller.validate_request_id(100);
    } catch (const std::exception&) {
      stale_request_threw = true;
    }
    check(
        "controller pending: completed request ID rejected",
        stale_request_threw);

    auto ignored_state = makeState();
    WorkerCancellationState ignored_cancellation;
    auto ignored =
        controller.activate(107, ignored_state, ignored_cancellation);
    ignored_cancellation.enter_decode(fake(ignored_state));
    check(
        "controller pending: conflicting burst IDs were not accumulated",
        !ignored_cancellation.cancelled() &&
            fake(ignored_state).stop_calls == 0);
    ignored.reset();

    const uint64_t frames_before_stale =
        controller.processed_frame_count_for_testing();
    check(
        "controller pending: stale completed ID written",
        writeCancelFrame(descriptors[1], 100));
    check("controller pending: stale completed ID consumed", waitFor([&]() {
            return controller.processed_frame_count_for_testing() ==
                frames_before_stale + 1;
          }));
    auto next_state = makeState();
    WorkerCancellationState next_cancellation;
    auto next = controller.activate(108, next_state, next_cancellation);
    next_cancellation.enter_decode(fake(next_state));
    check(
        "controller pending: stale ID does not cancel next request",
        !next_cancellation.cancelled() && fake(next_state).stop_calls == 0);
  }
  ::close(descriptors[1]);
}

void test_controller_reasserts_cancel_after_prefill() {
  int descriptors[2] = {-1, -1};
  const bool pipe_ok = ::pipe(descriptors) == 0;
  check("controller prefill: pipe created", pipe_ok);
  if (!pipe_ok) {
    return;
  }
  {
    WorkerCancellationController controller(descriptors[0]);
    auto st = makeState();
    fake(st).block_prefill.store(true);
    WorkerCancellationState cancellation;
    auto active = controller.activate(200, st, cancellation);
    Emitted em;
    std::thread request([&]() {
      em =
          run(st,
              true,
              idsReq({1, 2}),
              {},
              {},
              nlohmann::json::object(),
              &cancellation);
    });
    const bool entered = waitFor([&]() {
      return fake(st).prefill_entered.load(std::memory_order_acquire);
    });
    check("controller prefill: request reached prefill", entered);
    check(
        "controller prefill: matching frame written",
        writeCancelFrame(descriptors[1], 200));
    const bool cancellation_latched =
        waitFor([&]() { return cancellation.cancelled(); });
    check(
        "controller prefill: cancellation is latched without an early stop",
        cancellation_latched && fake(st).stop_calls == 0);
    const uint64_t frames_before_duplicate =
        controller.processed_frame_count_for_testing();
    check(
        "controller prefill: duplicate frame written",
        writeCancelFrame(descriptors[1], 200));
    check("controller prefill: duplicate frame consumed", waitFor([&]() {
            return controller.processed_frame_count_for_testing() ==
                frames_before_duplicate + 1;
          }));
    check(
        "controller prefill: duplicate remains latched without an early stop",
        fake(st).stop_calls == 0);

    fake(st).release_prefill.store(true, std::memory_order_release);
    request.join();
    check(
        "controller prefill: exactly one stop at the decode boundary",
        fake(st).stop_calls == 1);
    check(
        "controller prefill: stop delivered before decode",
        fake(st).stop_calls == 1 && fake(st).decode_saw_stop);
    check(
        "controller prefill: request terminates as cancelled",
        !em.threw && em.done["cancelled"] == true &&
            em.done["finish_reason"] == "stop" && st.dirty);
  }
  ::close(descriptors[1]);
}

void test_controller_seal_blocks_late_cancel_and_next_request() {
  int descriptors[2] = {-1, -1};
  const bool pipe_ok = ::pipe(descriptors) == 0;
  check("controller seal: pipe created", pipe_ok);
  if (!pipe_ok) {
    return;
  }
  {
    WorkerCancellationController controller(descriptors[0]);
    auto completed_state = makeState();
    fake(completed_state).steps = {{0, "", true, true}};
    WorkerCancellationState completed_cancellation;
    auto completed =
        controller.activate(300, completed_state, completed_cancellation);
    const auto em =
        run(completed_state,
            true,
            idsReq({1}),
            {},
            {},
            nlohmann::json::object(),
            &completed_cancellation);
    check(
        "controller seal: normal terminal seals cleanly",
        !em.done.contains("cancelled") && !completed_state.dirty);
    const uint64_t frames_before_late =
        controller.processed_frame_count_for_testing();
    check(
        "controller seal: late matching frame written",
        writeCancelFrame(descriptors[1], 300));
    check("controller seal: late matching frame consumed", waitFor([&]() {
            return controller.processed_frame_count_for_testing() ==
                frames_before_late + 1;
          }));
    check(
        "controller seal: late match cannot stop or dirty completed request",
        fake(completed_state).stop_calls == 0 && !completed_state.dirty);
    completed.reset();

    auto next_state = makeState();
    WorkerCancellationState next_cancellation;
    auto next = controller.activate(301, next_state, next_cancellation);
    next_cancellation.enter_decode(fake(next_state));
    const uint64_t frames_before_stale =
        controller.processed_frame_count_for_testing();
    check(
        "controller seal: stale prior frame written during next request",
        writeCancelFrame(descriptors[1], 300));
    check("controller seal: stale prior frame consumed", waitFor([&]() {
            return controller.processed_frame_count_for_testing() ==
                frames_before_stale + 1;
          }));
    check(
        "controller seal: prior request cannot contaminate next request",
        fake(next_state).stop_calls == 0 && !next_cancellation.cancelled() &&
            !next_state.dirty);
    check(
        "controller seal: matching next frame written",
        writeCancelFrame(descriptors[1], 301));
    check(
        "controller seal: matching active request still cancels",
        waitFor([&]() { return fake(next_state).stop_calls.load() == 1; }) &&
            next_cancellation.cancelled());
  }
  ::close(descriptors[1]);
}

void test_controller_invalid_descriptor_is_unsupported() {
  int descriptors[2] = {-1, -1};
  const bool pipe_ok = ::pipe(descriptors) == 0;
  check("controller invalid fd: pipe created", pipe_ok);
  if (!pipe_ok) {
    return;
  }
  {
    WorkerCancellationController controller(descriptors[1]);
    descriptors[1] = -1; // rejected descriptors are still controller-owned
    check(
        "controller invalid fd: write-only descriptor unsupported",
        !controller.supported());
  }
  ::close(descriptors[0]);

  const int regular_fd = ::open("/dev/null", O_RDONLY);
  check("controller invalid fd: regular descriptor opened", regular_fd >= 0);
  if (regular_fd >= 0) {
    WorkerCancellationController controller(regular_fd);
    check(
        "controller invalid fd: non-pipe descriptor unsupported",
        !controller.supported());
  }
}

void test_controller_shutdown_joins_with_open_writer() {
  int descriptors[2] = {-1, -1};
  const bool pipe_ok = ::pipe(descriptors) == 0;
  check("controller shutdown: pipe created", pipe_ok);
  if (!pipe_ok) {
    return;
  }
  {
    WorkerCancellationController controller(descriptors[0]);
    check("controller shutdown: descriptor supported", controller.supported());
  }
  check(
      "controller shutdown: writer remains independently closable",
      ::close(descriptors[1]) == 0);
}
#else
void test_non_posix_controller_is_unsupported() {
  WorkerCancellationController controller(-1);
  check("controller: non-POSIX build is unsupported", !controller.supported());
}
#endif

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

void test_additional_terminal_stats() {
  auto st = makeState();
  fake(st).steps = {{10, "a", false, false}};
  auto em = run(st, false, idsReq({1}), {}, {}, {{"vision_encoder_ms", 12.5}});
  check("terminal stats: request succeeds", !em.threw);
  check(
      "terminal stats: custom field emitted",
      em.done["vision_encoder_ms"] == 12.5);
}

void test_additional_terminal_stats_reject_reserved_key() {
  auto st = makeState();
  auto em = run(st, false, idsReq({1}), {}, {}, {{"prefill_ms", 12.5}});
  check("terminal stats: reserved key rejected", em.threw);
  check("terminal stats: no partial terminal event", em.done.is_null());
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
  test_sampling_config_forwarded();
  test_invalid_sampling_config_rejected();
  test_new_full_prefill();
  test_exact_prefix_warm_suffix();
  test_mismatch_full_reset();
  test_equal_prompt_no_empty_prefill();
  test_anonymous_never_warm();
  test_generated_token_ids_excludes_terminal();
  test_prompt_prefix_ids_prepend_text_prompt_once();
  test_stop_string_marks_dirty_and_omits_ids();
  test_cancel_request_id_validation();
  test_cancellation_state_seals_once();
  test_cancelled_terminal_metadata_and_dirty_reset();
#if defined(__unix__) || defined(__APPLE__)
  test_controller_partial_zero_duplicate_and_conflicting_frames();
  test_controller_bounded_pending_and_preactivation_match();
  test_controller_reasserts_cancel_after_prefill();
  test_controller_seal_blocks_late_cancel_and_next_request();
  test_controller_invalid_descriptor_is_unsupported();
  test_controller_shutdown_joins_with_open_writer();
#else
  test_non_posix_controller_is_unsupported();
#endif
  test_prefill_failure_marks_dirty();
  test_decode_failure_marks_dirty();
  test_utf8_split_across_pieces_emits_once_intact();
  test_stop_straddles_pieces();
  test_additional_terminal_stats();
  test_additional_terminal_stats_reject_reserved_key();
  test_reset_named_only_clears_on_success();
  printf(
      "\n%s (%d failure(s))\n",
      g_failures ? "FAILURES" : "ALL PASS",
      g_failures);
  return g_failures ? 1 : 0;
}
