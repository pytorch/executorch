/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Shared model-worker generation loop + JSONL protocol, used by every model
// worker (the generic text_llm_worker and model-specific workers like
// qwen3_5_moe_worker). A worker only constructs its engine/tokenizer and calls
// run_worker_stdio_loop(); the protocol, session management, and the decode
// loop live here once, so protocol changes land in a single place.
//
// V2a (isolation): the worker owns one LLMEngine (weights loaded once) and
// hands out multiple isolated LLMSessions keyed by session_id, each with its
// own KV/recurrent state, up to the engine's serving capacity. Execution is
// synchronous -- one in-flight request at a time, the control plane serializes.
//
// V2b.1 (warm append-only resume): a named session keeps its decoded context
// across requests. On the next request the worker compares the new prompt's
// token ids against the session's resident token ids; if the resident ids are
// an exact prefix, it prefills ONLY the suffix (continuing the KV/recurrent
// state at pos>0) instead of resetting and re-prefilling the whole prompt. The
// check is exact-token (never string/retokenized text) and falls back to a full
// reset+prefill whenever exact reuse can't be proven, so it is always correct;
// the win is when the prompt is a genuine token extension of the prior turn.
// See plan_prefill().
//
// Sessions:
//   - Named: an explicit session_id -> session + resident token ids, created on
//     first use (or via an `open` op), capped at max_named_sessions = capacity
//     - 1 (the scratch slot is reserved). 0 when the backend hosts one session.
//     Warm resume applies to named sessions (unless disabled).
//   - Scratch: one session for anonymous requests (no session_id), reset every
//     request -- distinct anonymous callers must never reuse each other's
//     state.
//
// Protocol (one JSON object per line; matches worker_client.py):
//   worker -> stdout, once:    {"ready": true, "max_sessions": int,
//                               "max_named_sessions": int}
//   client -> stdin:
//     generate:   {"max_new_tokens": int, "temperature": float,
//                  "stop": [str, ...], "session_id"?: str,
//                  and exactly one prompt form:
//                    "prompt": str
//                    "prompt_segments": [{"text": str} | {"ids": [int, ...]}]}
//     open:       {"op": "open",  "session_id": str}
//     close:      {"op": "close", "session_id": str}
//     reset:      {"op": "reset", "session_id": str}  // clear context, keep
//     slot
//   worker -> stdout:
//     generate:   {"token": str} *   (streamed)
//                 {"done": true, "prompt_tokens": int, "completion_tokens":
//                 int,
//                  "finish_reason": "stop"|"length",
//                  "reused_prompt_tokens": int, "prefilled_prompt_tokens": int,
//                  "session_reset_reason": "new"|"exact_prefix"|"dirty"|
//                                          "mismatch"|"equal",
//                  "generated_token_ids"?: [int, ...]}  // omitted if
//                  stop-trimmed
//     open:       {"opened": true, "session_id": str}
//     close:      {"closed": true, "session_id": str}
//     reset:      {"reset": true,  "session_id": str}
//     error:      {"error": str, "code"?: str}  // code: "capacity_exhausted",
//                                               // "unsupported_session"
//
// stdout carries ONLY protocol JSON; all logs go to stderr (ET_LOG). One
// request at a time (the control plane serializes).

#include <nlohmann/json.hpp>

#include <executorch/extension/llm/runner/constants.h>
#include <executorch/extension/llm/runner/llm_session.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/llm/server/cpp/worker_prefill_plan.h>
#include <pytorch/tokenizers/tokenizer.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace executorch {
namespace extension {
namespace llm {

// Emit one protocol object as a JSON line on stdout. error_handler::replace
// keeps a stray invalid UTF-8 byte (byte-level BPE) from aborting
// serialization.
inline void worker_emit(const nlohmann::json& obj) {
  std::cout << obj.dump(
                   -1, ' ', false, nlohmann::json::error_handler_t::replace)
            << "\n";
  std::cout.flush();
}

// A named session plus the warm-resume bookkeeping the worker maintains for it.
// Invariant (while not mid-mutation): resident_token_ids.size() ==
// session->position() -- the resident ids are exactly the tokens currently in
// the session's KV/recurrent state, in order.
struct WorkerSessionState {
  std::unique_ptr<LLMSession> session;
  std::vector<uint64_t> resident_token_ids;
  // Set when the resident state can no longer be trusted as an exact token
  // prefix (e.g. a stop-string trimmed the emitted text mid-token, or a
  // prefill/decode failed after mutating state). Forces a reset next request.
  bool dirty = false;
};

// One generation request against a session. Encodes the prompt, chooses a
// prefill plan (warm suffix reuse for named sessions, or a full reset+prefill),
// then streams complete-UTF-8 text pieces from decode_one(). A terminal step
// (EOS or cooperative stop) ends generation and is not emitted or counted.
// Maintains st.resident_token_ids / st.dirty. Throws std::runtime_error on
// failure; the caller reports it as {"error": ...}.
inline void worker_handle_request(
    WorkerSessionState& st,
    bool warm,
    ::tokenizers::Tokenizer& tokenizer,
    const std::unordered_map<std::string, int64_t>& metadata,
    const nlohmann::json& req) {
  LLMSession& session = *st.session;
  int64_t max_new = req.value("max_new_tokens", static_cast<int64_t>(-1));
  const float temperature = req.value("temperature", 0.0f);
  // Stop strings (the request's `stop` sequences): terminate at the token
  // boundary where one appears so we don't generate to EOS/max_new past it. The
  // control plane also enforces these as a backstop.
  const std::vector<std::string> stops =
      req.value("stop", std::vector<std::string>{});

  // The prompt is either a single rendered string ("prompt") or an ordered list
  // of segments ("prompt_segments"), each a {"text": ...} chunk to tokenize or
  // a
  // {"ids": [...]} run of literal token ids. Segments let the control plane
  // splice the exact generated token ids of prior assistant turns back in,
  // instead of re-tokenizing the chat template's lossy re-rendering of them (so
  // warm resume can hit on tool-use turns). Text is encoded with no special
  // tokens (already rendered), matching the runner's own encode path.
  const bool has_prompt = req.contains("prompt");
  const bool has_segments = req.contains("prompt_segments");
  if (has_prompt == has_segments) {
    throw std::runtime_error(
        "exactly one of prompt / prompt_segments is required");
  }
  std::vector<uint64_t> ids;
  auto encode_text = [&](const std::string& text) {
    auto enc = tokenizer.encode(text, /*bos=*/0, /*eos=*/0);
    if (!enc.ok()) {
      throw std::runtime_error("prompt encode failed");
    }
    ids.insert(ids.end(), enc->begin(), enc->end());
  };
  if (has_segments) {
    for (const auto& seg : req.at("prompt_segments")) {
      if (seg.contains("ids")) {
        for (const auto& id : seg.at("ids")) {
          ids.push_back(id.get<uint64_t>());
        }
      } else if (seg.contains("text")) {
        encode_text(seg.at("text").get<std::string>());
      } else {
        throw std::runtime_error("prompt_segment needs `text` or `ids`");
      }
    }
  } else {
    encode_text(req.at("prompt").get<std::string>());
  }
  if (ids.empty()) {
    throw std::runtime_error("empty prompt");
  }
  const int64_t num_prompt = static_cast<int64_t>(ids.size());

  // Bound generation to the context window: default to filling the remaining
  // room, and clamp an explicit max_new_tokens too, so decode never steps past
  // the window (which would error mid-generation after partial output). The
  // bound is on the FULL prompt length (= pos after prefill), regardless of how
  // much is reused.
  const auto ctx_it = metadata.find(kMaxContextLen);
  if (ctx_it != metadata.end()) {
    const int64_t room = ctx_it->second - num_prompt;
    if (room <= 0) {
      throw std::runtime_error(
          "prompt fills the context window; no room to generate");
    }
    if (max_new <= 0 || max_new > room) {
      max_new = room;
    }
  } else if (max_new <= 0) {
    max_new = 2048;
  }

  // Decide full vs warm-suffix prefill. Anonymous (scratch) and warm-disabled
  // sessions always full-prefill from a clean state.
  PrefillPlan plan = warm ? plan_prefill(st.resident_token_ids, ids, st.dirty)
                          : PrefillPlan{PrefillPlan::kFull, 0, "new"};
  int64_t reused = 0;
  std::vector<uint64_t> to_prefill;
  if (plan.action == PrefillPlan::kSuffix) {
    reused = static_cast<int64_t>(plan.suffix_start);
    to_prefill.assign(ids.begin() + plan.suffix_start, ids.end());
  } else {
    if (session.reset() != ::executorch::runtime::Error::Ok) {
      st.dirty = true;
      throw std::runtime_error("session reset failed");
    }
    st.resident_token_ids.clear();
    st.dirty = false;
    to_prefill = ids;
  }
  const int64_t prefilled = static_cast<int64_t>(to_prefill.size());

  SamplingConfig sampling;
  sampling.temperature = temperature;
  if (session.prefill_tokens(std::move(to_prefill), &sampling) !=
      ::executorch::runtime::Error::Ok) {
    st.dirty = true; // state may be partially mutated; force a reset next time
    throw std::runtime_error("prefill failed");
  }
  // The resident state now equals the full prompt (resident prefix + prefilled
  // suffix, or the whole prompt). Keep the invariant
  // resident.size()==position().
  st.resident_token_ids = ids;

  std::string buf; // bytes not yet forming a complete UTF-8 prefix
  std::string pending; // complete-UTF-8 text held back for stop-string matching
  int64_t num_generated = 0;
  std::string finish = "length"; // EOS or stop string -> "stop"
  bool stop_string = false; // a request stop string was matched
  for (int64_t step = 0; step < max_new; ++step) {
    auto step_result = session.decode_one(sampling);
    if (step_result.error() != ::executorch::runtime::Error::Ok) {
      st.dirty = true;
      throw std::runtime_error("decode failed");
    }
    const auto& d = step_result.get();
    if (d.is_terminal) {
      finish = "stop";
      break; // terminal step (EOS / cooperative stop): not emitted or counted
    }
    // The token was forwarded into the cache (pos advanced); track it so the
    // resident-ids/position invariant holds. EOS/terminal tokens are not
    // forwarded, so they are not appended (above).
    st.resident_token_ids.push_back(d.token_id);
    ++num_generated;
    buf += d.text_piece;
    const size_t cut = utf8_complete_prefix_len(buf);
    if (cut > 0) {
      pending += buf.substr(0, cut);
      buf.erase(0, cut);
    }
    bool stop_hit = false;
    const size_t safe = stop_safe_prefix_len(pending, stops, stop_hit);
    if (safe > 0) {
      worker_emit({{"token", pending.substr(0, safe)}});
      pending.erase(0, safe);
    }
    if (stop_hit) {
      finish = "stop"; // reached a stop string: drop it and everything after
      stop_string = true;
      // The emitted text was trimmed at the stop string, so the next turn's
      // rendered prompt won't be an exact token extension of resident: force a
      // reset rather than risk a false prefix match.
      st.dirty = true;
      break;
    }
  }
  if (!stop_string) {
    // EOS or length: flush held-back text + any trailing incomplete bytes
    // (replaced if invalid). A stop-string hit drops the remainder instead.
    pending += buf;
    if (!pending.empty()) {
      worker_emit({{"token", pending}});
    }
  }
  // finish_reason: "stop" if the model emitted EOS or hit a stop string, else
  // "length" -- it ran to max_new (possibly clamped to the context window).
  // reused/prefilled sum to prompt_tokens; session_reset_reason explains the
  // prefill plan (for measuring warm-resume hit rate).
  nlohmann::json done = {
      {"done", true},
      {"prompt_tokens", num_prompt},
      {"completion_tokens", num_generated},
      {"finish_reason", finish},
      {"reused_prompt_tokens", reused},
      {"prefilled_prompt_tokens", prefilled},
      {"session_reset_reason", plan.reason}};
  // generated_token_ids = the (non-terminal) tokens made resident this turn,
  // for the control plane to splice back as an `ids` segment. Only emit them
  // when they faithfully decode to the emitted text: a stop-string trim kept
  // the post-stop tokens resident but dropped them from the output, so splicing
  // them would inject text the client never saw. Omitting them makes the
  // control plane record this turn as not resumable (falls back to a text
  // re-render).
  if (!stop_string) {
    done["generated_token_ids"] = std::vector<uint64_t>(
        st.resident_token_ids.end() - num_generated,
        st.resident_token_ids.end());
  }
  worker_emit(done);
}

// Owns the engine's sessions for one worker: named sessions keyed by id plus a
// single scratch session for anonymous requests. Single-threaded (driven by the
// stdio loop), so no internal locking.
class WorkerSessions {
 public:
  explicit WorkerSessions(LLMEngine& engine)
      : engine_(engine),
        // Reserve one capacity slot for the scratch (anonymous) session when
        // the backend can host more than one; a single-session backend hosts
        // only the scratch and reports 0 named sessions.
        max_named_(std::max(
            0,
            engine.serving_capacity()
                    .max_physical_sessions_without_weight_duplication -
                1)) {}

  int32_t max_named() const {
    return max_named_;
  }

  // Resolve (and admit, creating on first use) a named session. Returns nullptr
  // and sets code on failure: "unsupported_session" when the backend hosts no
  // named sessions, "capacity_exhausted" when all named slots are taken.
  WorkerSessionState* open_named(const std::string& id, std::string& code) {
    auto it = named_.find(id);
    if (it != named_.end()) {
      return &it->second; // idempotent open / reuse across requests
    }
    if (max_named_ == 0) {
      code = "unsupported_session";
      return nullptr;
    }
    if (static_cast<int32_t>(named_.size()) >= max_named_) {
      code = "capacity_exhausted";
      return nullptr;
    }
    auto result = engine_.create_session();
    if (result.error() != ::executorch::runtime::Error::Ok) {
      code = "capacity_exhausted"; // engine-side capacity backstop
      return nullptr;
    }
    WorkerSessionState& st = named_[id];
    st.session = std::move(result.get());
    return &st;
  }

  // Destroy a named session (freeing its per-session state); idempotent.
  void close_named(const std::string& id) {
    named_.erase(id);
  }

  // Clear a named session's context (reset KV/recurrent + resident ids) while
  // keeping its capacity slot allocated. No-op if the session doesn't exist.
  // Returns Ok (including the absent no-op); on a failed reset returns the
  // session's error and leaves resident state intact, so the control plane
  // keeps its transcript in lockstep instead of clearing it after a failed
  // reset.
  ::executorch::runtime::Error reset_named(const std::string& id) {
    auto it = named_.find(id);
    if (it == named_.end()) {
      return ::executorch::runtime::Error::Ok;
    }
    auto err = it->second.session->reset();
    if (err != ::executorch::runtime::Error::Ok) {
      return err;
    }
    it->second.resident_token_ids.clear();
    it->second.dirty = false;
    return ::executorch::runtime::Error::Ok;
  }

  // The scratch session for anonymous requests, created on first use. Throws if
  // the engine cannot create it.
  WorkerSessionState* scratch() {
    if (!scratch_.session) {
      auto result = engine_.create_session();
      if (result.error() != ::executorch::runtime::Error::Ok) {
        throw std::runtime_error("failed to create scratch session");
      }
      scratch_.session = std::move(result.get());
    }
    return &scratch_;
  }

 private:
  LLMEngine& engine_;
  int32_t max_named_;
  std::unordered_map<std::string, WorkerSessionState> named_;
  WorkerSessionState scratch_;
};

// Emit {"ready": true, ...}, then read JSONL requests from stdin and dispatch
// each (generate / open / close / reset), reporting exceptions as
// {"error": ...} and continuing to serve. Returns 0 when stdin closes.
// enable_warm_resume gates V2b.1 warm suffix reuse for named sessions (off ->
// every request resets, the V2a behavior; useful for A/B measurement).
inline int run_worker_stdio_loop(
    LLMEngine& engine,
    ::tokenizers::Tokenizer& tokenizer,
    const std::unordered_map<std::string, int64_t>& metadata,
    bool enable_warm_resume = true) {
  WorkerSessions sessions(engine);
  worker_emit(
      {{"ready", true},
       {"max_sessions",
        engine.serving_capacity()
            .max_physical_sessions_without_weight_duplication},
       {"max_named_sessions", sessions.max_named()}});

  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) {
      continue;
    }
    try {
      const nlohmann::json req = nlohmann::json::parse(line);
      const std::string op = req.value("op", std::string{});

      if (op == "open" || op == "close" || op == "reset") {
        const std::string id = req.at("session_id").get<std::string>();
        if (id.empty()) {
          throw std::runtime_error("session_id required for op");
        }
        if (op == "close") {
          sessions.close_named(id);
          worker_emit({{"closed", true}, {"session_id", id}});
        } else if (op == "reset") {
          // idempotent (no-op if absent); only acks success if the reset took
          if (sessions.reset_named(id) != ::executorch::runtime::Error::Ok) {
            worker_emit(
                {{"error", "session reset failed"}, {"session_id", id}});
          } else {
            worker_emit({{"reset", true}, {"session_id", id}});
          }
        } else { // open
          std::string code;
          if (sessions.open_named(id, code) == nullptr) {
            worker_emit(
                {{"error", "cannot open session"},
                 {"code", code},
                 {"session_id", id}});
          } else {
            worker_emit({{"opened", true}, {"session_id", id}});
          }
        }
        continue;
      }

      // Generation. A session_id routes to its named session (admitted on first
      // use, warm-resumable); its absence uses the shared scratch session,
      // which is always reset per request.
      const std::string id = req.value("session_id", std::string{});
      WorkerSessionState* st = nullptr;
      bool warm = false;
      if (id.empty()) {
        st = sessions.scratch();
      } else {
        std::string code;
        st = sessions.open_named(id, code);
        if (st == nullptr) {
          worker_emit(
              {{"error", "cannot open session"},
               {"code", code},
               {"session_id", id}});
          continue;
        }
        warm = enable_warm_resume;
      }
      worker_handle_request(*st, warm, tokenizer, metadata, req);
    } catch (const std::exception& e) { // report and keep serving
      worker_emit({{"error", std::string(e.what())}});
    }
  }
  return 0;
}

} // namespace llm
} // namespace extension
} // namespace executorch
