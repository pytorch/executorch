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
// still synchronous -- one in-flight request at a time, the control plane
// serializes -- so this proves "one model, many isolated contexts without
// duplicating weights", NOT concurrent streaming. It also does NOT yet reuse
// context across requests: worker_handle_request() resets the session at the
// top of every request (warm append-only resume is a follow-up).
//
// Sessions:
//   - Named: an explicit session_id -> LLMSession, created on first use (or via
//     an `open` op), capped at max_named_sessions = capacity - 1 (the scratch
//     slot is reserved). 0 when the backend can host only one session.
//   - Scratch: one session for anonymous requests (no session_id), reset each
//     request -- preserves the original single-session behavior.
//
// Protocol (one JSON object per line; matches worker_client.py):
//   worker -> stdout, once:    {"ready": true, "max_sessions": int,
//                               "max_named_sessions": int}
//   client -> stdin:
//     generate:   {"prompt": str, "max_new_tokens": int, "temperature": float,
//                  "stop": [str, ...], "session_id"?: str}
//     open:       {"op": "open",  "session_id": str}
//     close:      {"op": "close", "session_id": str}
//   worker -> stdout:
//     generate:   {"token": str} *   (streamed)
//                 {"done": true, "prompt_tokens": int,
//                  "completion_tokens": int, "finish_reason": "stop"|"length"}
//     open:       {"opened": true, "session_id": str}
//     close:      {"closed": true, "session_id": str}
//     error:      {"error": str, "code"?: str}  // code: "capacity_exhausted",
//                                               // "unsupported_session"
//
// stdout carries ONLY protocol JSON; all logs go to stderr (ET_LOG). One
// request at a time (the control plane serializes).

#include <nlohmann/json.hpp>

#include <executorch/extension/llm/runner/constants.h>
#include <executorch/extension/llm/runner/llm_session.h>
#include <executorch/extension/llm/runner/util.h>
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

// One generation request: reset the session, encode the prompt, prefill, then
// loop decode_one() streaming complete-UTF-8 text pieces. A terminal step (EOS
// or cooperative stop) ends generation and is not emitted or counted. Throws
// std::runtime_error on failure; the caller reports it as {"error": ...}.
inline void worker_handle_request(
    LLMSession& session,
    ::tokenizers::Tokenizer& tokenizer,
    const std::unordered_map<std::string, int64_t>& metadata,
    const nlohmann::json& req) {
  const std::string prompt = req.at("prompt").get<std::string>();
  int64_t max_new = req.value("max_new_tokens", static_cast<int64_t>(-1));
  const float temperature = req.value("temperature", 0.0f);
  // Stop strings (the request's `stop` sequences): terminate at the token
  // boundary where one appears so we don't generate to EOS/max_new past it. The
  // control plane also enforces these as a backstop.
  const std::vector<std::string> stops =
      req.value("stop", std::vector<std::string>{});

  if (session.reset() != ::executorch::runtime::Error::Ok) {
    throw std::runtime_error("session reset failed");
  }
  // No special tokens: the prompt is already rendered (the control plane
  // applied the chat template), matching the runner's own encode path.
  auto encode_result = tokenizer.encode(prompt, /*bos=*/0, /*eos=*/0);
  if (!encode_result.ok()) {
    throw std::runtime_error("prompt encode failed");
  }
  std::vector<uint64_t> ids = std::move(*encode_result);
  if (ids.empty()) {
    throw std::runtime_error("empty prompt");
  }
  const int64_t num_prompt = static_cast<int64_t>(ids.size());

  // Bound generation to the context window: default to filling the remaining
  // room, and clamp an explicit max_new_tokens too, so decode never steps past
  // the window (which would error mid-generation after partial output).
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

  SamplingConfig sampling;
  sampling.temperature = temperature;
  if (session.prefill_tokens(std::move(ids), &sampling) !=
      ::executorch::runtime::Error::Ok) {
    throw std::runtime_error("prefill failed");
  }

  std::string buf; // bytes not yet forming a complete UTF-8 prefix
  std::string pending; // complete-UTF-8 text held back for stop-string matching
  int64_t num_generated = 0;
  std::string finish = "length"; // EOS or stop string -> "stop"
  bool stop_string = false; // a request stop string was matched
  for (int64_t step = 0; step < max_new; ++step) {
    auto step_result = session.decode_one(sampling);
    if (step_result.error() != ::executorch::runtime::Error::Ok) {
      throw std::runtime_error("decode failed");
    }
    const auto& d = step_result.get();
    if (d.is_terminal) {
      finish = "stop";
      break; // terminal step (EOS / cooperative stop): not emitted or counted
    }
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
  worker_emit(
      {{"done", true},
       {"prompt_tokens", num_prompt},
       {"completion_tokens", num_generated},
       {"finish_reason", finish}});
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
  LLMSession* open_named(const std::string& id, std::string& code) {
    auto it = named_.find(id);
    if (it != named_.end()) {
      return it->second.get(); // idempotent open / reuse across requests
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
    auto* session = result.get().get();
    named_.emplace(id, std::move(result.get()));
    return session;
  }

  // Destroy a named session (freeing its per-session state); idempotent.
  void close_named(const std::string& id) {
    named_.erase(id);
  }

  // The scratch session for anonymous requests, created on first use. Throws if
  // the engine cannot create it.
  LLMSession* scratch() {
    if (!scratch_) {
      auto result = engine_.create_session();
      if (result.error() != ::executorch::runtime::Error::Ok) {
        throw std::runtime_error("failed to create scratch session");
      }
      scratch_ = std::move(result.get());
    }
    return scratch_.get();
  }

 private:
  LLMEngine& engine_;
  int32_t max_named_;
  std::unordered_map<std::string, std::unique_ptr<LLMSession>> named_;
  std::unique_ptr<LLMSession> scratch_;
};

// Emit {"ready": true, ...}, then read JSONL requests from stdin and dispatch
// each (generate / open / close), reporting exceptions as {"error": ...} and
// continuing to serve. Returns 0 when stdin closes.
inline int run_worker_stdio_loop(
    LLMEngine& engine,
    ::tokenizers::Tokenizer& tokenizer,
    const std::unordered_map<std::string, int64_t>& metadata) {
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

      if (op == "open" || op == "close") {
        const std::string id = req.at("session_id").get<std::string>();
        if (id.empty()) {
          throw std::runtime_error("session_id required for op");
        }
        if (op == "close") {
          sessions.close_named(id);
          worker_emit({{"closed", true}, {"session_id", id}});
          continue;
        }
        std::string code;
        if (sessions.open_named(id, code) == nullptr) {
          worker_emit(
              {{"error", "cannot open session"},
               {"code", code},
               {"session_id", id}});
        } else {
          worker_emit({{"opened", true}, {"session_id", id}});
        }
        continue;
      }

      // Generation. A session_id routes to its named session (admitted on first
      // use); its absence uses the shared scratch session.
      const std::string id = req.value("session_id", std::string{});
      LLMSession* session = nullptr;
      if (id.empty()) {
        session = sessions.scratch();
      } else {
        std::string code;
        session = sessions.open_named(id, code);
        if (session == nullptr) {
          worker_emit(
              {{"error", "cannot open session"},
               {"code", code},
               {"session_id", id}});
          continue;
        }
      }
      worker_handle_request(*session, tokenizer, metadata, req);
    } catch (const std::exception& e) { // report and keep serving
      worker_emit({{"error", std::string(e.what())}});
    }
  }
  return 0;
}

} // namespace llm
} // namespace extension
} // namespace executorch
