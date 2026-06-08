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
// qwen3_5_moe_worker). A worker only constructs its engine/session/tokenizer
// and calls run_worker_stdio_loop(); the protocol and the decode loop live here
// once, so protocol changes (e.g. multi-session) land in a single place.
//
// Protocol (one JSON object per line; matches worker_client.py):
//   worker -> stdout, once:         {"ready": true}
//   client -> stdin,  per request:  {"prompt": str, "max_new_tokens": int,
//                                    "temperature": float, "stop": [str, ...]}
//   worker -> stdout, per request:  {"token": str} *   (streamed)
//                                   {"done": true, "prompt_tokens": int,
//                                    "completion_tokens": int,
//                                    "finish_reason": "stop" | "length"}
//                               or  {"error": str}
//
// stdout carries ONLY protocol JSON; all logs go to stderr (ET_LOG). One
// request at a time (the control plane serializes; V1 is one worker == one
// session).

#include <nlohmann/json.hpp>

#include <executorch/extension/llm/runner/constants.h>
#include <executorch/extension/llm/runner/llm_session.h>
#include <executorch/extension/llm/runner/util.h>
#include <pytorch/tokenizers/tokenizer.h>

#include <cstdint>
#include <iostream>
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
  // "length" — it ran to max_new (possibly clamped to the context window).
  worker_emit(
      {{"done", true},
       {"prompt_tokens", num_prompt},
       {"completion_tokens", num_generated},
       {"finish_reason", finish}});
}

// Emit {"ready": true}, then read JSONL requests from stdin and dispatch each
// to worker_handle_request, reporting exceptions as {"error": ...} and
// continuing to serve. Returns 0 when stdin closes.
inline int run_worker_stdio_loop(
    LLMSession& session,
    ::tokenizers::Tokenizer& tokenizer,
    const std::unordered_map<std::string, int64_t>& metadata) {
  worker_emit({{"ready", true}});
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) {
      continue;
    }
    try {
      worker_handle_request(
          session, tokenizer, metadata, nlohmann::json::parse(line));
    } catch (const std::exception& e) { // report and keep serving
      worker_emit({{"error", std::string(e.what())}});
    }
  }
  return 0;
}

} // namespace llm
} // namespace extension
} // namespace executorch
